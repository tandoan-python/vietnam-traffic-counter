# import thư viện, sử dụng file requirements.txt để cài đặt
import streamlit as st
import cv2
import tempfile
import numpy as np
import supervision as sv
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import subprocess
import pandas as pd

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="AI Traffic Analytics",
    page_icon="🚗",
    layout="wide"
)

# CSS: Tùy chỉnh giao diện
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #d32f2f; text-align: center; }
    .stButton>button { width: 100%; background-color: #d32f2f; color: white; font-weight: bold; border-radius: 16px;}
    .stat-box { padding: 15px; background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
    .stat-number { font-size: 2em; font-weight: bold; color: #d32f2f; }
    .tracker-info { padding: 10px; background-color: #e3f2fd; color: #0d47a1; border-radius: 5px; margin-bottom: 10px; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# Hàm chuyển đổi video sang H.264 để tương thích web
def convert_video_to_h264(input_path):
    output_path = input_path.replace('.mp4', '_converted.mp4')
    command = ['ffmpeg', '-y', '-i', input_path, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', output_path]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        return input_path

# Tải model với caching để tránh tải lại nhiều lần. 
# @st.cache_resource RẤT QUAN TRỌNG KHÔNG THAY ĐỔI
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Xử lí video đếm phương tiện với tracker được chọn (trên giao diện chọn ByteTrack hoặc BoT-SORT)
def process_video(video_path, model, zone_polygon, tracker_type):
    video_info = sv.VideoInfo.from_video_path(video_path)
    
    # Xác định file config cho tracker (Ultralytics có sẵn 2 file này, hoặc có thể tự cấu hình lại)
    tracker_yaml = "bytetrack.yaml" if tracker_type == "ByteTrack" else "botsort.yaml"
    
    # Khởi tạo Zone: vùng đếm phương tiện
    zone = sv.PolygonZone(
        polygon=zone_polygon, 
        triggering_anchors=(sv.Position.CENTER,) # RẤT QUAN TRỌNG: Chỉ dùng tâm hộp để tránh đếm sai lệch
    )
    
    # Annotators: Vẽ hộp, nhãn, vùng
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=5)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.RED, 
        thickness=2, 
        text_thickness=2, 
        text_scale=1
    )

    # Video Writer để lưu video kết quả
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    raw_output_path = tfile.name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_output_path, fourcc, video_info.fps, video_info.resolution_wh)

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Biến đếm tích lũy
    class_counts = {} 
    counted_ids = set()

    # Lấy generator frame từ video
    frame_generator = sv.get_video_frames_generator(video_path)
    total_frames = video_info.total_frames
    
    # Xử lý từng frame
    for i, frame in enumerate(frame_generator):
        
        # Sử dụng model.track thay vì model.predict: để theo dõi ID
        # persist=True: Nhớ giữ ID giữa các frame, nếu chọn false sẽ mất ID ngay lập tức khi qua frame
        # tracker=tracker_yaml: Chọn thuật toán (ByteTrack/BoT-SORT)
        # verbose=False: Tắt log không cần thiết
        results = model.track(frame, persist=True, tracker=tracker_yaml, verbose=False)[0]
        
        # Chuyển đổi kết quả sang Supervision
        detections = sv.Detections.from_ultralytics(results)
        
        # Lưu ý: Khi dùng model.track, detections.tracker_id có thể là None nếu không track được
        if detections.tracker_id is not None:
            
            # Logic đếm (chỉ chạy khi có ID)
            mask = zone.trigger(detections=detections)
            detections_in_zone = detections[mask]
            
            for tracker_id, class_id in zip(detections_in_zone.tracker_id, detections_in_zone.class_id):
                if tracker_id not in counted_ids:
                    counted_ids.add(tracker_id)
                    class_name = model.names[class_id]
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1
        
            # Vẽ Label kèm ID
            labels = [f"#{tid} {model.names[cid]}" for tid, cid in zip(detections.tracker_id, detections.class_id)]
        else:
            # Nếu không có ID (mất dấu), chỉ hiện tên class
            labels = [f"{model.names[cid]}" for cid in detections.class_id]

        # Vẽ hình ảnh kết quả
        annotated_frame = frame.copy()
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Hiển thị thông tin Tracker và Tổng số
        cv2.putText(annotated_frame, f"Mode: {tracker_type}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Total IN: {len(counted_ids)}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        out.write(annotated_frame)

        # Cập nhật tiến trình mỗi 10 frame, có thể thay đổi 10 thành số khác để mượt hơn
        if i % 10 == 0:
            progress_bar.progress(min(i / total_frames, 1.0))
            status_text.text(f"Đang xử lý frame {i}/{total_frames} | Tracker: {tracker_type} | Count: {len(counted_ids)}")

    out.release()
    progress_bar.progress(1.0)
    status_text.success("✅ Xử lý hoàn tất!")
    
    final_output = convert_video_to_h264(raw_output_path)
    return final_output, class_counts

# Hàm main
def main():
    st.title("🚗 Hệ Thống Thống kê và Phân tích giao thông tại Việt Nam")

    # Cấu hình sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình Model")
        model_source = st.radio("Nguồn Model:", ["Mặc định (yolov11m.pt)", "Chọn Yolo khác (.pt)"])
        model_path = "yolov11m.pt"
        if model_source == "Chọn Yolo khác (.pt)":
            up_model = st.file_uploader("File .pt", type=['pt'])
            if up_model:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                    tmp.write(up_model.read())
                    model_path = tmp.name

        # Cấu hình Tracker
        st.divider()
        st.header("🎯 Cấu hình Tracker")
        tracker_type = st.selectbox(
            "Chọn thuật toán Tracking:",
            ("ByteTrack", "BoT-SORT"),
            help="ByteTrack: Nhanh, nhẹ, dựa trên IoU.\nBoT-SORT: Chính xác hơn khi vật thể bị che khuất, dựa trên ngoại hình (ReID). \nIoU: Viết tắt của Intersection over Union, là phép đo trong xử lý ảnh dựa trên Intersection (phép giao) và Union (phép hợp). Nó là tên gọi của chỉ số Jaccard trong xử lý ảnh. Công thức = Diện tích phần GIAO /Diện tích phần HỢP"
        )
        
        # Hiển thị giải thích ngắn gọn
        if tracker_type == "ByteTrack":
            st.info("**ByteTrack**: Tốc độ xử lý cao (FPS cao), phù hợp video ít che khuất.")
        else:
            st.info("**BoT-SORT**: Tốc độ thấp hơn, nhưng giữ ID tốt hơn khi xe bị che hoặc chuyển động phức tạp.")

    try:
        model = load_model(model_path)
    except:
        st.error("Không tìm thấy Model. Hãy tải lên hoặc đảm bảo file yolov11m.pt có sẵn.")
        st.stop()

    uploaded_video = st.file_uploader("1. Tải lên Video giám sát", type=['mp4', 'avi', 'mov'])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.subheader("2. Hãy bắt đầu vẽ vùng đếm (Entry Zone)")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        zone_polygon = None
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            
            # Tính toán resize (Max width 700px)
            bg_w, bg_h = pil_img.size
            canvas_w = 700
            canvas_h = int(bg_h * (canvas_w / bg_w))
            
            # Resize ảnh về đúng kích thước hiển thị
            pil_img_resized = pil_img.resize((canvas_w, canvas_h))

            canvas = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2, stroke_color="#ff0000",
                background_image=pil_img_resized, # Nếu chạy local, có thể dùng frame trực tiếp, pil_img
                height=canvas_h, width=canvas_w,
                drawing_mode="polygon",
                key="canvas",
            )

            if canvas.json_data and canvas.json_data["objects"]:
                raw_points = canvas.json_data["objects"][-1]["path"]
                if len(raw_points) > 2:
                    
                    # Restore scale
                    scale_x = bg_w / canvas_w
                    scale_y = bg_h / canvas_h
                    
                    # # local version
                    # pts = [[int(p[1]*scale_x), int(p[2]*scale_y)] for p in raw_points if len(p)>=3]
                    
                    # Streamlit cloud version
                    pts = []
                    for p in raw_points:
                        if len(p) >= 3:
                            pts.append([int(p[1] * scale_x), int(p[2] * scale_y)])
                    

                    zone_polygon = np.array(pts)
                    st.success("✅ Đã xác định vùng đếm!")

            if zone_polygon is not None and len(zone_polygon) >= 3:
                if st.button(f"BẮT ĐẦU PHÂN TÍCH ({tracker_type})"):
                    with st.spinner(f'Hệ thống đang xử lí {tracker_type}...'):
                        
                        # Truyền thêm tracker_type vào hàm xử lý
                        video_out, stats = process_video(video_path, model, zone_polygon, tracker_type)
                    
                    st.divider()
                    st.subheader("3. Kết quả Phân tích")
                    
                    col1, col2 = st.columns([1.5, 1])
                    
                    with col1:
                        st.video(video_out)
                    
                    with col2:
                        st.markdown(f"### 📊 Thống kê ({tracker_type})")
                        if stats:
                            df = pd.DataFrame(list(stats.items()), columns=['Loại', 'Số lượng'])
                            total_vehicles = sum(stats.values())
                            st.markdown(f"""
                            <div class="stat-box">
                                <div>TỔNG PHƯƠNG TIỆN ({tracker_type})</div>
                                <div class="stat-number">{total_vehicles}</div>
                            </div>
                            <br>
                            """, unsafe_allow_html=True)
                            
                            st.bar_chart(df.set_index('Loại'), color="#d32f2f")
                            st.dataframe(df, hide_index=True, use_container_width=True)
                        else:
                            st.warning("Chưa có phương tiện nào đi vào vùng.")

if __name__ == "__main__":
    main()