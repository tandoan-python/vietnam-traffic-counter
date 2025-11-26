# Xây dựng ứng dụng Thống kê & Phân tích Lưu lượng Giao thông tại Việt Nam sử dụng YOLOv11 & Multi-Object Tracking

## 1. Giới thiệu (Overview)

* Dự án này là một ứng dụng Web phục vụ việc thống kê và phân tích lưu lượng phương tiện giao thông, được tối ưu hóa cho điều kiện giao thông hỗn hợp tại Việt Nam (xe máy, ô tô, xe tải chen chúc).
* Hệ thống sử dụng mô hình Deep Learning YOLOv11m kết hợp với các thuật toán theo vết đa đối tượng (Multi-Object Tracking) để giải quyết bài toán đếm xe tích lũy (Cumulative Counting) với độ chính xác cao.

## 2. Tính năng

1. **Nhận diện đa phương tiện:** Phát hiện chính xác các loại xe: Ô tô, Xe máy, Xe tải, Xe buýt... dựa trên mô hình YOLOv11 đã được fine-tune.
2. **Vùng đếm tùy chỉnh:** Người dùng có thể tự vẽ vùng quan tâm (Entry Zone) trực tiếp trên giao diện để đếm xe đi vào khu vực cụ thể.
3. **Đếm tích lũy thông minh:** Sử dụng logic kiểm tra ID duy nhất (Unique ID).

   *Đảm bảo đếm tổng số xe đi vào (IN) chính xác, không bị phụ thuộc vào việc xe có còn trong khung hình hay không.*
4. **So sánh thuật toán Tracking:** Tích hợp tùy chọn chuyển đổi linh hoạt giữa:

* ***ByteTrack:*** Tốc độ cao, phù hợp video ít che khuất.
* ***BoT-SORT:*** Độ chính xác cao, xử lý tốt tình trạng che khuất và chuyển động phức tạp.

5. **Báo cáo trực quan:** Hiển thị video kết quả song song với biểu đồ thống kê (Bar Chart) thời gian thực.
6. **Tương thích Web:** Hệ thống tự động mã hóa video chuẩn H.264 để xem mượt mà trên mọi trình duyệt.

## 3. Dataset

Dự án huấn luyện Yolo11m dựa trên dataset UIT-ADrone. Website: [UIT-ADrone DATASET](https://https://uit-together.github.io/datasets/UIT-ADrone/)

Cite: T. M. Tran, T. N. Vu, T. V. Nguyen and K. Nguyen, "UIT-ADrone: A Novel Drone Dataset for Traffic Anomaly Detection," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, doi: 10.1109/JSTARS.2023.3285905.

## 4. Công nghệ sử dụng

Hệ thống được xây dựng dựa trên kiến trúc Microservice đơn giản hóa, tách biệt giữa Core AI và Giao diện người dùng.

### 1. Nhóm Lõi AI & Computer Vision

* `ultralytics` (YOLOv11): Phát hiện vật thể (Object Detection) & Gán ID (Object Tracking). tích hợp sẵn (ByteTrack, BoT-SORT)
* `supervision` (by Roboflow): Bộ công cụ xử lý hậu kỳ (Post-processing).
  * `PolygonZone`: Tính toán toán học để xác định tâm xe nằm trong hay ngoài vùng vẽ.
  * `Annotators`: Vẽ Bounding Box, Nhãn (Label) và Vùng đếm lên video đầu ra.
* `cv2` (OpenCV): Đọc/Ghi video (Video I/O). Xử lý ảnh cơ bản (Resize, Color conversion) và vẽ các thông tin Text lên frame.

### 2. Nhóm Giao diện & Tương tác (Frontend & UI)

* `streamlit`: Framework chính để xây dựng Web App, quản lý luồng dữ liệu và giao diện.
* `streamlit_drawable_canvas`: Cho phép người dùng vẽ tương tác (Interactive Drawing) lên frame video để xác định vùng đếm.
* `PIL` (Pillow): Xử lý ảnh tĩnh để hiển thị nền cho Canvas vẽ.

### 3. Nhóm Xử lý Dữ liệu & Hệ thống (Data & System)

* `numpy`: Xử lý các ma trận ảnh và tính toán tọa độ vùng vẽ.
* `pandas`: Tổng hợp số liệu đếm được thành bảng (DataFrame) để vẽ biểu đồ.
* `subprocess` (FFmpeg): Gọi phần mềm FFmpeg để chuyển đổi (Transcode) video đầu ra sang chuẩn H.264/MP4, giúp video hiển thị được trên trình duyệt Web (Chrome/Edge/Safari).
* `tempfile`: Quản lý việc lưu trữ tạm thời video upload và video xử lý.

## 5. Luồng hoạt động (System Workflow)

**1. Input:** Người dùng tải Video lên hệ thống.

**2. Setup:** Hệ thống hiển thị frame đầu tiên, người dùng vẽ vùng đếm (Polygon) và chọn thuật toán (ByteTrack/BoT-SORT).

**3. AI Processing (Vòng lặp từng frame):**

* `YOLOv11` phát hiện đối tượng & gán ID (Tracking).
* Hệ thống kiểm tra ID so với vùng vẽ (`Zone Trigger`).
* Nếu xe vào vùng & ID chưa tồn tại -> Cộng +1 vào bộ đếm.
* Vẽ kết quả lên frame.

**4. Post-Processing:**

* Video được nối lại bởi `OpenCV`.
* `FFmpeg` nén video sang chuẩn Web.

**5. Output:** Hiển thị Video kết quả và Biểu đồ thống kê chi tiết từng loại xe.

## 6. Cài đặt & Chạy

1. Yêu cầu hệ thống

* Python 3.8 - 3.11 (khuyến nghị 3.10)
* FFmpeg (bắt buộc)

2. Các bước thực hiện

* Clone dự án:
  * `git clone https://github.com/tandoan-python/vietnam-traffic-counter.git`
  * `cd vietnam-traffic-counter`

3. Cài đặt thư viện:

   `pip install -r requirements.txt`
4. Cập nhật file packages.txt (dành cho triển khai trên streamlit)
   `ffmpeg`
5. Tải ffmpeg.exe từ từ: [gyan.dev/ffmpeg/builds](https://https://www.gyan.dev/ffmpeg/builds/) (chọn bản "essentials").
   Giải nén, tìm file `ffmpeg.exe` trong thư mục `bin`
6. Chạy ứng dụng (local):

   `streamlit run app.py`

## 7. Cấu trúc thư mục

```
├── app.py                # Source code chính của ứng dụng
├── requirements.txt      # Danh sách thư viện Python
├── packages.txt          # Danh sách gói hệ thống (cho Streamlit Cloud)
├── yolov11m.pt           # Model YOLOv11 (Weights)
└── README.md             # Tài liệu hướng dẫn
```

## 8. Tác giả

* Dự án: Giải pháp công nghệ cho bài toán giao thông thông minh tại Việt Nam
* Thực hiện: Nhóm UIT2520

Đây là dự án miễn phí dành cho sinh viên/học viên ngành Khoa học máy tính tại Việt Nam.
