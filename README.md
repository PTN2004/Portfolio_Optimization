
# Tối ưu Danh mục Đầu tư bằng Deep Reinforcement Learning (PPO + Transformer)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pytorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-RL-green)
![HUTECH](https://img.shields.io/badge/HUTECH-University-red)

## 📖 Giới thiệu

Dự án này là **Đồ án Chuyên ngành** ngành Công nghệ Thông tin (Học máy và ứng dụng) tại trường Đại học Công nghệ TP.HCM (HUTECH).

Mục tiêu của dự án là xây dựng một hệ thống tối ưu hóa danh mục đầu tư tự động trên thị trường chứng khoán Việt Nam (VN30), sử dụng kết hợp giữa thuật toán **Proximal Policy Optimization (PPO)** và kiến trúc mạng **Transformer**. Mô hình được thiết kế để tự động học chiến lược phân bổ tài sản nhằm tối đa hóa lợi nhuận đã điều chỉnh rủi ro (Risk-adjusted Return) trong bối cảnh thị trường phi tuyến và biến động mạnh.

**Sinh viên thực hiện:** Phạm Ngọc Tú  
**Giảng viên hướng dẫn:** ThS. Nguyễn Hữu Trung

---

## 🚀 Tính năng nổi bật

* **Mô hình cốt lõi:** Sử dụng thuật toán PPO (Proximal Policy Optimization) - một trong những thuật toán SOTA của Reinforcement Learning.
* **Kiến trúc mạng:** Tích hợp **Transformer Encoder** làm Policy Network để trích xuất đặc trưng chuỗi thời gian dài hạn và các mối quan hệ phi tuyến giữa các tài sản.
* **Quản trị rủi ro:** Hàm thưởng (Reward Function) được thiết kế thông minh, phạt nặng các mức sụt giảm tài sản (Max Drawdown) để bảo vệ vốn.
* **Dữ liệu thực tế:** Huấn luyện và kiểm thử trên dữ liệu lịch sử của rổ cổ phiếu **VN30** (2010 - 2025).
* **Backtesting:** Hệ thống đánh giá toàn diện với các chỉ số tài chính chuyên sâu (Sharpe Ratio, Calmar Ratio, MDD).

---

## 🛠️ Kiến trúc hệ thống

Mô hình hoạt động dựa trên quy trình tương tác giữa Agent và Environment:

1.  **Input (State):** Tensor 3 chiều `(Window Size × Max Assets × Features)` bao gồm Giá (OHLC), Volume, và Giá trị giao dịch trong 30 phiên gần nhất.
2.  **Network:**
    * **Feature Extractor:** Transformer Encoder (Multi-Head Self-Attention) giúp nắm bắt xu hướng thị trường.
    * **Policy Network:** Đưa ra hành động là vector trọng số danh mục (Portfolio Weights).
    * **Value Network:** Ước lượng giá trị trạng thái để tính toán Advantage (GAE).
3.  **Action:** Phân bổ tỷ trọng tài sản (đã chuẩn hóa, không bán khống).
4.  **Reward:** Log-return của danh mục trừ đi chi phí giao dịch và phạt drawdown.

---

## 📊 Kết quả thực nghiệm

Mô hình đã được huấn luyện trên 2 triệu bước (timesteps) và kiểm thử trên tập dữ liệu từ 01/01/2023 đến 10/11/2025. Kết quả cho thấy sự vượt trội về quản lý rủi ro so với thị trường chung.

### So sánh hiệu suất (Test Set)

| Chỉ tiêu | AI Agent (PPO + Transformer) | Benchmark (VN30) | Equal-Weight |
| :--- | :---: | :---: | :---: |
| **Lợi nhuận năm (ARR)** | `24.88%` | 29.76% | 25.38% |
| **Max Drawdown (MDD)** | **-11.35%** | -16.22% | -18.09% |
| **Sharpe Ratio** | **1.323** | 1.500 | 1.340 |
| **Calmar Ratio** | **2.193** | 1.835 | 1.403 |

> **Nhận xét:** AI Agent có mức sụt giảm tài sản (Max Drawdown) thấp nhất và chỉ số Calmar Ratio cao nhất, cho thấy khả năng bảo toàn vốn và hiệu quả đầu tư bền vững hơn so với việc nắm giữ VN30 thụ động.

---

## 📂 Cấu trúc dự án


```

├── data/                   # Dữ liệu chứng khoán VN30 (Raw & Processed)
├── envs/                   # Môi trường Gym tủy chỉnh cho Trading
│   └── environment.py
├── models/                 # Kiến trúc mạng PPO và Transformer
│   └── transformer_policy.py
├── notebooks/              # Jupyter Notebooks phân tích và trực quan hóa
├── train.py                # Script huấn luyện mô hình
├── backtest.py             # Script kiểm thử và đánh giá
├── requirements.txt        # Các thư viện cần thiết
└── README.md

```

---

## ⚙️ Cài đặt và Sử dụng

### 1. Yêu cầu hệ thống
* Python 3.8 trở lên
* Thư viện: PyTorch, Stable-Baselines3, Pandas, Numpy, Gymnasium.

### 2. Cài đặt

```bash
# Clone repository
git clone [https://github.com/username/portfolio-optimization-drl.git](https://github.com/username/portfolio-optimization-drl.git)
cd portfolio-optimization-drl

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

```

### 3. Huấn luyện mô hình

Để bắt đầu huấn luyện mô hình PPO với Transformer policy:

```bash
python train.py --timesteps 2000000

```

### 4. Backtest

Để xem kết quả giao dịch trên tập Test:

```bash
python backtest.py --model_path logs/best_model.zip

```

---

## 📚 Tài liệu tham khảo

Dự án được xây dựng dựa trên các nghiên cứu nền tảng:

1. *Proximal Policy Optimization Algorithms* - Schulman et al. (2017).
2. *Attention is All You Need* - Vaswani et al. (2017).
3. *Deep Reinforcement Learning in Finance* - Ye et al. (2020).
4. Tài liệu thực hiện đồ án tốt nghiệp của SV Phạm Ngọc Tú - HUTECH (2025).

---

## 📝 Liên hệ

Mọi thắc mắc hoặc đóng góp cho dự án, vui lòng liên hệ:

* **Phạm Ngọc Tú**
* Email: [Email của bạn]
* GitHub: [Link GitHub của bạn]

### Một số lưu ý để file README đẹp hơn:

1.  **Ảnh minh họa:** Trong báo cáo của bạn có **Ảnh 3.2.1 (Biểu đồ tăng trưởng tài sản)**. Bạn nên chụp màn hình biểu đồ đó (file ảnh), lưu vào thư mục `images/` trong dự án và chèn vào file README (ngay dưới phần Kết quả thực nghiệm) bằng cú pháp: `![Kết quả Backtest](images/chart_result.png)`. Nó sẽ làm dự án trông rất thuyết phục.
2.  **Link Github:** Nhớ thay thế các placeholder `username` và `Link GitHub của bạn` bằng đường dẫn thật.
3.  **Email:** Điền email thật nếu bạn muốn người khác liên hệ (ví dụ nhà tuyển dụng).

