import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class TradingEvaluator:
    def __init__(self, portfolio_values, weights, dates, benchmark_values=None, risk_free_rate=0.0):
        """
        portfolio_values: List hoặc array giá trị danh mục theo ngày
        weights: List hoặc array các trọng số [num_days, num_assets]
        dates: List các ngày giao dịch
        benchmark_values: (Optional) Giá trị của danh mục đối chứng (VN30)
        """
        self.dates = pd.to_datetime(dates)
        self.portfolio_series = pd.Series(portfolio_values, index=self.dates)
        self.returns = self.portfolio_series.pct_change().fillna(0)
        
        # Xử lý weights
        self.weights_df = pd.DataFrame(weights, index=self.dates)
        # Đảm bảo weights không có NaN
        self.weights_df = self.weights_df.fillna(0)
        
        self.benchmark_series = None
        if benchmark_values is not None:
            self.benchmark_series = pd.Series(benchmark_values, index=self.dates)
            self.benchmark_returns = self.benchmark_series.pct_change().fillna(0)
            
        self.rf = risk_free_rate
        self.metrics = {}

    def calculate_level_1_metrics(self):
        """Cấp độ 1: Hiệu quả Tài chính & Rủi ro"""
        total_return = (self.portfolio_series.iloc[-1] / self.portfolio_series.iloc[0]) - 1
        
        # Annualized Return
        n_years = len(self.portfolio_series) / 252
        ann_return = (1 + total_return) ** (1 / n_years) - 1
        
        # Volatility
        ann_vol = self.returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe = (ann_return - self.rf) / (ann_vol + 1e-8)
        
        # Sortino Ratio (Chỉ tính biến động xấu)
        downside_returns = self.returns[self.returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (ann_return - self.rf) / (downside_std + 1e-8)
        
        # Max Drawdown
        rolling_max = self.portfolio_series.cummax()
        drawdown = (self.portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        calmar = ann_return / abs(max_drawdown + 1e-8)

        self.metrics.update({
            "Total Return": total_return,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown": max_drawdown,
            "Calmar Ratio": calmar
        })
        return self.metrics

    def calculate_level_2_behavior(self):
        """Cấp độ 2: Phân tích Hành vi (Turnover & Concentration)"""
        # 1. Turnover (Tần suất đảo danh mục)
        # Công thức: 1/2 * Sum(|weight_t - weight_{t-1}|)
        weight_change = self.weights_df.diff().abs().sum(axis=1)
        avg_daily_turnover = weight_change.mean() / 2.0
        
        # 2. Cash Utilization (Tỷ lệ giữ tiền)
        # Giả sử tổng weight < 1 thì phần còn lại là Cash
        total_asset_weight = self.weights_df.sum(axis=1)
        avg_cash_position = 1.0 - total_asset_weight.mean()
        
        # 3. Max Concentration (Tỷ trọng lớn nhất vào 1 mã)
        max_concentration = self.weights_df.max(axis=1).mean()

        self.metrics.update({
            "Avg Daily Turnover": avg_daily_turnover,
            "Avg Cash Position": avg_cash_position,
            "Avg Max Asset Weight": max_concentration
        })

    def check_sanity_level_4(self):
        """Cấp độ 4: Kiểm tra lỗi & Độ tin cậy"""
        # 1. Kiểm tra độ mượt của đường cong vốn (R-squared của log equity)
        # Nếu R^2 quá gần 1 (> 0.99) mà không có drawdown -> Nghi ngờ Look-ahead bias
        y = np.log(self.portfolio_series.values)
        x = np.arange(len(y))
        correlation_matrix = np.corrcoef(x, y)
        r_squared = correlation_matrix[0, 1]**2
        
        warning = "PASS"
        if r_squared > 0.98 and self.metrics["Sharpe Ratio"] > 3.5:
            warning = "NGUY HIỂM: Có khả năng bị Look-ahead Bias (Kết quả quá hoàn hảo)"
            
        self.metrics.update({
            "Equity Curve Linearity (R^2)": r_squared,
            "Sanity Check": warning
        })

    def print_report(self):
        self.calculate_level_1_metrics()
        self.calculate_level_2_behavior()
        self.check_sanity_level_4()
        
        print("\n" + "="*40)
        print(" BÁO CÁO ĐÁNH GIÁ CHIẾN LƯỢC (4 CẤP ĐỘ)")
        print("="*40)
        
        print(f"\n--- CẤP ĐỘ 1: HIỆU QUẢ TÀI CHÍNH ---")
        print(f"Total Return:      {self.metrics['Total Return']*100:.2f}%")
        print(f"Ann. Return:       {self.metrics['Annualized Return']*100:.2f}%")
        print(f"Max Drawdown:      {self.metrics['Max Drawdown']*100:.2f}%")
        print(f"Sharpe Ratio:      {self.metrics['Sharpe Ratio']:.2f} (Recommended > 1.0)")
        print(f"Sortino Ratio:     {self.metrics['Sortino Ratio']:.2f} (Downside risk only)")
        print(f"Calmar Ratio:      {self.metrics['Calmar Ratio']:.2f}")

        print(f"\n--- CẤP ĐỘ 2: HÀNH VI GIAO DỊCH ---")
        print(f"Avg Turnover/Day:  {self.metrics['Avg Daily Turnover']*100:.2f}%")
        print(f" -> Ý nghĩa: Thay đổi {self.metrics['Avg Daily Turnover']*100:.2f}% danh mục mỗi ngày.")
        print(f"Avg Cash (Tiền):   {self.metrics['Avg Cash Position']*100:.2f}%")
        print(f"Concentration:     {self.metrics['Avg Max Asset Weight']*100:.2f}% (Trung bình dồn vào mã to nhất)")

        print(f"\n--- CẤP ĐỘ 4: SANITY CHECK (KIỂM TRA LỖI) ---")
        print(f"Độ mượt (R^2):     {self.metrics['Equity Curve Linearity (R^2)']:.4f}")
        print(f"ĐÁNH GIÁ:          {self.metrics['Sanity Check']}")
        print("="*40 + "\n")

    def plot_dashboard(self, save_path="comprehensive_evaluation.png"):
        """Vẽ biểu đồ Dashboard"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2)

        # Plot 1: Equity Curve & Drawdown
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.portfolio_series, label='AI Agent', color='blue', linewidth=2)
        if self.benchmark_series is not None:
            ax1.plot(self.benchmark_series, label='Benchmark (VN30)', color='gray', linestyle='--', alpha=0.7)
        ax1.set_title("Cấp độ 1: Tăng trưởng Tài sản & So sánh Benchmark")
        ax1.set_ylabel("Giá trị (VND)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Underwater Plot (Drawdown)
        ax2 = fig.add_subplot(gs[1, 0])
        rolling_max = self.portfolio_series.cummax()
        drawdown = (self.portfolio_series - rolling_max) / rolling_max
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.plot(drawdown, color='red', linewidth=1)
        ax2.set_title(f"Max Drawdown: {drawdown.min()*100:.2f}%")
        ax2.set_ylabel("Sụt giảm (%)")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Rolling Sharpe Ratio (Stability - Cấp độ 3)
        ax3 = fig.add_subplot(gs[1, 1])
        # Rolling 6 tháng (126 ngày giao dịch)
        rolling_sharpe = self.returns.rolling(window=126).mean() / (self.returns.rolling(window=126).std() + 1e-8) * np.sqrt(252)
        ax3.plot(rolling_sharpe, color='green')
        ax3.axhline(1.0, color='black', linestyle='--', linewidth=1)
        ax3.set_title("Cấp độ 3: Tính Ổn định (Rolling 6-month Sharpe)")
        ax3.set_ylabel("Sharpe Ratio")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Asset Allocation Area Chart (Behavior - Cấp độ 2)
        ax4 = fig.add_subplot(gs[2, :])
        
        # Thêm Cash vào biểu đồ
        plot_weights = self.weights_df.copy()
        plot_weights['CASH'] = 1.0 - plot_weights.sum(axis=1)
        # Clip để tránh lỗi hiển thị nếu tổng > 1 một chút do làm tròn
        plot_weights = plot_weights.clip(0, 1)
        
        ax4.stackplot(plot_weights.index, plot_weights.T, labels=plot_weights.columns, alpha=0.8)
        ax4.set_title("Cấp độ 2: Phân bổ Tài sản & Tiền mặt")
        ax4.set_ylabel("Tỷ trọng")
        ax4.set_ylim(0, 1)
        # Chỉ hiện legend nếu số lượng mã ít, nếu nhiều quá thì thôi
        if len(plot_weights.columns) <= 15:
            ax4.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Đã lưu biểu đồ đánh giá tại: {save_path}")
        plt.show()