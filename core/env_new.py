import gymnasium as gym
import numpy as np
import pandas as pd

class EnvironmentTrading(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        X_features,
        X_prices,
        mask,
        index,
        symbols,
        max_num_assets=100,
        lambda_drawdown=0.5,
        window_size=30,
        initial_balance=1e6,
        transaction_cost=0.001,
    ):
        super(EnvironmentTrading, self).__init__()

        self.X_features = X_features
        self.mask = mask
        self.prices = X_prices
        self.index = index
        self.symbols = symbols

        self.num_assets, self.num_days = X_prices.shape
        _, _, self.num_features = X_features.shape

        self.lambda_drawdown = lambda_drawdown
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.max_num_assets = max_num_assets

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.max_num_assets,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size,
                                                                     self.max_num_assets,
                                                                     self.num_features),
                                    dtype=np.float32),
            "mask": gym.spaces.Box(low=0, high=1, shape=(self.window_size,
                                                         self.max_num_assets),
                                   dtype=np.float32)
        })
        
        if self.num_assets > self.max_num_assets:
            raise ValueError(f"Số lượng tài sản ({self.num_assets}) nhiều hơn mức tối đa ({self.max_num_assets})")

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.num_assets, dtype=np.float32)
        self.peak_value = self.initial_balance
        
        return self._get_obs(), {}

    def _get_obs(self):
        start = self.current_step - self.window_size
        end = self.current_step

        state = self.X_features[:, start:end, :].transpose(1, 0, 2).astype(np.float32)
        mask = self.mask[:, start:end].transpose(1, 0).astype(np.float32)
        state_padded = np.zeros(
            (self.window_size, self.max_num_assets, self.num_features), 
            dtype=np.float32
        )
        mask_padded = np.zeros(
            (self.window_size, self.max_num_assets), 
            dtype=np.float32
        )
        
        state_padded[:, :self.num_assets, :] = state
        mask_padded[:, :self.num_assets] = mask
        return {
            "state" : state_padded,
            "mask" : mask_padded
        }

    def step(self, action):
        action = action[:self.num_assets]
        mask_today = self.mask[:, self.current_step]
        if self.current_step + 1 < self.num_days:
            mask_tomorrow = self.mask[:, self.current_step + 1]
        else:
            mask_tomorrow = np.zeros_like(mask_today)

        tradable_mask = mask_today * mask_tomorrow

        action_positive = np.maximum(action, 0)
        valid_action = action_positive * tradable_mask

        action_sum = np.sum(valid_action)
        if action_sum < 1e-8:
            target_weights = np.zeros_like(action)
        else:
            target_weights = valid_action / action_sum

        prices_today = self.prices[:, self.current_step]
        prices_today[~mask_today.astype(bool)] = 0.0

        current_cash = self.balance
        current_holdings_value = self.holdings * prices_today
        current_portfolio_value = current_cash + np.sum(current_holdings_value)
        
        safe_portfolio_value = max(1e-8, current_portfolio_value)
        current_weights = current_holdings_value / safe_portfolio_value

        trade_delta_weights = target_weights - current_weights
        
        trade_value_per_asset = trade_delta_weights * current_portfolio_value
        
        total_trade_value = np.sum(np.abs(trade_value_per_asset))
        
        transaction_fees = total_trade_value * self.transaction_cost

        target_value_per_asset = current_portfolio_value * target_weights
        
        self.holdings = target_value_per_asset / (prices_today + 1e-8)
        self.holdings = np.nan_to_num(self.holdings, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.balance = current_portfolio_value - np.sum(target_value_per_asset) - transaction_fees
        
        prices_next = self.prices[:, self.current_step + 1]
        prices_next[~mask_tomorrow.astype(bool)] = 0.0

        next_portfolio_value = self.balance + np.sum(self.holdings * prices_next)
        next_portfolio_value = max(0.0, next_portfolio_value)

        self.peak_value = max(self.peak_value, next_portfolio_value)

        reward = np.log((next_portfolio_value + 1e-8) / safe_portfolio_value) * 10.0

        drawdown = (self.peak_value - next_portfolio_value) / (self.peak_value + 1e-8)
        reward -= self.lambda_drawdown * drawdown * 10.0

        asset_returns = (prices_next - prices_today) / (prices_today + 1e-8)
        portfolio_volatility = np.dot(target_weights, np.abs(asset_returns))
        lambda_risk = 0.1 # Hệ số phạt rủi ro biến động
        reward_risk = portfolio_volatility * lambda_risk * 10.0

        reward -= reward_risk
        reward = np.clip(reward, -1.0, 1.0)

        self.current_step += 1
        done = self.current_step >= self.num_days - 2

        obs = self._get_obs()
        
        actual_turnover = np.sum(np.abs(trade_delta_weights)) / 2.0
        
        info = {
            "turnover": actual_turnover, 
            "portfolio_value": next_portfolio_value,
            "transaction_fees": transaction_fees,
            "drawdown": drawdown,
            "date": str(self.index[self.current_step]),
            "weights": target_weights

        }

        if done:
            portfolio_values_ep = [self.initial_balance, float(next_portfolio_value)]
            daily_returns_ep = np.diff(portfolio_values_ep) / (np.array(portfolio_values_ep[:-1]) + 1e-8)

            sharpe_ep = np.mean(daily_returns_ep) / (np.std(daily_returns_ep) + 1e-8) * np.sqrt(252)
            peak_ep = np.maximum.accumulate(portfolio_values_ep)
            drawdown_ep = (peak_ep - portfolio_values_ep) / (peak_ep + 1e-8)
            max_drawdown_ep = np.max(drawdown_ep)

            info["episode_sharpe"] = sharpe_ep
            info["episode_mdd"] = max_drawdown_ep
            
            
        return obs, float(reward), done, False, info

    def render(self, mode="human"):
        current_prices = self.prices[:, self.current_step]
        value = self.balance + np.sum(self.holdings * current_prices)
        weights = (self.holdings * current_prices) / (value + 1e-8)
        date = self.index[self.current_step]
        print(f"[{date.date()}] Portfolio Value: {value:,.2f} \tWeights: {np.round(weights, 2)}")

    def close(self):
        pass