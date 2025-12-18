import torch
import os
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from core.feature_extract import FeatureExtractor
from core.environment import EnvironmentTrading
from core.callbacks import TrainLogCallback
from preprocessing.reprocess_data import listing_symbols, repare_trading_data

def main():
    log_dir = "./logs_sb3/"
    tensorboard_log_dir = "./tensorboard_logs_sb3/"
    best_model_save_path = './best_model/'
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(best_model_save_path, exist_ok=True)

    stats_path = os.path.join(best_model_save_path, "vec_normalize.pkl")
    model_path = os.path.join(best_model_save_path, "best_model_256.zip")
    
    symbols = listing_symbols(group_name="VN30")
    MAX_NUM_ASSETS = 100
    
    X_features_full, X_prices_full, mask_full, index_full, symbols = repare_trading_data(
        symbols,
        start_date="2010-01-01",
        end_date="2025-11-11"
    )
    
    train_end_date = "2020-12-31"
    eval_start_date = "2021-01-01"
    eval_end_date = "2022-12-31"

    train_end_idx = index_full.searchsorted(train_end_date)
    eval_start_idx = index_full.searchsorted(eval_start_date)
    eval_end_idx = index_full.searchsorted(eval_end_date)

    X_features_train = X_features_full[:, :train_end_idx, :]
    X_prices_train = X_prices_full[:, :train_end_idx]
    mask_train = mask_full[:, :train_end_idx]
    index_train = index_full[:train_end_idx]

    X_features_eval = X_features_full[:, eval_start_idx:eval_end_idx, :]
    X_prices_eval = X_prices_full[:, eval_start_idx:eval_end_idx]
    mask_eval = mask_full[:, eval_start_idx:eval_end_idx]
    index_eval = index_full[eval_start_idx:eval_end_idx]
    
    def make_train_env():
        return EnvironmentTrading(
            X_features=X_features_train,
            X_prices=X_prices_train,
            mask=mask_train,
            index=index_train,
            symbols=symbols,
            max_num_assets=MAX_NUM_ASSETS,
            window_size=30,
            lambda_drawdown=0.5,  
            initial_balance=1e6,
            transaction_cost=0.001
        )
        
    def make_eval_env():
        return EnvironmentTrading(
            X_features=X_features_eval,
            X_prices=X_prices_eval,
            mask=mask_eval,
            index=index_eval,
            symbols=symbols,
            max_num_assets=MAX_NUM_ASSETS,
            window_size=30,
            lambda_drawdown=0.5,  
            initial_balance=1e6,
            transaction_cost=0.001
        )

    env = DummyVecEnv([make_train_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path=log_dir,
        eval_freq=max(5000 // 1, 1), 
        deterministic=True,
        render=False
    )
    
    train_callback = TrainLogCallback()
    list_callbacks = [train_callback, eval_callback]
    
    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
        features_extractor_kwargs=dict(
            embed_dim=128 
        ),
        net_arch=dict(pi=[256, 128], vf=[128, 128]) 
    )

    if os.path.exists("/root/Portfolio_Optimatio/best_model_flexible/best_model_256.zip"):
        print(f"--- Dang tai mo hinh da luu tu: /root/Portfolio_Optimatio/best_model_flexible/best_model_256.zip ---")
        # env = VecNormalize.load(stats_path, DummyVecEnv([make_train_env]))
        model = PPO.load("/root/Portfolio_Optimatio/best_model_flexible/best_model_256.zip", env=env, tensorboard_log=tensorboard_log_dir)
        # eval_env.load_running_average(stats_path)
    else:
        print(f"--- Khong tim thay mo hinh. Tao mo hinh moi. ---")
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=tensorboard_log_dir,
            verbose=1
        )

    print("=== Bat dau Huan luyen voi Stable-Baselines3 ===")
    model.learn(
        total_timesteps=1_000_000, 
        callback=list_callbacks
    )
    
    print("\n=== Huan luyen Hoan tat ===")
    print(f"=== Mo hinh tot nhat da duoc luu tai: {best_model_save_path} ===")


if __name__ == "__main__":
    main()