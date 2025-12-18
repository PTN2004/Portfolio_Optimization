import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque

class TrainLogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainLogCallback, self).__init__(verbose)
        self.recent_rewards = deque(maxlen=100)
        self.recent_values = deque(maxlen=100)

    def _on_step(self) -> bool:
        
        if self.locals['dones'][0]:
            info = self.locals["infos"][0]
            
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_final_value = info.get("portfolio_value", 0)
                
                self.recent_rewards.append(ep_reward)
                self.recent_values.append(ep_final_value)
                
                if len(self.recent_rewards) > 0 and len(self.recent_rewards) % 10 == 0:
                    avg_rew = np.mean(self.recent_rewards)
                    avg_val = np.mean(self.recent_values)
                    
                    print(f"  [Train Log] Buoc: {self.num_timesteps} | " +
                          f"Ep Reward: {ep_reward:,.2f} | " +
                          f"Ep Final Value: {ep_final_value:,.2f} | " +
                          f"Avg100 Reward: {avg_rew:,.2f}")

        return True