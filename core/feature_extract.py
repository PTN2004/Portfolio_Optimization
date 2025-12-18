import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class HybridAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(HybridAttention, self).__init__()
        
        self.temporal_att = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.asset_att = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.fn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_mask=None):
        """
        x: (B, W, A, F)
        time_mask: (B, W, A)
        """
        B, W, A, F = x.shape
        x_temp = x.permute(0, 2, 1, 3).reshape(B * A, W, F)
        
        temporal_kpm = None
        if time_mask is not None:
            temporal_kpm = (~time_mask.bool()).reshape(B * A, W)
            all_masked_rows = temporal_kpm.all(dim=1)
            if all_masked_rows.any():
                temporal_kpm[all_masked_rows, 0] = False

        temp_out, _ = self.temporal_att(x_temp, x_temp, x_temp, key_padding_mask=temporal_kpm)
        temp_out = self.norm1(temp_out + x_temp)
        temp_out = self.dropout(self.fn(temp_out))
        
        # Lấy embedding cuối cùng của chuỗi thời gian
        temp_emb = temp_out[:, -1, :].reshape(B, A, F)

        # Attention giữa các asset
        asset_mask = None
        if time_mask is not None:
            derived = time_mask.any(dim=1)
            asset_mask = (~derived.bool())

        if asset_mask is not None:
            all_masked_rows = asset_mask.all(dim=1)
            if all_masked_rows.any():
                asset_mask[all_masked_rows, 0] = False

        asset_out, _ = self.asset_att(temp_emb, temp_emb, temp_emb, key_padding_mask=asset_mask)
        asset_out = self.norm2(asset_out + temp_emb)
        asset_out = self.dropout(self.fn(asset_out))

        fused = temp_emb + asset_out
        return fused  # (B, A, F)


class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, embed_dim: int = 128):
        super().__init__(observation_space, features_dim=embed_dim)
        self.embed_dim = embed_dim

        self.proj = nn.Linear(observation_space["state"].shape[-1], embed_dim)
        self.hybrid_att = HybridAttention(embed_dim=embed_dim)

    def forward(self, observations: dict) -> torch.Tensor:
        x = observations["state"]
        mask = observations["mask"]

        if x.ndim == 3:
            x = x.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        B, W, A, F = x.shape
        emd = self.proj(x)
        fused = self.hybrid_att(emd, time_mask=mask)  

        if mask is not None:
            asset_mask = mask.any(dim=1).float()  
            summed = (fused * asset_mask.unsqueeze(-1)).sum(dim=1)
            count = asset_mask.sum(dim=1, keepdim=True) + 1e-8
            pooled = summed / count  
        else:
            pooled = fused.mean(dim=1)  

        return pooled  
