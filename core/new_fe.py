import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class HybridAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.temporal_att = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_ln = nn.LayerNorm(embed_dim)
        self.temporal_ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.asset_att = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.asset_ln = nn.LayerNorm(embed_dim)
        self.asset_ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.time_pos = nn.Parameter(torch.randn(1, 512, embed_dim))   

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.time_pos, std=0.02)

    def forward(self, x, mask=None):
        B, W, A, F = x.shape
        assert F == self.embed_dim, f"Expected feature dim {self.embed_dim}, got {F}"

        x_temp = x.permute(0, 2, 1, 3).reshape(B * A, W, F)  # batch over assets

        if W <= self.time_pos.shape[1]:
            pos = self.time_pos[:, :W, :].repeat(B * A, 1, 1)  # (B*A, W, F)
            x_temp = x_temp + pos
        else:
            pos = self.time_pos.repeat(B * A, (W // self.time_pos.shape[1]) + 1, 1)[:, :W, :]
            x_temp = x_temp + pos

        temporal_kpm = None
        if mask is not None:
            pad_mask = (~mask.bool()).permute(0, 2, 1).reshape(B * A, W)  # (B*A, W)
            temporal_kpm = pad_mask

            all_padded = temporal_kpm.all(dim=1)
            if all_padded.any():
                temporal_kpm[all_padded, 0] = False

        temp_out, _ = self.temporal_att(x_temp, x_temp, x_temp, key_padding_mask=temporal_kpm)
        temp_out = self.temporal_ln(temp_out + x_temp)
        temp_out = self.temporal_ff(temp_out) + temp_out  

        if mask is not None:
            mask_perm = mask.permute(0, 2, 1).reshape(B * A, W)
            last_idx = (mask_perm.sum(dim=1) - 1).long().clamp(min=0)
            temp_emb = temp_out[torch.arange(B * A, device=x.device), last_idx, :].reshape(B, A, F)
        else:
            temp_emb = temp_out[:, -1, :].reshape(B, A, F)
            
        asset_in = temp_emb
        asset_kpm = None
        if mask is not None:
            asset_valid = mask.any(dim=1)  
            asset_kpm = (~asset_valid).bool()  
            all_assets_padded = asset_kpm.all(dim=1)
            if all_assets_padded.any():
                idxs = torch.where(all_assets_padded)[0]
                asset_kpm[idxs, 0] = False

        asset_out, _ = self.asset_att(asset_in, asset_in, asset_in, key_padding_mask=asset_kpm)
        asset_out = self.asset_ln(asset_out + asset_in)
        asset_out = self.asset_ff(asset_out) + asset_out
        fused = torch.cat([temp_emb, asset_out], dim=-1)  # (B, A, 2F)
        fused_proj = self.out_proj(fused)

        return fused_proj  


class FeatureExtractor(BaseFeaturesExtractor):
    
    def __init__(self, observation_space: gym.spaces.Dict, embed_dim: int = 128):
        super().__init__(observation_space, features_dim=embed_dim)
        self.embed_dim = embed_dim

        fin = observation_space["state"].shape[-1]
        self.proj = nn.Linear(fin, embed_dim)
        self.hybrid_att = HybridAttention(embed_dim=embed_dim)
        self.pool_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, observations: dict) -> torch.Tensor:
        x = observations["state"]
        mask = observations.get("mask", None)
        if x.ndim == 3:
           
            x = x.unsqueeze(0)
        if mask is not None and mask.ndim == 2:
            mask = mask.unsqueeze(0)

        B, W, A, F_in = x.shape
        emd = self.proj(x)  

        fused = self.hybrid_att(emd, mask)  

        if mask is not None:
            asset_valid = mask.any(dim=1).float()  
            asset_valid = asset_valid.to(fused.dtype)
            summed = (fused * asset_valid.unsqueeze(-1)).sum(dim=1) 
            count = asset_valid.sum(dim=1, keepdim=True)  
            count = count.clamp(min=1.0)
            pooled = summed / count
        else:
            pooled = fused.mean(dim=1)

        out = self.pool_proj(pooled)  
        return out
