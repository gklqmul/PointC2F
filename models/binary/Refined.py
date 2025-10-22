import torch
import torch.nn as nn
import torch.nn.functional as F

class Refiner(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 回归一个偏移值（或置信度）
        )

    def forward(self, x, coarse_ids):
        # x: [B, T, 3], coarse_ids: [B, K]
        B, T, C = x.shape
        K = coarse_ids.shape[1]
        refined = []

        for b in range(B):
            feats = []
            for k in range(K):
                idx = coarse_ids[b, k].item()
                window = x[b, max(0, idx-2):min(T, idx+3)]  # 5帧窗口
                pooled = window.mean(dim=0)                 # 简单平均池化
                feats.append(pooled)
            feats = torch.stack(feats)  # [K, C]
            refined.append(self.mlp(feats).squeeze(-1))  # [K]
        return torch.stack(refined)  # [B, K]
