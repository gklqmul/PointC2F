import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import fps, radius
from torch_scatter import scatter_mean


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, features):
        # xyz: (B, N, 3)
        # features: (B, C, N)
        B, N, _ = xyz.shape
        new_xyz_list, new_features_list = [], []

        for b in range(B):
            xyz_b = xyz[b]  # (N, 3)
            features_b = features[b] if features is not None else None  # (C, N) or None

            # FPS采样
            batch_b = torch.zeros(N, dtype=torch.long, device=xyz.device)
            fps_idx = fps(xyz_b, batch=batch_b, ratio=self.npoint / N)
            new_xyz_b = xyz_b[fps_idx]  # (npoint, 3)

            # 距离和分组
            dists = torch.cdist(new_xyz_b, xyz_b)  # (npoint, N)
            group_idx = dists.topk(self.nsample, largest=False)[1]  # (npoint, nsample)

            grouped_xyz = xyz_b[group_idx] - new_xyz_b.unsqueeze(1)  # (npoint, nsample, 3)
            if features_b is not None:
                grouped_features = features_b.transpose(0, 1)[group_idx]  # (npoint, nsample, C)
                new_features = torch.cat([grouped_xyz, grouped_features], dim=-1)  # (npoint, nsample, 3+C)
            else:
                new_features = grouped_xyz

            new_features = new_features.permute(0, 2, 1).unsqueeze(-1)  # (npoint, C', nsample, 1)
            new_features = self.mlp(new_features)  # (npoint, mlp[-1], nsample, 1)
            new_features = torch.max(new_features, 2)[0].squeeze(-1)  # (npoint, mlp[-1])

            new_xyz_list.append(new_xyz_b)
            new_features_list.append(new_features)

        new_xyz = torch.stack(new_xyz_list, dim=0)  # (B, npoint, 3)
        new_features = torch.stack(new_features_list, dim=0)  # (B, npoint, mlp[-1])
        return new_xyz, new_features


# ----------- 注意力机制聚合 -----------
class AttentionPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.query = nn.Linear(in_dim, hidden_dim)
        self.key = nn.Linear(in_dim, hidden_dim)
        self.value = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        # x: (B, N, C)
        Q, K, V = self.query(x), self.key(x), self.value(x)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5), dim=-1)  # (B, N, N)
        out = attn @ V  # (B, N, hidden_dim)
        return out.mean(1)  # (B, hidden_dim)


# 局部卷积 + 全局注意力的帧级点云特征提取器 3D
class FramePointCloudEncoder(nn.Module):
    def __init__(self, input_dim=3, emb_dim=256):
        super(FramePointCloudEncoder, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=32,
                                          in_channel=input_dim, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32,
                                          in_channel=128 + 3, mlp=[128, 128, 256])

        self.attn_pool = AttentionPooling(256, emb_dim)

    def forward(self, xyz, features=None):
        # xyz: (B, N, 3)
        # features: (B, C, N) or None
        xyz1, feat1 = self.sa1(xyz, features)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        feat2 = feat2.unsqueeze(0) if feat2.dim() == 2 else feat2  # 防止batch=1出错

        B = xyz.size(0)
        feat2 = feat2.view(B, -1, feat2.size(-1))  # (B, N', C)
        out = self.attn_pool(feat2)  # (B, emb_dim)
        return out
