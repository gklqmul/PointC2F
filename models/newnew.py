import torch
import torch.nn as nn
from torch_geometric.nn import fps, radius, global_max_pool, PointNetConv, MLP

# -------------------------
# SAModule: 局部特征聚合
# -------------------------
class SAModule(nn.Module):
    def __init__(self, ratio, r, mlp_fn, in_channels):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(mlp_fn(in_channels), add_self_loops=False)

    def forward(self, x, pos, batch):
        # fps 下采样
        idx = fps(pos, batch, ratio=self.ratio)
        # 半径搜索邻居
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.nn = mlp

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNetFeatureExtractorFlexible(nn.Module):

    def __init__(self, input_feat_dim, output_dim=256):
        super().__init__()
        self.input_feat_dim = input_feat_dim

        # 局部特征聚合模块
        self.sa1_module = SAModule(
            ratio=0.5,
            r=0.2,
            mlp_fn=lambda in_ch: MLP([in_ch, 64, 64, 128]),
            in_channels=input_feat_dim
        )
        self.sa2_module = SAModule(
            ratio=0.25,
            r=0.4,
            mlp_fn=lambda in_ch: MLP([in_ch + 3, 128, 128, 256]),  # 加上坐标
            in_channels=128
        )
        # 全局特征聚合
        self.sa3_module = GlobalSAModule(
            MLP([256 + 3, 256, 512, 1024])
        )
        self.fc = nn.Linear(1024, output_dim)

    def forward(self, data):
        """
        data: [B, N, C]  # C>=3
        """
        B, N, C = data.shape
        x = data.view(B*N, C)
        batch = torch.arange(B, device=data.device).repeat_interleave(N)
        pos = data[..., :3].reshape(B*N, 3)  # 前3维为坐标

        # SA模块
        x, pos, batch = self.sa1_module(x, pos, batch)
        x, pos, batch = self.sa2_module(x, pos, batch)
        x, pos, batch = self.sa3_module(x, pos, batch)

        x = self.fc(x)  # [B, output_dim]
        return x
