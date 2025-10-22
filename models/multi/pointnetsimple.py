import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class PointNetPPFrameClassifier(nn.Module):
    """
    帧级 PointNet++ 分类
    输入:
      - [B, N, C] -> [B, num_classes]
      - [B, T, N, C] -> [B, T, num_classes]
    仅使用前三维为坐标；若存在附加属性 (C>3) 会拼接到初始点特征中（可选）。
    """
    def __init__(self,
                 num_classes: int = 6,
                 use_extra_features: bool = True,  # 是否利用除坐标外的特征
                 output_global_feat: bool = False, # 是否同时返回全局特征
                 feat_dim: int = 1024):
        super().__init__()
        self.num_classes = num_classes
        self.use_extra_features = use_extra_features
        self.output_global_feat = output_global_feat

        # 与原始结构一致
        # 首层 MLP 输入固定为 6(=3*2)（PointNetConv 聚合 pos 生成 6 输入），
        # 如果加入额外特征，会在进入 sa1 前拼接后再线性映射到 3 维再走与原逻辑一致的通路（保持稳定）
        self.extra_proj = None  # 延迟构建

        self.sa1_module = SAModule(0.5, 0.2, MLP([3 * 2, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, feat_dim]))

        self.classifier = MLP([feat_dim, 512, 256, num_classes],
                              dropout=0.5, norm=None)

    def _prepare_extra(self, C_extra: int, device):
        # 将 (xyz + extra) 通过线性 -> xyz_like(3维) 以复用原始结构
        if self.extra_proj is None and C_extra > 0:
            self.extra_proj = nn.Sequential(
                nn.Linear(3 + C_extra, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 3)
            ).to(device)

    def _forward_frames(self, data_fnc):
        """
        data_fnc: [F, N, C]
        返回: [F, num_classes] (+ 可选全局特征)
        """
        F, N, C = data_fnc.shape
        assert C >= 3, "输入最后一维至少包含3个坐标"
        pos = data_fnc[..., :3].reshape(F * N, 3)
        if C > 3 and self.use_extra_features:
            extra = data_fnc[..., 3:]
            self._prepare_extra(extra.shape[-1], data_fnc.device)
            # 融合额外特征 -> 3维伪坐标特征输入第一层
            fused0 = torch.cat([data_fnc[..., :3], extra], dim=-1)  # [F,N,3+E]
            fused0 = self.extra_proj(fused0)                        # [F,N,3]
            x0 = fused0.reshape(F * N, 3)
        else:
            x0 = pos  # 只用坐标

        batch = torch.arange(F, device=data_fnc.device).repeat_interleave(N)

        sa0_out = (x0, pos, batch)          # (features, pos, batch)
        x, pos, batch = self.sa1_module(*sa0_out)
        x, pos, batch = self.sa2_module(x, pos, batch)
        x, pos, batch = self.sa3_module(x, pos, batch)   # x: [F, feat_dim]

        logits = self.classifier(x)          # [F, num_classes]
        if self.output_global_feat:
            return logits, x
        return logits

    def forward(self, data):
        """
        data:
          [B,N,C] 或 [B,T,N,C]
        """
        if data.dim() == 3:
            out = self._forward_frames(data)
            return out
        elif data.dim() == 4:
            B, T, N, C = data.shape
            flat = data.view(B * T, N, C)
            out = self._forward_frames(flat)  # [B*T,num_classes] 或 (logits, feats)
            if self.output_global_feat:
                logits, feats = out
                return logits.view(B, T, -1), feats.view(B, T, -1)
            return out.view(B, T, -1)
        else:
            raise ValueError(f"不支持的输入形状: {data.shape}")

