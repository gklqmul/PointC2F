import torch
import torch.nn as nn

from models.multi.bi_gru import FeatureEncoder

class MLPClassifier(nn.Module):
    def __init__(self, num_classes, input_dim, point_cloud_extractor, feat_emb_dim=256, hidden_dim=256, dropout=0.5):
        """
        非时序模型：逐帧 MLP 分类器（用于对比 BiLSTM、GRU 等时序模型）。

        Args:
            num_classes (int): 输出类别数。
            input_dim (int): 原始1D特征维度（如 34）。
            point_cloud_extractor: PointNet 或 PointNet++ 模块。
            hidden_dim (int): MLP隐藏层维度。
            dropout (float): Dropout比例。
        """
        super().__init__()
        self.point_cloud_extractor = point_cloud_extractor
        self.feature_encoder = FeatureEncoder(input_dim, feat_emb_dim)

        fused_input_dim = feat_emb_dim + 256

        self.mlp = nn.Sequential(
            nn.Linear(fused_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, src, point_cloud, src_key_padding_mask=None):
        """
        Args:
            src (torch.Tensor):         [B, T, C1] 原始1D特征序列
            point_cloud (torch.Tensor): [B, T, N, 3] 点云序列
        Returns:
            output_logits: [B, T, num_classes]
        """
        B, T, N, D = point_cloud.shape

        # 提取点云特征：[B*T, N, D] → [B*T, C2]
        point_cloud_flat = point_cloud.view(B * T, N, D)
        point_feats = self.point_cloud_extractor(point_cloud_flat)
        point_feats = point_feats.view(B, T, -1)  # [B, T, C2]

        feat_emb = self.feature_encoder(src)
        fused = torch.cat([feat_emb, point_feats], dim=-1)  # [B, T, C1 + C2]

        # 展平时间维做逐帧 MLP 分类
        fused_flat = fused.view(B * T, -1)              # [B*T, C]
        logits_flat = self.mlp(fused_flat)              # [B*T, num_classes]
        output_logits = logits_flat.view(B, T, -1)      # [B, T, num_classes]

        return output_logits
