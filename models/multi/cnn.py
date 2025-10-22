import torch
import torch.nn as nn

from models.multi.bi_gru import FeatureEncoder

class CNN1DClassifier(nn.Module):
    def __init__(self, num_classes, input_dim, point_cloud_extractor, feat_emb_dim=256,dropout=0.5):
        super().__init__()
        self.point_cloud_extractor = point_cloud_extractor
        self.dropout = nn.Dropout(dropout)
        self.feature_encoder = FeatureEncoder(input_dim, feat_emb_dim)
        self.fused_dim = feat_emb_dim + 256  # 假设PointNet++输出256维特征

        self.cnn = nn.Sequential(
            nn.Conv1d(self.fused_dim, 256, kernel_size=3, padding=1),  # 保持时间长度
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(256, 128, kernel_size=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, src, point_cloud, src_key_padding_mask=None):
        B, T, N, D = point_cloud.shape

        # 点云特征提取
        point_cloud_flat = point_cloud.view(B * T, N, D)
        point_feats = self.point_cloud_extractor(point_cloud_flat)  # [B*T, pc_feat_dim]
        point_feats = point_feats.view(B, T, -1)                    # [B, T, pc_feat_dim]

        # 融合特征
        feat_emb = self.feature_encoder(src)  # [B, T, feat_emb_dim]
        fused = torch.cat([feat_emb, point_feats], dim=-1)  # [B, T, fused_dim]

        x = fused.permute(0, 2, 1)

        x = self.cnn(x)

        x = x.permute(0, 2, 1)

        x = self.dropout(x)

        logits = self.classifier(x)  # [B, T, num_classes]

        return logits
