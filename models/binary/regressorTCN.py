import torch
import torch.nn as nn
import os
from collections import OrderedDict

from models.pointnet import PointNetFeatureExtractor

class DilatedTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, dropout_p=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.downsample(x)

class KeyframeRegressor(nn.Module):
    def __init__(self, input_dim_1d, point_feature_dim, num_channels, dropout_p, num_keyframes=6):
        super().__init__()
       

        self.point_cloud_extractor = PointNetFeatureExtractor()
        model_weights_path = "./pointnet_adapted.pth"

        try:
            if os.path.exists(model_weights_path):
                print(f"加载 PointNet++ 预训练权重: {model_weights_path}")
                full_model_state_dict = torch.load(model_weights_path, map_location='cpu')
                extractor_weights = OrderedDict()
                for key, value in full_model_state_dict.items():
                    if key.startswith("point_cloud_extractor."):
                        new_key = key[len("point_cloud_extractor."):]
                        extractor_weights[new_key] = value
                if len(extractor_weights) == 0:
                    print("⚠️ 没找到提取器权重，尝试直接加载")
                    self.point_cloud_extractor.load_state_dict(full_model_state_dict, strict=False)
                else:
                    self.point_cloud_extractor.load_state_dict(extractor_weights, strict=True)
                    print("✅ 成功加载 PointNet++ 预训练特征提取器")
            else:
                print("⚠️ 未找到预训练权重，使用随机初始化")

        except Exception as e:
            print(f"加载 PointNet++ 过程中出错: {e}")

        print("🔒 冻结 PointNetFeatureExtractor 权重")
        for param in self.point_cloud_extractor.parameters():
            param.requires_grad = False

        self.point_cloud_extractor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # TCN encoder
        fused_input_dim = input_dim_1d + point_feature_dim
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = fused_input_dim if i == 0 else num_channels[i - 1]
            layers.append(DilatedTCNBlock(in_ch, out_ch, dilation, dropout_p=dropout_p))
        self.encoder = nn.Sequential(*layers)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.regression_head = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_keyframes)
        )

    def forward(self, x, point_cloud, src_key_padding_mask=None):
        # x: [B, T, D_1d]
        # point_cloud: [B, T, N, 3]  (每帧的点云)

        # 提取每一帧的点云特征（逐帧提取）
        B, T, N, _ = point_cloud.shape
        point_cloud = point_cloud.view(B * T, N, 3)  # [B*T, N, 3]
        point_features = self.point_cloud_extractor(point_cloud)  # [B*T, D_pc]
        point_features = point_features.view(B, T, -1)  # [B, T, D_pc]

        # 拼接 1D 特征与点云特征
        fused = torch.cat([x, point_features], dim=-1)  # [B, T, D_1d + D_pc]
        if src_key_padding_mask is not None:
            # mask是True的地方是padding，所以我们要把这些地方的特征置为0
            # unsqueeze(-1) 是为了让mask能和多维特征进行广播操作
            fused = fused.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        fused = fused.permute(0, 2, 1)  # [B, D, T] for Conv1d

        # 编码器 + pooling + regression
        embedding = self.encoder(fused)  # [B, D_out, T]
        global_feature = self.pooling(embedding).squeeze(-1)  # [B, D_out]
        predicted_coords_logits = self.regression_head(global_feature)  # [B, num_keyframes]
        predicted_coords_normalized = torch.sigmoid(predicted_coords_logits)  # [0,1] 范围

        return predicted_coords_normalized
