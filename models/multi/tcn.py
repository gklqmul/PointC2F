import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import os

from models.multi.bi_gru import FeatureEncoder

class Chomp1d(nn.Module):
    def __init__(self, padding):
        super(Chomp1d, self).__init__()
        self.padding = padding

    def forward(self, x):
        return x[:, :, :-self.padding]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)  # 使用自定义模块替代lambda函数
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 使用自定义模块替代lambda函数
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNClassifier(nn.Module):
    def __init__(self, num_classes, input_dim, point_cloud_extractor,
                 feat_emb_dim=256, hidden_dim=128, num_layers=2, 
                 dropout=0.1, kernel_size=3):
        """
        A Temporal Convolutional Network (TCN) for frame-level classification.

        Args:
            num_classes (int): number of output classes.
            input_dim (int): raw feature input dimension.
            point_cloud_extractor (nn.Module): feature extractor for point clouds.
            feat_emb_dim (int): dimension of encoded raw features.
        """
        super(TCNClassifier, self).__init__()

        self.point_cloud_extractor = point_cloud_extractor
        self.feature_encoder = FeatureEncoder(input_dim, feat_emb_dim)

        # 融合维度 = 编码后的特征维度 + 点云特征维度
        self.pointcloud_out_dim = getattr(point_cloud_extractor, "output_dim", 256)
        fused_input_dim = feat_emb_dim + self.pointcloud_out_dim

        # TCN 层
        layers = []
        num_channels = [hidden_dim] * num_layers
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = fused_input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout)]
        self.tcn = nn.Sequential(*layers)

        # 分类层
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, src, point_cloud, src_key_padding_mask=None):
        """
        src: [B, T, C1] 原始特征
        point_cloud: [B, T, N, D] 点云
        """
        B, T, N, D = point_cloud.shape

        # 1. 编码原始特征
        src_encoded = self.feature_encoder(src)   # [B, T, feat_emb_dim]

        # 2. 提取点云特征
        point_cloud_flat = point_cloud.view(B * T, N, D)
        point_feats = self.point_cloud_extractor(point_cloud_flat)  # [B*T, pc_dim]
        point_feats = point_feats.view(B, T, -1)                    # [B, T, pc_dim]

        # 3. 拼接
        fused = torch.cat([src_encoded, point_feats], dim=-1)       # [B, T, feat_emb+pc_dim]

        # 4. TCN 输入 [B, C, T]
        fused_permuted = fused.permute(0, 2, 1)
        tcn_output = self.tcn(fused_permuted)                       # [B, hidden_dim, T]
        tcn_output_permuted = tcn_output.permute(0, 2, 1)           # [B, T, hidden_dim]

        # 5. 分类
        output_logits = self.output_layer(tcn_output_permuted)      # [B, T, num_classes]
        return output_logits