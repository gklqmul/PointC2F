import torch
import torch.nn as nn
import os
from collections import OrderedDict

from models.multi.bi_gru import FeatureEncoder


class MultiClassBiLSTMNEW(nn.Module):
    def __init__(self, num_classes, input_dim, point_cloud_extractor,
                 feat_emb_dim=256, hidden_dim=128, num_layers=2, dropout=0.1):
        """
        适用于多类别帧级别分类的双向LSTM模型。

        Args:
            num_classes (int): 输出类别的总数。
            input_dim_1d (int): 原始1D特征的维度 (例如，34)。
            point_feature_dim (int): PointNet++输出的点云特征维度 (例如，256)。
            hidden_dim (int): LSTM隐藏层的维度。
            num_layers (int): LSTM的层数。
            dropout (float): 在LSTM层之间应用的Dropout比例。
        """
        super(MultiClassBiLSTMNEW, self).__init__()
        
        
        # --- 1. 点云特征提取器 (与您的Transformer版本完全相同) ---
        self.point_cloud_extractor = point_cloud_extractor
        self.feature_encoder = FeatureEncoder(input_dim, feat_emb_dim)
        fused_input_dim = feat_emb_dim + 256

        self.fusion_proj = nn.Linear(fused_input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=fused_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,    
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True   
        )
        
        self.output_layer = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, src, point_cloud, src_key_padding_mask=None):
        """
        Args:
            src (torch.Tensor):          [B, T, C1]  原始1D特征序列
            point_cloud (torch.Tensor):  [B, T, N, 3]  点云序列
            src_key_padding_mask:        (未使用，为保持兼容性保留)
        """
        B, T, N, D = point_cloud.shape
        
        # 1. 提取每帧的点云特征 (与您的Transformer版本完全相同)
        point_cloud_flat = point_cloud.view(B * T, N, D)
        point_feats = self.point_cloud_extractor(point_cloud_flat)
        point_feats = point_feats.view(B, T, -1)  # 恢复形状为 [B, T, C2]
        # encode raw features
        feat_emb = self.feature_encoder(src)  # [B, T, feat_emb_dim]

        # fused features
        fused = torch.cat([feat_emb, point_feats], dim=-1)  # [B, T, feat_emb_dim+256]

        # 3. 将融合后的特征送入双向LSTM
        # LSTM返回 (output, (hidden_state, cell_state))
        # 我们只需要output
        lstm_output, _ = self.lstm(fused) # output形状: [B, T, hidden_dim * 2]

        # 4. 分类层
        output_logits = self.output_layer(lstm_output) # [B, T, num_classes]

        return output_logits
    



class ShuffledBiLSTM(nn.Module):
    def __init__(self, num_classes, input_dim, point_cloud_extractor, feat_emb_dim=256,
                 hidden_dim=128, num_layers=2, dropout=0.1, shuffle_time=False):
        """
        支持时序打乱的多类别帧级分类模型（BiLSTM版）。

        Args:
            num_classes (int): 输出类别数。
            input_dim (int): 原始1D特征维度（如 34）。
            point_cloud_extractor: PointNet++ 等特征提取器。
            hidden_dim (int): LSTM 隐层维度。
            num_layers (int): LSTM 层数。
            dropout (float): Dropout 概率。
            shuffle_time (bool): 是否在前向传播时打乱帧的顺序。
        """
        super().__init__()
        self.shuffle_time = shuffle_time

        self.point_cloud_extractor = point_cloud_extractor
        self.feature_encoder = FeatureEncoder(input_dim, feat_emb_dim)

        fused_input_dim = feat_emb_dim + 256

        self.lstm = nn.LSTM(
            input_size=fused_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.output_layer = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, src, point_cloud, src_key_padding_mask=None):
        """
        Args:
            src (torch.Tensor):         [B, T, C1] 原始特征
            point_cloud (torch.Tensor): [B, T, N, 3] 点云帧序列
        """
        B, T, N, D = point_cloud.shape

        # --- 随机打乱时序（仅在需要时执行） ---
        if self.training and self.shuffle_time:
            shuffled_indices = []
            for i in range(B):
                perm = torch.randperm(T, device=src.device)
                shuffled_indices.append(perm)
            # 将每个样本独立打乱
            shuffled_src = torch.stack([src[i, perm, :] for i, perm in enumerate(shuffled_indices)], dim=0)
            shuffled_pc = torch.stack([point_cloud[i, perm, :, :] for i, perm in enumerate(shuffled_indices)], dim=0)
        else:
            shuffled_src = src
            shuffled_pc = point_cloud

        # 1. 点云特征提取
        pc_flat = shuffled_pc.view(B * T, N, D)
        point_feats = self.point_cloud_extractor(pc_flat)
        point_feats = point_feats.view(B, T, -1)

        feat_emb = self.feature_encoder(src)  # [B, T, feat_emb_dim]

        # 2. 特征拼接
        fused = torch.cat([feat_emb, point_feats], dim=-1)  # [B, T, C1+C2]

        lstm_output, _ = self.lstm(fused)

        # 4. 分类
        output_logits = self.output_layer(lstm_output)  # [B, T, num_classes]

        return output_logits