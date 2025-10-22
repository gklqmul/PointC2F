import torch
import torch.nn as nn
import os


class MultiClassGRUOLD(nn.Module):
    def __init__(self, num_classes, input_dim, point_cloud_extractor, hidden_dim=128, num_layers=2, dropout=0.1):
        """
        适用于多类别帧级别分类的双向GRU模型。

        Args:
            num_classes (int): 输出类别的总数。
            input_dim_1d (int): 原始1D特征的维度。
            point_cloud_extractor (nn.Module): 外部传入的、已初始化的点云特征提取器。
            point_feature_dim (int): PointNet输出的点云特征维度。
            hidden_dim (int): GRU隐藏层的维度。
            num_layers (int): GRU的层数。
            dropout (float): 在GRU层之间应用的Dropout比例。
        """
        super(MultiClassGRUOLD, self).__init__()
   
        self.point_cloud_extractor = point_cloud_extractor
        
        fused_input_dim = input_dim
        
        self.gru = nn.GRU(
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
            src (torch.Tensor):          [B, T, C1]   原始1D特征序列
            point_cloud (torch.Tensor):  [B, T, N, 3] 点云序列
        """
        B, T, N, D = point_cloud.shape

        # a. 提取每帧的点云特征
        point_cloud_flat = point_cloud.view(B * T, N, D)
        point_feats = self.point_cloud_extractor(point_cloud_flat)
        point_feats = point_feats.view(B, T, -1)

        # b. 拼接原始特征与点云特征
        fused = torch.cat([src, point_feats], dim=-1)

        gru_output, _ = self.gru(fused) # output形状: [B, T, hidden_dim * 2]

        # d. 分类层
        output_logits = self.output_layer(gru_output) # [B, T, num_classes]

        return output_logits

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, emb_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.fc(x)
    
class MultiClassGRU(nn.Module):
    def __init__(self, num_classes, input_dim, point_cloud_extractor,
                 feat_emb_dim=256, hidden_dim=128, num_layers=2, dropout=0.1):
        super(MultiClassGRU, self).__init__()
   
        self.point_cloud_extractor = point_cloud_extractor
        self.feature_encoder = FeatureEncoder(input_dim, feat_emb_dim)

        # fused input dim = encoded feature dim + point cloud dim
        fused_input_dim = feat_emb_dim + 256  # 注意：这里假设点云编码器输出 256

        # projection + norm
        self.fusion_proj = nn.Linear(fused_input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, src, point_cloud, src_key_padding_mask=None):
        """
        src: [B, T, input_dim] (原始一维特征序列)
        point_cloud: [B, T, N, D]
        """
        B, T, N, D = point_cloud.shape

        # point cloud features
        point_cloud_flat = point_cloud.view(B * T, N, D)
        point_feats = self.point_cloud_extractor(point_cloud_flat)  # [B*T, 256]
        point_feats = point_feats.view(B, T, -1)  # [B, T, 256]

        # encode raw features
        feat_emb = self.feature_encoder(src)  # [B, T, feat_emb_dim]

        # fused features
        fused = torch.cat([feat_emb, point_feats], dim=-1)  # [B, T, feat_emb_dim+256]
        fused = self.dropout(self.norm(self.fusion_proj(fused)))  # [B, T, hidden_dim]

        # temporal modeling
        gru_output, _ = self.gru(fused)  # [B, T, hidden_dim*2]

        # classification
        output_logits = self.output_layer(gru_output)  # [B, T, num_classes]

        return output_logits