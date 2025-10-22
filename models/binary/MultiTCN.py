import torch
import torch.nn as nn
import torch.nn.functional as F

# 我们仍然使用之前的 DilatedTCNBlock 作为构建模块
class DilatedTCNBlock(nn.Module):
    # ... (这个类的代码保持不变，和您之前的一样)
    def __init__(self, in_channels, out_channels, dilation, dropout_p=0.3):
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


# --- 【核心】新的多头模型 ---
class MultiHeadTCN(nn.Module):
    def __init__(self, shape_feat_dim, event_feat_dim, dropout_p=0.3):
        """
        Args:
            shape_feat_dim (int): “形态”特征的数量 (例如，原始3个 + 导数4个 = 7)
            event_feat_dim (int): “事件”特征的数量 (例如，mask+angle = 2)
        """
        super().__init__()
        self.shape_feat_dim = shape_feat_dim
        self.event_feat_dim = event_feat_dim

        # --- 分支一：“形态”特征编码器 (一个深度TCN) ---
        self.shape_encoder = nn.Sequential(
            DilatedTCNBlock(shape_feat_dim, 32, dilation=1, dropout_p=dropout_p),
            DilatedTCNBlock(32, 64, dilation=2, dropout_p=dropout_p),
            DilatedTCNBlock(64, 64, dilation=4, dropout_p=dropout_p)
        )
        
        # --- 分支二：“事件”特征编码器 (一个浅层TCN) ---
        self.event_encoder = nn.Sequential(
            DilatedTCNBlock(event_feat_dim, 16, dilation=1, dropout_p=dropout_p),
            DilatedTCNBlock(16, 32, dilation=2, dropout_p=dropout_p)
        )

        # --- 融合与分类头 (Fusion and Classifier Head) ---
        # 融合后的特征维度是 64 (来自shape) + 32 (来自event) = 96
        self.fusion_head = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=1), # 1x1 卷积用于融合和降维
            nn.ReLU(),
            nn.Dropout(0.5), # 在最终分类前使用更强的Dropout
            nn.Conv1d(64, 1, kernel_size=1)   # 输出最终的logit
        )

    def forward(self, x):
        """
        x: 完整的输入特征张量 [B, T, C_total]
        """
        # 0. 输入转置
        x = x.permute(0, 2, 1) # -> [B, C_total, T]

        # 1. 根据维度切分输入，送入不同分支
        shape_features = x[:, :self.shape_feat_dim, :]
        event_features = x[:, self.shape_feat_dim:, :]
        
        # 2. 通过各自的编码器
        shape_embedding = self.shape_encoder(shape_features) # -> [B, 64, T]
        event_embedding = self.event_encoder(event_features) # -> [B, 32, T]

        # 3. 融合 (Concatenate)
        fused_embedding = torch.cat([shape_embedding, event_embedding], dim=1) # -> [B, 96, T]
        
        # 4. 通过融合头得到最终输出
        logits = self.fusion_head(fused_embedding) # -> [B, 1, T]
        
        return logits.squeeze(1) # -> [B, T]