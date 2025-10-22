
import torch.nn as nn
import torch
import torch.nn.functional as F


class RadarKeyframeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 统一特征提取（处理全部8维特征）
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=5, padding=2),  # 输入通道改为8
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, dilation=2, padding=2),  # 扩大感受野
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 时序上下文建模
        self.temporal_attn = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        
        # 帧级预测头
        self.head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1)  # 输出每帧的预测概率
        )

    def forward(self, x):
        # x: [B, T, 8] -> 转换为通道优先 [B, 8, T]
        x = x.permute(0, 2, 1)
        
        # 特征提取
        feat = self.feature_extractor(x)  # [B, 64, T]
        
        # 时序注意力
        feat = feat.permute(0, 2, 1)  # [B, T, 64]
        attn_out, _ = self.temporal_attn(feat, feat, feat)  # [B, T, 64]
        
        # 帧级预测
        logits = self.head(attn_out.permute(0, 2, 1))  # [B, 1, T]
        return torch.sigmoid(logits.squeeze(1))  # [B, T]
    

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha  # 正样本权重
        self.gamma = gamma  # 难样本聚焦参数

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        BCE_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 防止数值溢出
        loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return loss.mean()