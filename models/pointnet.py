import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius

from models.multi.bi_gru import FeatureEncoder

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

class PointNetFeatureExtractor(torch.nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        # 只用 pos，不需要额外 node feature
        self.sa1_module = SAModule(0.5, 0.2, MLP([3 * 2, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
        self.fc = nn.Linear(1024, output_dim)  # 降维到 256

    def forward(self, data):
        # 输入 shape: [B, N, 3]
        batchsize, npoints, _ = data.shape
        x = data.reshape(batchsize * npoints, 3)
        batch = torch.arange(batchsize, device=data.device).repeat_interleave(npoints)

        # PointNet++ 三层提取
        x, pos, batch = self.sa1_module(x, x, batch)
        x, pos, batch = self.sa2_module(x, pos, batch)
        x, pos, batch = self.sa3_module(x, pos, batch)  # x shape: [B, 1024]
        x = self.fc(x)  # 降维到 [B, 256]

        return x  # 返回全局特征

class SimplePointNetFeatureExtractor(nn.Module):
    def __init__(self, input_dim=5, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        
        # 点特征提取器
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, self.output_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.output_dim)
        
        # 全局特征提取器
        self.fc1 = nn.Linear(self.output_dim, self.output_dim)
        self.bn4 = nn.BatchNorm1d(self.output_dim)
        
    def forward(self, x):
        # 输入形状: [B, N, 3]
        B, N, _ = x.shape
        
        # 确保输入没有NaN或无穷值
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 转置为点云格式 [B, 3, N]
        x = x.transpose(1, 2)
        
        # 点特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 最大池化获取全局特征
        x = torch.max(x, dim=2, keepdim=False)[0]  # [B, output_dim]
        
        # 全局特征提取
        x = F.relu(self.bn4(self.fc1(x)))
        
        return x

class PointNetClassifier(nn.Module):
    def __init__(self, num_classes, input_dim, point_cloud_extractor, feat_emb_dim=256 ):
        """
        用于帧级别分类的最终模型。

        Args:
            num_classes (int): 最终要分类的类别数。
            point_feature_dim (int): 特征提取器输出的维度。
        """
        super().__init__()
        
        # 1. 在这里直接实例化您写好的特征提取器
        self.point_cloud_extractor = point_cloud_extractor
        self.feature_encoder = FeatureEncoder(input_dim, feat_emb_dim)
        #    它的输入维度必须与特征提取器的输出维度完全匹配
       
        fused_input_dim = feat_emb_dim+ 256
        self.classifier_head = nn.Linear(fused_input_dim, num_classes)


    def forward(self, src, point_cloud, src_key_padding_mask=None):
        """
        前向传播非常简单：先提取特征，再分类。

        Args:
            x (torch.Tensor): 输入的一批点云，形状为 [B, N, 3]
        """

        B, T, N, D = point_cloud.shape
        src_encoded = self.feature_encoder(src)
        point_cloud_flat = point_cloud.view(B * T, N, D)
        point_feats = self.point_cloud_extractor(point_cloud_flat)
        point_feats = point_feats.view(B, T, -1)  

        fused = torch.cat([src_encoded, point_feats], dim=-1)  # [B, T, C1+C2]
        
        # 2. 使用分类头进行最终的分类
        # 输入: [B, 256] -> 输出: [B, num_classes]
        logits = self.classifier_head(fused)
        
        return logits

    def get_extractor_state_dict(self):
        """单独获取特征提取器的状态字典"""
        return self.point_cloud_extractor.state_dict()
    
    def load_extractor_state_dict(self, state_dict):
        """单独加载特征提取器的状态字典"""
        self.point_cloud_extractor.load_state_dict(state_dict)