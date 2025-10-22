# from https://github.com/ma-xu/pointMLP-pytorch/blob/main/classification_ModelNet40/models/pointmlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import fps, knn

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        sample_ratio = S/N
        xbatch = torch.arange(B).repeat_interleave(N).to(xyz.device)
        ybatch = torch.arange(B).repeat_interleave(S).to(xyz.device)
        fps_idx = fps(xyz.reshape((B*N, C)), batch=xbatch, ratio=sample_ratio)   # [B* npoint]
        fps_idx = fps_idx - ybatch*N  # [B* npoint]
        fps_idx = fps_idx.reshape((B, S))  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]
        idx = knn(xyz.reshape((B*N, C)), new_xyz.reshape((B*S, C)), self.kneighbors, xbatch, ybatch)[1]
        idx = idx - torch.arange(B).repeat_interleave((S*self.kneighbors)).to(xyz.device)*N
        idx = idx.reshape((B, S, self.kneighbors))
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class PointMLP(nn.Module):
    def __init__(self, embed_dim=64, groups=1, res_expansion=1,
                 activation="relu", bias=False, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[12, 12, 12, 12], reducers=[2, 2, 2, 2],
                 input_channels=9, num_classes=6, **kwargs):
        """
        支持帧级分类 (逐帧标签):
          输入:
            - [B, T, N, C] -> 输出 [B, T, num_classes]
            - [B, N, C]    -> 输出 [B, num_classes]
        C 包含 3D 坐标与附加属性 (前3维必须为坐标)
        """
        super().__init__()
        self.stages = len(pre_blocks)
        self.num_classes = num_classes
        self.points = 512            # 预期每帧点数 (数据集固定 N=512 时保持一致)
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.activation = activation

        self.embedding = ConvBNReLU1D(self.input_channels, embed_dim, bias=bias, activation=activation)

        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent."

        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()

        last_channel = embed_dim
        anchor_points = self.points
        for i in range(self.stages):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            self.local_grouper_list.append(
                LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)
            )
            self.pre_blocks_list.append(
                PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                              res_expansion=res_expansion, bias=bias,
                              activation=activation, use_xyz=use_xyz)
            )
            self.pos_blocks_list.append(
                PosExtraction(out_channel, pos_block_num, groups=groups,
                              res_expansion=res_expansion, bias=bias, activation=activation)
            )
            last_channel = out_channel

        self.last_channel = last_channel
        self.act = get_activation(activation)
        self.output = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

    # ...existing code (other helpers / build_classifier if any)...

    def _forward_batch(self, x):
        """
        x: [B, N, C]  ->  [B, num_classes]
        """
        B, N, C = x.shape
        if C != self.input_channels:  # 若通道变化（异常），动态重建首层
            self.input_channels = C
            self.embedding = ConvBNReLU1D(C, self.embed_dim, activation=self.activation).to(x.device)

        xyz = x[..., :3]                 # 坐标
        feat = x.permute(0, 2, 1)        # [B,C,N]
        feat = self.embedding(feat)      # [B,embed_dim,N]
        feat_t = feat.permute(0, 2, 1)   # [B,N,embed_dim]

        local_xyz = xyz
        for lg, pre_blk, pos_blk in zip(self.local_grouper_list,
                                        self.pre_blocks_list,
                                        self.pos_blocks_list):
            local_xyz, grouped = lg(local_xyz, feat_t)    # grouped: [B,g,k,d]
            grouped = pre_blk(grouped)                   # [B,d,g]
            grouped = pos_blk(grouped)                   # [B,d,g]
            feat_t = grouped.permute(0, 2, 1)            # -> [B,g,d] (为下一层当作 N,d)

        feat_last = grouped                               # [B,d,g]
        feat_last = F.adaptive_max_pool1d(feat_last, 1).squeeze(-1)  # [B,d]
        logits = self.output(feat_last)                   # [B,num_classes]
        return logits

    def forward(self, x):
        """
        支持:
          x: [B, N, C]    -> [B, num_classes]
          x: [B, T, N, C] -> [B, T, num_classes]
        """
        if x.dim() == 3:
            return self._forward_batch(x)
        elif x.dim() == 4:
            B, T, N, C = x.shape
            x_flat = x.view(B * T, N, C)
            logits = self._forward_batch(x_flat)          # [B*T,num_classes]
            return logits.view(B, T, self.num_classes)
        else:
            raise ValueError(f"不支持的输入形状: {x.shape}")
        
    def build_classifier(self, k):
        return nn.Sequential(
            nn.Linear(self.last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, k)
        )
