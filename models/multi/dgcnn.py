# from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dgcnn_segmentation.py
import torch
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
from torch import nn

class DGCNN(nn.Module):
    """
    逐帧点云分类:
      输入:
        - [B, T, N, C]  -> 输出 [B, T, num_classes]
        - [B, N, C]     -> 输出 [B, num_classes]
    每一帧视作一个独立图 (N 个点)。
    """
    def __init__(
        self,
        num_classes=6,
        input_channels=9,          # 点特征维(坐标+属性)
        k=20,
        aggr='max',
        conv_channels=(32, 32, 32),
        dense_channels=(1024, 1024, 256, 128),
        dropout=0.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.convs = nn.ModuleList()
        in_ch = input_channels
        # EdgeConv 序列
        for ch in conv_channels:
            self.convs.append(
                DynamicEdgeConv(
                    MLP([in_ch * 2, ch, ch, ch]),
                    k,
                    aggr
                )
            )
            in_ch = ch
        # 汇聚前线性
        self.lin1 = MLP([sum(conv_channels), dense_channels[0]])
        # 分类头
        self.output = MLP([*dense_channels, num_classes], dropout=dropout, norm=None)

    def _forward_frames(self, frames):
        """
        frames: [F, N, C]
        返回: [F, num_classes]
        """
        F, N, C = frames.shape
        x = frames.reshape(F * N, C)                       # [F*N, C]
        batch = torch.arange(F, device=frames.device).repeat_interleave(N)  # [F*N]
        feats_list = []
        for conv in self.convs:
            x = conv(x, batch)                            # [F*N, ch]
            feats_list.append(x)
        feat_cat = torch.cat(feats_list, dim=1)           # [F*N, sum(conv_channels)]
        feat_lin = self.lin1(feat_cat)                    # [F*N, dense0]
        global_feat = global_max_pool(feat_lin, batch)    # [F, dense0]
        logits = self.output(global_feat)                 # [F, num_classes]
        return logits

    def forward(self, data):
        """
        data: [B,T,N,C] 或 [B,N,C]
        """
        if data.dim() == 4:
            B, T, N, C = data.shape
            frames = data.view(B * T, N, C)
            logits = self._forward_frames(frames)         # [B*T, num_classes]
            return logits.view(B, T, self.num_classes)    # 逐帧
        elif data.dim() == 3:
            B, N, C = data.shape
            logits = self._forward_frames(data)           # [B, num_classes]
            return logits
        else:
            raise ValueError(f"不支持的输入形状: {data.shape}")