import torch
from torch.nn import Linear as Lin
from torch_geometric.nn import (
    MLP,
    PointTransformerConv,
    fps,
    global_mean_pool,
    knn,
    knn_graph,
)
from torch_geometric.utils import scatter



class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None,
                           plain_last=False)

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                        dim_size=id_clusters.size(0), reduce='max')

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class PointTransformer(torch.nn.Module):
    """
    帧级(Point Cloud Frame)分类版本
    支持输入:
      1) data: [B, T, N, C] -> 输出 [B, T, num_classes]
      2) data: [B, N, C]    -> 输出 [B, num_classes]
    说明:
      - 前3维视为坐标用于 KNN / FPS，其余视为附加特征
      - 每一帧内点云独立建图，不在时间维做交互
    """
    def __init__(self,
                 dim_model=(32, 64, 128, 256, 512),
                 k=16,
                 in_channels=9,
                 pos_dim=3,
                 num_classes=6):
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.pos_dim = pos_dim
        self.num_classes = num_classes

        # 输入特征映射到 dim_model[0]
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)
        self.transformer_input = TransformerBlock(in_channels=dim_model[0],
                                                  out_channels=dim_model[0])

        # 下采样 + Transformer 堆叠
        self.transition_down = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        for i in range(len(dim_model) - 1):
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1],
                               k=self.k)
            )
            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1],
                                 out_channels=dim_model[i + 1])
            )

        # 分类头
        self.mlp_output = MLP([dim_model[-1], 64, num_classes], norm=None)

    def _forward_single_batch(self, data_bt):
        """
        data_bt: [F, N, C]  (F = B 或 B*T)
        返回: [F, num_classes]
        """
        F, N, C = data_bt.shape
        assert C >= self.pos_dim, f"输入特征维 {C} 小于 pos_dim {self.pos_dim}"

        # 位置与特征
        pos = data_bt[..., :self.pos_dim].reshape(F * N, self.pos_dim)      # [F*N, pos_dim]
        feats = data_bt.reshape(F * N, C)                                   # [F*N, C]
        batch = torch.arange(F, device=data_bt.device).repeat_interleave(N) # [F*N]

        # 初始块
        x = self.mlp_input(feats)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # 下采样骨干
        for td, blk in zip(self.transition_down, self.transformers_down):
            x, pos, batch = td(x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = blk(x, pos, edge_index)

        # 全局聚合
        x = global_mean_pool(x, batch)  # [F, dim_last]
        out = self.mlp_output(x)        # [F, num_classes]
        return out

    def forward(self, data):
        """
        data 形状:
          - [B, N, C]
          - [B, T, N, C]
        """
        if data.dim() == 3:
            # [B,N,C]
            return self._forward_single_batch(data)
        elif data.dim() == 4:
            B, T, N, C = data.shape
            data_flat = data.view(B * T, N, C)
            logits = self._forward_single_batch(data_flat)  # [B*T, num_classes]
            return logits.view(B, T, self.num_classes)
        else:
            raise ValueError(f"不支持的输入形状: {data.shape}")

