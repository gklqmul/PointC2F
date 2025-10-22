import torch
from torch.nn import Linear as Lin
from torch_geometric.nn import (
    MLP,
    PointTransformerConv,
    fps,
    global_mean_pool,
    knn_graph,
)
from torch_geometric.utils import scatter


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)
        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)
        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None, plain_last=False)
        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, batch):
        x = self.lin_in(x).relu()
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)
        sub_batch = batch[id_clusters] if batch is not None else None
        id_k_neighbor = knn_graph(pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch)
        x = self.mlp(x)
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                        dim_size=id_clusters.size(0), reduce='max')
        sub_pos = pos[id_clusters]
        return x_out, sub_pos, sub_batch


class PointTransformerExtractor(torch.nn.Module):
    def __init__(self, dim_model=[32, 64, 128, 256], k=16):
        super().__init__()
        self.k = k
        in_channels = 3  # 只用点坐标作为初始特征
        self.dim_model = dim_model

        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)
        self.transformer_input = TransformerBlock(dim_model[0], dim_model[0], k=k)

        self.transition_downs = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        for i in range(len(dim_model) - 1):
            self.transition_downs.append(
                TransitionDown(dim_model[i], dim_model[i + 1], k=k)
            )
            self.transformers_down.append(
                TransformerBlock(dim_model[i + 1], dim_model[i + 1], k=k)
            )
        self.output_dim = dim_model[-1]

    def forward(self, data):
        # data shape: [B*T, N, 3]
        batchsize = data.shape[0]
        npoints = data.shape[1]

        x = data.reshape((batchsize * npoints, 3))
        batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)
        pos = x

        x = self.mlp_input(x)
        x = self.transformer_input(x, pos, batch)

        for td, trans in zip(self.transition_downs, self.transformers_down):
            x, pos, batch = td(x, pos, batch)
            x = trans(x, pos, batch)

        x = global_mean_pool(x, batch)  # [B*T, C]
        return x


class PointTransformerClassifier(torch.nn.Module):
    """
    逐帧分类模型，不考虑时序，输入：
      point_cloud: [B, T, N, 3]
      extra_feat: [B, T, D1]  # 额外的1D特征
    输出：
      logits: [B, T, num_classes]
    """
    def __init__(self, num_classes, input_dim, dim_model=[32, 64, 128, 256, 512],
                 k=16, dropout=0.5, hidden_dim=256):
        super().__init__()
        self.point_cloud_extractor = PointTransformerExtractor(dim_model=dim_model, k=k)

        input_dim = input_dim

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, src, point_cloud, src_key_padding_mask=None):
        """
        Args:
            point_cloud: [B, T, N, 3]
            extra_feat: [B, T, D1] or None
        Returns:
            logits: [B, T, num_classes]
        """
        B, T, N, _ = point_cloud.shape
        point_cloud_flat = point_cloud.view(B * T, N, 3)
        point_features = self.point_cloud_extractor(point_cloud_flat)  # [B*T, C]

        point_features = point_features.view(B, T, -1)  # [B, T, C]

        if src is not None:
            # concat点云特征和1D特征
            x = torch.cat([point_features, src], dim=-1)  # [B, T, C + D1]
        else:
            x = point_features

        logits = self.mlp(x)  # [B, T, num_classes]
        return logits