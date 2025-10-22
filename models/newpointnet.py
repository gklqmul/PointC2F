import torch
import torch.nn as nn
import torch.nn.functional as F

class NeighborBuilder:
    def __init__(self, k=20, k0=8, radius_scale=1.5, mode='adaptive_radius', mutual=False):
        """
        k: base kNN, used for distance statistics or fallback
        k0: k0-th neighbor used to estimate local density
        radius_scale: radius = radius_scale * dist_to_k0
        mode: 'adaptive_radius' or 'knn'
        mutual: True for mutual nearest neighbor filtering
        """
        self.k = k
        self.k0 = k0
        self.radius_scale = radius_scale
        self.mode = mode
        self.mutual = mutual

    @torch.no_grad()
    def build(self, coords):
        """
        coords: [B, N, 3]
        return: idx [B, N, k], mask [B, N, k] (True=valid)
        """
        B, N, _ = coords.shape
        dist = torch.cdist(coords, coords)  # [B, N, N]
        topk = self.k if self.mode == 'knn' else max(self.k, self.k0)
        dvals, nbrs = dist.topk(k=topk + 1, largest=False)  # Contains itself
        dvals, nbrs = dvals[:, :, 1:], nbrs[:, :, 1:]  # Remove self => [B, N, topk]

        if self.mode == 'adaptive_radius':
            k0 = min(self.k0, topk)
            r = self.radius_scale * dvals[:, :, k0 - 1:k0]  # [B, N, 1]
            mask = dvals <= r  # [B, N, topk]
        else:
            mask = torch.ones_like(dvals, dtype=torch.bool)

        if self.mutual:
            mutual_mask = torch.zeros_like(mask)
            for b in range(B):
                nb = nbrs[b]  # [N, topk]
                for i in range(N):
                    js = nb[i]
                    js_nbrs = nb[js]
                    cond = (js_nbrs == i).any(dim=1)
                    mutual_mask[b, i] = cond
            mask = mask & mutual_mask

        # Sort based on mask (True first), then by distance
        # 统一裁成 k 个邻居：优先保留 mask=True 的
        order = (~mask).float() + (dvals / (dvals.max() + 1e-6)) * 1e-3
        _, sel = order.sort(dim=-1)
        sel = sel[:, :, :self.k]
        gather_idx = nbrs.gather(dim=-1, index=sel)
        gather_mask = mask.gather(dim=-1, index=sel)

        return gather_idx.contiguous(), gather_mask.contiguous()


def get_edge_features(x, k=15):
    # x: [B, C, N] -> edge: [B, 2C, N, k]
    B, C, N = x.size()
    with torch.no_grad():
        dist = torch.cdist(x.transpose(2, 1), x.transpose(2, 1))  # [B,N,N]
        idx = dist.topk(k=k, largest=False)[1]                    # [B,N,k]
        idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
        idx = (idx + idx_base).view(-1)
    x_t = x.transpose(2, 1).contiguous()                          # [B,N,C]
    neigh = x_t.view(B * N, C)[idx, :].view(B, N, k, C)           # [B,N,k,C]
    central = x_t.view(B, N, 1, C).expand(-1, -1, k, -1)          # [B,N,k,C]
    edge = torch.cat([neigh - central, central], dim=3)           # [B,N,k,2C]
    edge = edge.permute(0, 3, 1, 2).contiguous()                  # [B,2C,N,k]
    return edge

def build_edge_features_from_idx(x, idx):
    # x: [B,C,N], idx: [B,N,k]
    B, C, N = x.shape
    k = idx.shape[-1]
    x_t = x.transpose(2,1).contiguous()           # [B,N,C]
    base = torch.arange(B, device=x.device).view(-1,1,1)*N
    gather = x_t.view(B*N, C)[(idx+base).view(-1)].view(B, N, k, C)  # [B,N,k,C]
    center = x_t.view(B, N, 1, C).expand(-1, -1, k, -1)              # [B,N,k,C]
    edge = torch.cat([gather - center, center], dim=-1)              # [B,N,k,2C]
    edge = edge.permute(0,3,1,2).contiguous()                        # [B,2C,N,k]
    return edge


# class FirstEdgeConvBlock(nn.Module):
#     def __init__(self, k=20, coord_dim=3, attr_dim=6, out_channels=64):
#         super().__init__()
#         self.k = k
#         self.coord_dim = coord_dim
#         self.attr_dim = attr_dim
#         in_channels = 2*coord_dim + 2*attr_dim 
#         self.mlp = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, pts):
#         coords = pts[..., :self.coord_dim]   # [B,N,3]
#         attrs  = pts[..., self.coord_dim:]   # [B,N,6]

#         B, N, _ = coords.shape
#         with torch.no_grad():
#             dist = torch.cdist(coords, coords)           # [B,N,N]
#             idx = dist.topk(k=self.k, largest=False)[1]  # [B,N,k]
#             idx_base = torch.arange(0, B, device=pts.device).view(-1,1,1)*N
#             idx_flat = (idx + idx_base).view(-1)

#         # 展开 gather
#         coords_c = coords.view(B*N, -1)[idx_flat, :].view(B, N, self.k, -1)  # neigh coords
#         attrs_c  = attrs.view(B*N, -1)[idx_flat, :].view(B, N, self.k, -1)   # neigh attrs

#         coords_i = coords.view(B, N, 1, -1).expand(-1, -1, self.k, -1)
#         attrs_i  = attrs.view(B, N, 1, -1).expand(-1, -1, self.k, -1)

#         # 构造边特征: (coord_j - coord_i, coord_i, attr_j - attr_i, attr_i)
#         edge_feat = torch.cat([coords_c - coords_i,
#                                coords_i,
#                                attrs_c - attrs_i,
#                                attrs_i], dim=3)  # [B,N,k,18]
#         edge_feat = edge_feat.permute(0, 3, 1, 2).contiguous()     # [B,18,N,k]

#         out = self.mlp(edge_feat)          # [B,64,N,k]
#         out = out.max(dim=-1)[0]           # [B,64,N]
#         return out


# ---------- 后续 EdgeConv 块（输入 C，内部自动构造 2C） ----------
# class EdgeConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, k=20):
#         super().__init__()
#         self.k = k
#         self.mlp = nn.Sequential(
#             nn.Conv2d(2*in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         # x: [B,C,N]
#         edge = get_edge_features(x, k=self.k)  # [B,2C,N,k]
#         out = self.mlp(edge)                  # [B,out,N,k]
#         out = out.max(dim=-1)[0]              # [B,out,N]
#         return out

class SoftEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_attr=True, tau=0.2):
        super().__init__()
        self.use_attr = use_attr
        self.tau = tau
        self.mlp = nn.Sequential(
            nn.Conv2d(2*in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # 边打分器（基于局部特征差异）
        self.edge_scorer = nn.Sequential(
            nn.Conv2d(2*in_channels, out_channels//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, 1, 1)
        )

    def forward(self, x, idx, mask, edge_feat_builder):
        """
        x: [B,C,N]  (这里的C可以是上一层特征维度)
        idx: [B,N,k]
        mask: [B,N,k] True=valid
        edge_feat_builder: 函数，输入(x, idx)返回 [B, 2C, N, k] 边特征
        """
        B, C, N = x.shape
        edge = edge_feat_builder(x, idx)         # [B,2C,N,k]
        logits = self.edge_scorer(edge) / self.tau   # [B,1,N,k]
        # 对无效邻居置 -inf，避免参与softmax
        neg_inf = torch.finfo(x.dtype).min
        logits = logits.masked_fill(~mask[:, None, :, :], neg_inf)
        alpha = torch.softmax(logits, dim=-1)         # [B,1,N,k]

        h = self.mlp(edge)                            # [B,out,N,k]
        out = (h * alpha).sum(dim=-1)                 # [B,out,N]
        return out
    
# ---------- 注意力汇聚 ----------
class AttentionPooling(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, 1)
        )
    def forward(self, x):
        # x: [B,C,N]
        w = torch.softmax(self.att(x), dim=-1)   # [B,1,N]
        return torch.sum(x * w, dim=-1)          # [B,C]

# choose fixed 10 neighbors
# class ImprovedPointNetExtractorNEW(nn.Module):
#     def __init__(self, output_dim=256, k=10):
#         super().__init__()
#         self.output_dim = output_dim
#         self.k = k
#         self.ec1 = FirstEdgeConvBlock(k=k, coord_dim=3, attr_dim=2, out_channels=64)
#         self.ec2 = SoftEdgeConv(64, 128)
#         self.ec3 = SoftEdgeConv(128, 256)
#         self.pool = AttentionPooling(256)
#         self.fc = nn.Sequential(
#             nn.Linear(256, output_dim),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         # x: [B,N,9]
#         f1 = self.ec1(x)                 # [B,64,N]
#         f2 = self.ec2(f1)                # [B,128,N]
#         f3 = self.ec3(f2)                # [B,256,N]
#         pooled = self.pool(f3)           # [B,256]
#         out = self.fc(pooled)            # [B,output_dim]
#         return out

class FirstEdgeConvBlock(nn.Module):
    def __init__(self, k=20, coord_dim=3, attr_dim=2, out_channels=64):
        super().__init__()
        self.k = k
        self.coord_dim = coord_dim
        self.attr_dim = attr_dim
        in_channels = 2*coord_dim + 2*attr_dim 
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, pts):
        coords = pts[..., :self.coord_dim]   # [B,N,3]
        attrs  = pts[..., self.coord_dim:]   # [B,N,2]

        B, N, _ = coords.shape
        with torch.no_grad():
            dist = torch.cdist(coords, coords)           # [B,N,N]
            idx = dist.topk(k=self.k, largest=False)[1]  # [B,N,k]
            idx_base = torch.arange(0, B, device=pts.device).view(-1,1,1)*N
            idx_flat = (idx + idx_base).view(-1)

        # 展开 gather
        coords_c = coords.view(B*N, -1)[idx_flat, :].view(B, N, self.k, -1)   # neigh coords
        attrs_c  = attrs.view(B*N, -1)[idx_flat, :].view(B, N, self.k, -1)    # neigh attrs

        coords_i = coords.view(B, N, 1, -1).expand(-1, -1, self.k, -1)
        attrs_i  = attrs.view(B, N, 1, -1).expand(-1, -1, self.k, -1)

        # 构造边特征: (coord_j - coord_i, coord_i, attr_j - attr_i, attr_i)
        edge_feat = torch.cat([coords_c - coords_i,
                               coords_i,
                               attrs_c - attrs_i,
                               attrs_i], dim=3)  # [B,N,k,18]
        edge_feat = edge_feat.permute(0, 3, 1, 2).contiguous()      # [B,18,N,k]

        out = self.mlp(edge_feat)             # [B,64,N,k]
        out = out.max(dim=-1)[0]              # [B,64,N]
        return out


class ImprovedPointNetExtractorNEW(nn.Module):
    def __init__(self, output_dim=256, k=10):
        super().__init__()
        self.output_dim = output_dim
        self.k = k
        self.neighbor_builder = NeighborBuilder(k=k, mode='knn')
        self.ec1 = FirstEdgeConvBlock(k=k, coord_dim=3, attr_dim=2, out_channels=64)
        self.ec2 = SoftEdgeConv(64, 128)
        self.ec3 = SoftEdgeConv(128, 256)
        self.pool = AttentionPooling(256)
        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, N, 9] (coords + attributes)
        B, N, C = x.shape
        coords = x[:, :, :3].contiguous()
        
        # Find neighbors for the SoftEdgeConv layers
        # 为 SoftEdgeConv 层寻找邻居
        idx, mask = self.neighbor_builder.build(coords)
        
        # Define the edge feature builder function to be passed to SoftEdgeConv
        # 定义要传递给 SoftEdgeConv 的边特征构建函数
        edge_feat_builder_func = lambda features, neighbor_idx: build_edge_features_from_idx(features, neighbor_idx)

        # First EdgeConv
        # 第一层 EdgeConv
        f1 = self.ec1(x)  # [B, 64, N]
        
        # SoftEdgeConv layers
        # SoftEdgeConv 层
        f2 = self.ec2(f1, idx, mask, edge_feat_builder_func) # [B, 128, N]
        f3 = self.ec3(f2, idx, mask, edge_feat_builder_func) # [B, 256, N]
        
        pooled = self.pool(f3)  # [B, 256]
        out = self.fc(pooled)   # [B, output_dim]
        return out
    
# automatic neighbor construction
class AutomaticNeighborExtractor(nn.Module):
    def __init__(self, output_dim=256, k=8, tau=0.2):
        super().__init__()
        self.k = k
        self.tau = tau

        # ---------- 邻居构造器（可以提前复用） ----------
        self.nb = NeighborBuilder(k=k, k0=4, radius_scale=1.8, mode='adaptive_radius', mutual=True)

        # ---------- 第一层 EdgeConv 保留 max pooling ----------
        self.ec1 = FirstEdgeConvBlock(k=k, coord_dim=3, attr_dim=2, out_channels=64)

        # ---------- 后续层使用 SoftEdgeConv ----------
        self.ec2 = SoftEdgeConv(in_channels=64, out_channels=128, tau=tau)
        self.ec3 = SoftEdgeConv(in_channels=128, out_channels=256, tau=tau)

        # ---------- 注意力汇聚 + 全连接 ----------
        self.pool = AttentionPooling(256)
        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, pts):
        """
        pts: [B, N, 9]  (coords + attributes)
        """
        B, N, _ = pts.shape
        coords = pts[..., :3]

        idx, mask, _, _ = self.nb.build(coords)  # [B,N,k], [B,N,k]

        f1 = self.ec1(pts)  # [B,64,N]

        def edge_builder(x, idx):
            return build_edge_features_from_idx(x, idx)  # [B,2C,N,k]

        # ---------- 第二层 SoftEdgeConv ----------
        f2 = self.ec2(f1, idx, mask, edge_builder)  # [B,128,N]

        # ---------- 第三层 SoftEdgeConv ----------
        f3 = self.ec3(f2, idx, mask, edge_builder)  # [B,256,N]

        # ---------- 注意力汇聚 + 输出 ----------
        pooled = self.pool(f3)  # [B,256]
        out = self.fc(pooled)   # [B, output_dim]
        return out


def get_local_diffs(x, k=8):
    """
    x: [B, N, C]
    return: [B, C*2, N] 局部差分均值特征
    """
    B, N, C = x.shape
   
    dists = torch.cdist(x[:, :, :3], x[:, :, :3])  # 只用坐标计算距离
    idx = dists.topk(k=k+1, largest=False)[1][:, :, 1:]  # 去掉自身
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx_flat = (idx + idx_base).view(-1)

    x_flat = x.view(B*N, C)
    neigh = x_flat[idx_flat, :].view(B, N, k, C)
    central = x.unsqueeze(2).expand(-1, -1, k, -1)
    diff = neigh - central  # [B, N, k, C]
    diff_mean = diff.mean(dim=2)  # [B, N, C]
    out = torch.cat([diff_mean, x], dim=2)  # [B, N, 2C]
    return out.transpose(1, 2).contiguous()  # [B, 2C, N]


class GlobalAttention(nn.Module):
    def __init__(self, in_channels, hidden=128):
        super().__init__()
        self.q = nn.Conv1d(in_channels, hidden, 1)
        self.k = nn.Conv1d(in_channels, hidden, 1)
        self.v = nn.Conv1d(in_channels, hidden, 1)
        self.fc = nn.Conv1d(hidden, in_channels, 1)

    def forward(self, x):
        """
        x: [B, C, N]
        """
        Q = self.q(x)       # [B,H,N]
        K = self.k(x)       # [B,H,N]
        V = self.v(x)       # [B,H,N]

        attn = torch.softmax(torch.bmm(Q.transpose(1,2), K), dim=-1)  # [B,N,N]
        out = torch.bmm(V, attn.transpose(1,2))                        # [B,H,N]
        out = self.fc(out)                                              # [B,C,N]
        return out + x  # 残差连接


# 轻量级点云编码器，带全局注意力 + 局部卷积
class LightweightPointCloudEncoderWithGlobalAttention(nn.Module):
    def __init__(self, input_dim=9, hidden_dims=[64,128,256], output_dim=256, k=8):
        super().__init__()
        self.k = k
        self.convs = nn.ModuleList()
        in_channels = input_dim * 2
        for h in hidden_dims:
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels, h, 1),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True)
            ))
            in_channels = h

        self.global_attn = GlobalAttention(in_channels)
        self.pool = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.ReLU(inplace=True)
        )
        self.att = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        x: [B, N, 9]
        """
        # 局部差分
        x = get_local_diffs(x, k=self.k)  # [B, 2*9, N]

        # 局部卷积
        for conv in self.convs:
            x = conv(x)  # [B, C, N]

        # 全局注意力
        x = self.global_attn(x)  # [B, C, N]

        # Attention Pooling
        w = torch.softmax(self.att(x), dim=-1)  # [B,1,N]
        pooled = torch.sum(x * w, dim=-1)       # [B,C]

        # 输出 embedding
        out = self.fc(pooled)                    # [B, output_dim]
        return out
    
# ---------- 分类器 ----------
class NewPointNetClassifier(nn.Module):
    def __init__(self, num_classes, input_dim, point_cloud_extractor):
        super().__init__()
        self.point_cloud_extractor = point_cloud_extractor
        self.classifier_head = nn.Linear(input_dim, num_classes)

    def forward(self, src, point_cloud, src_key_padding_mask=None):
        # point_cloud: [B,T,N,9]
        B, T, N, D = point_cloud.shape
        pc_flat = point_cloud.view(B*T, N, D)
        pc_feat = self.point_cloud_extractor(pc_flat).view(B, T, -1)  # [B,T,F]
        fused = torch.cat([src, pc_feat], dim=-1)                     # [B,T,srcF+F]
        return self.classifier_head(fused)
    

def get_local_diffs_coords(coords, k=8):
    B, N, C = coords.shape
    dists = torch.cdist(coords, coords)  # [B,N,N]
    idx = dists.topk(k=k+1, largest=False)[1][:,:,1:]
    idx_base = torch.arange(0,B,device=coords.device).view(-1,1,1)*N
    idx_flat = (idx + idx_base).view(-1)
    coords_flat = coords.view(B*N,C)
    neigh = coords_flat[idx_flat,:].view(B,N,k,C)
    central = coords.unsqueeze(2).expand(-1,-1,k,-1)
    diff = neigh - central  # [B,N,k,C]
    diff_mean = diff.mean(dim=2)  # [B,N,C]
    return diff_mean.transpose(1,2).contiguous()  # [B,C,N]


# Separate class for PointCloudEncoder with Geometric and Radar features, two branches
class PointCloudEncoderGeomRadar(nn.Module):
    def __init__(self, input_coord=3, input_attr=2, hidden_dims=[64,128,256], output_dim=256, k=8):
        super().__init__()
        self.k = k
        # 坐标分支
        self.conv_coord = nn.ModuleList()
        in_c = input_coord
        for h in hidden_dims:
            self.conv_coord.append(nn.Sequential(
                nn.Conv1d(in_c, h, 1),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True)
            ))
            in_c = h

        # 雷达分支
        self.conv_attr = nn.ModuleList()
        in_a = input_attr
        for h in hidden_dims:
            self.conv_attr.append(nn.Sequential(
                nn.Conv1d(in_a, h, 1),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True)
            ))
            in_a = h

        self.global_attn = GlobalAttention(in_c + in_a)
        self.att_pool = nn.Sequential(
            nn.Conv1d(in_c + in_a, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_c + in_a, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, N, _ = x.shape
        coords = x[...,:3]
        attrs  = x[...,3:]

        # 坐标差分
        x_c = get_local_diffs_coords(coords, k=self.k)
        for conv in self.conv_coord:
            x_c = conv(x_c)

        # 雷达特征分支
        x_a = attrs.transpose(1,2).contiguous()  # [B,6,N]
        for conv in self.conv_attr:
            x_a = conv(x_a)

        # 拼接
        x_all = torch.cat([x_c, x_a], dim=1)  # [B, C_total, N]

        # 全局注意力
        x_all = self.global_attn(x_all)

        # Attention pooling
        w = torch.softmax(self.att_pool(x_all), dim=-1)
        pooled = torch.sum(x_all * w, dim=-1)

        out = self.fc(pooled)
        return out