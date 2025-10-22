
import torch
import torch.nn as nn
import math

from models.multi.bi_gru import FeatureEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的预期形状: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)




class MultiClassTransformer(nn.Module):
    def __init__(self, num_classes, input_dim, point_cloud_extractor,
                 feat_emb_dim=256, model_dim=64, nhead=4,
                 num_encoder_layers=3, dim_feedforward=256, dropout=0.1):
        """
        多类别帧级分类 Transformer
        """
        super(MultiClassTransformer, self).__init__()
        self.model_dim = model_dim

        # 原始特征编码器
        self.feature_encoder = FeatureEncoder(input_dim, feat_emb_dim)

        # 点云特征提取器
        self.point_cloud_extractor = point_cloud_extractor  # 假设输出维度为256

        # 融合后投影到Transformer维度
        fused_input_dim = feat_emb_dim + 256
        self.input_projection = nn.Linear(fused_input_dim, model_dim)

        # 位置编码
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # batch在第一维
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        # 输出分类层
        self.output_layer = nn.Linear(model_dim, num_classes)

    def forward(self, src, point_cloud, src_key_padding_mask=None):
        """
        src: [B, T, C1] 原始特征序列
        point_cloud: [B, T, N, 3] 点云序列
        """
        B, T, N, D = point_cloud.shape

        # 1. 编码原始特征
        src_encoded = self.feature_encoder(src)  # [B, T, feat_emb_dim]

        # 2. 提取每帧点云特征
        point_cloud_reshaped = point_cloud.view(B * T, N, D)  # 合并 batch 和时间维
        point_feats = self.point_cloud_extractor(point_cloud_reshaped)  # [B*T, 256]
        point_feats = point_feats.view(B, T, -1)  # [B, T, 256]

        # 3. 拼接编码后的原始特征与点云特征
        fused = torch.cat([src_encoded, point_feats], dim=-1)  # [B, T, feat_emb+256]

        # 4. 投影到Transformer输入维度
        src_proj = self.input_projection(fused) * math.sqrt(self.model_dim)

        # 5. 加位置编码
        src_transposed = src_proj.transpose(0, 1)  # [T, B, model_dim]
        src_with_pos = self.pos_encoder(src_transposed)
        src_final = src_with_pos.transpose(0, 1)  # [B, T, model_dim]

        output = self.transformer_encoder(src_final, src_key_padding_mask=src_key_padding_mask)

        output_logits = self.output_layer(output)  # [B, T, num_classes]
        return output_logits

    
