import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerKeyframeModel(nn.Module):
    def __init__(self, input_dim=5, model_dim=64, nhead=4, num_encoder_layers=3, dim_feedforward=256, dropout=0.1):
        """
        Args:
            input_dim: 输入特征的维度 (e.g., 5)
            model_dim: Transformer内部的特征维度 (必须能被nhead整除)
            nhead: 多头注意力的头数
            num_encoder_layers: Transformer Encoder的层数
        """
        super(TransformerKeyframeModel, self).__init__()
        self.model_dim = model_dim
        
        # 1. 输入线性层：将输入维度映射到模型维度
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        
        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)
        
        # 4. 输出层：将Transformer的输出映射回一个logit值，用于二分类
        self.output_layer = nn.Linear(model_dim, 1)
        
    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]
        
        # 1. 投影到模型维度并进行缩放
        src = self.input_projection(src) * math.sqrt(self.model_dim) # Shape: [batch_size, seq_len, model_dim]
        
        # 2. 添加位置编码 (核心修复部分)
        #    为了匹配 PositionalEncoding 的 [seq_len, batch, dim] 格式，我们需要转置
        src = src.transpose(0, 1) # Shape: [seq_len, batch_size, model_dim]
        src = self.pos_encoder(src) # 应用位置编码
        src = src.transpose(0, 1) # 转置回来以匹配 batch_first=True 的 Transformer
                                  # Shape: [batch_size, seq_len, model_dim]
        
        # 3. 通过Transformer Encoder
        output = self.transformer_encoder(src) # Shape: [batch_size, seq_len, model_dim]
        
        # 4. 输出分类 Logit
        output = self.output_layer(output) # Shape: [batch_size, seq_len, 1]
        
        return output.squeeze(-1) # Shape: [batch_size, seq_len]

# --- 在 main 函数中使用 ---
# model = TransformerKeyframeModel(input_dim=5, model_dim=128, nhead=8, num_encoder_layers=4).to(device)