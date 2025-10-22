import torch
import torch.nn as nn

class BiGRUForSegmentation(nn.Module):
    """
    一个用于序列分割（逐帧分类）的双向GRU模型。
    该结构与提供的BiLSTM版本完全对应。

    Args:
        input_dim (int): 输入特征的维度 (例如，7)。
        hidden_dim (int): GRU隐藏层的维度。这个值越大，模型容量越大。
        num_layers (int): GRU堆叠的层数。增加层数可以学习更复杂的模式。
        dropout (float): 在GRU层之间以及输出层之前的dropout比率，用于防止过拟合。
    """
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=2, dropout=0.3):
        super(BiGRUForSegmentation, self).__init__()
        
        # 定义BiGRU层
        # - 将 nn.LSTM 替换为 nn.GRU
        # - 其他参数 (batch_first, bidirectional, dropout) 与LSTM版本完全相同。
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 定义一个额外的Dropout层，用于输出层之前 (结构保持不变)
        self.dropout_layer = nn.Dropout(dropout)
        
        # 定义输出层
        # BiGRU的输出特征维度同样是 hidden_dim * 2 (前向hidden_dim + 后向hidden_dim)
        # 这个全连接层将为序列中的每一个时间点，将高维特征映射回一个单独的logit值。
        self.output_layer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: 输出的logits张量，形状为 [batch_size, seq_len]
        """
        # 通过BiGRU层
        # gru_out 的形状是 [batch_size, seq_len, hidden_dim * 2]
        # GRU只有一个隐藏状态h_n，没有细胞状态c_n。我们同样用 _ 忽略它。
        gru_out, _ = self.gru(x)
        
        # 应用Dropout
        out = self.dropout_layer(gru_out)
        
        # 通过输出层得到每个时间点的logits
        # output_layer作用于最后一个维度，输出形状为 [batch_size, seq_len, 1]
        logits = self.output_layer(out)
        
        # 移除最后一个维度，得到 [batch_size, seq_len] 以匹配标签形状
        return logits.squeeze(-1)