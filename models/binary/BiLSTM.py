import torch
import torch.nn as nn

class BiLSTMForSegmentation(nn.Module):
    """
    一个用于序列分割（逐帧分类）的双向LSTM模型。

    Args:
        input_dim (int): 输入特征的维度 (例如，你的数据是5)。
        hidden_dim (int): LSTM隐藏层的维度。这个值越大，模型容量越大。
        num_layers (int): LSTM堆叠的层数。增加层数可以学习更复杂的模式。
        dropout (float): 在LSTM层之间以及输出层之前的dropout比率，用于防止过拟合。
    """
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=2, dropout=0.3):
        super(BiLSTMForSegmentation, self).__init__()
        
        # 定义BiLSTM层
        # - batch_first=True 让输入/输出的张量形状为 [batch_size, seq_len, features]，这更直观。
        # - bidirectional=True 开启双向处理。
        # - dropout 参数在num_layers > 1时，在堆叠的LSTM层之间添加dropout。
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 定义一个额外的Dropout层，用于输出层之前
        self.dropout_layer = nn.Dropout(dropout)
        
        # 定义输出层
        # BiLSTM的输出特征维度是 hidden_dim * 2 (前向hidden_dim + 后向hidden_dim)
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
        # 通过BiLSTM层
        # lstm_out 的形状是 [batch_size, seq_len, hidden_dim * 2]
        # _ (下划线) 用来接收最后一个时间点的 hidden_state 和 cell_state，这里我们不需要
        lstm_out, _ = self.lstm(x)
        
        # 应用Dropout
        out = self.dropout_layer(lstm_out)
        
        # 通过输出层得到每个时间点的logits
        # output_layer作用于最后一个维度，输出形状为 [batch_size, seq_len, 1]
        logits = self.output_layer(out)
        
        # 移除最后一个维度，得到 [batch_size, seq_len] 以匹配标签形状
        return logits.squeeze(-1)