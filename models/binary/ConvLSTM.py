import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn



class ConvLSTMModel(nn.Module):
    def __init__(self, input_dim=5, conv_out_channels=32, kernel_size=3, lstm_hidden_dim=64, num_lstm_layers=2, dropout=0.2):
        super(ConvLSTMModel, self).__init__()
        
        # 1. 1D卷积层
        self.conv1d = nn.Conv1d(in_channels=input_dim, 
                                out_channels=conv_out_channels, 
                                kernel_size=kernel_size, 
                                padding='same') # 'same' padding 保持序列长度不变
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 2. LSTM层
        self.lstm = nn.LSTM(input_size=conv_out_channels, 
                            hidden_size=lstm_hidden_dim, 
                            num_layers=num_lstm_layers,
                            batch_first=True,  # 输入输出为 [batch, seq, feature]
                            bidirectional=True) # 双向LSTM通常效果更好

        # 3. 输出层
        # 因为是双向LSTM，所以输出维度是 2 * lstm_hidden_dim
        self.output_layer = nn.Linear(2 * lstm_hidden_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Conv1d需要 [batch, channels, length]
        x = x.permute(0, 2, 1) # -> [batch_size, input_dim, seq_len]
        
        conv_out = self.conv1d(x)
        conv_out = self.relu(conv_out)
        
        # 变回 [batch, seq, feature] 以输入LSTM
        lstm_in = conv_out.permute(0, 2, 1) # -> [batch_size, seq_len, conv_out_channels]
        lstm_in = self.dropout(lstm_in)

        lstm_out, _ = self.lstm(lstm_in) # lstm_out shape: [batch_size, seq_len, 2 * hidden_dim]
        
        output = self.output_layer(lstm_out) # -> [batch_size, seq_len, 1]
        
        return output.squeeze(-1) # -> [batch_size, seq_len]

# --- 在 main 函数中使用 ---
# model = ConvLSTMModel(input_dim=5).to(device)