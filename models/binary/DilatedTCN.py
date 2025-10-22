import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, dropout_p=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p), 
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.downsample(x)

# 步骤2: 修改DilatedTCNSegmenter以传递dropout_p
class DilatedTCNSegmenter(nn.Module):
    def __init__(self, in_channels=5, num_channels=[32, 64, 64, 64], dropout_p=0.3):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i-1]
            # 将dropout_p传递给每一个Block
            layers.append(DilatedTCNBlock(in_ch, out_ch, dilation, dropout_p=dropout_p))
            
        self.network = nn.Sequential(*layers)
        self.out_head = nn.Conv1d(num_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.network(x)
        logits = self.out_head(x)
        return logits.squeeze(1)


class CausalConv1d(nn.Module):
    """
    一个实现了因果卷积的模块。
    通过在左侧填充然后进行标准卷积，最后裁掉右侧多余部分来实现。
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        # 计算左侧需要填充的数量
        self.padding = (kernel_size - 1) * dilation
        
        # 定义标准卷积，注意这里内部不进行任何填充 (padding=0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=self.padding, **kwargs)

    def forward(self, x):
        # 卷积操作已经包含了左侧填充
        x = self.conv(x)
        # 裁掉右侧因填充而多出来的部分，以保证输出长度不变
        return x[:, :, :-self.padding]

class CausalTCNBlock(nn.Module):
    """
    标准的因果TCN残差模块。
    结构: CausalConv -> BN -> ReLU -> Dropout -> CausalConv -> BN -> ReLU -> Dropout
         然后将结果与输入相加，再通过一个ReLU。
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_p=0.3):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        
        # 如果输入输出通道数不同，使用1x1卷积进行下采样以匹配维度
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.conv_block(x)
        return self.final_relu(out + res)

class CausalTCNSegmenter(nn.Module):
    """
    用于语义分割的完整因果TCN模型。
    """
    def __init__(self, in_channels=5, num_classes=6, num_channels=[32, 64, 64, 64], kernel_size=3, dropout_p=0.3):
        super().__init__()
        
        layers = []
        channels = [in_channels] + num_channels
        
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = channels[i]
            out_ch = channels[i+1]
            layers.append(CausalTCNBlock(in_ch, out_ch, kernel_size, dilation, dropout_p=dropout_p))
        
        self.network = nn.Sequential(*layers)
        
        # 输出头的通道数现在由 num_classes 决定
        self.out_head = nn.Conv1d(num_channels[-1], num_classes, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, x):
        # x: [Batch, Time, Features]
        x = x.permute(0, 2, 1) # -> [Batch, Features, Time]
        
        x = self.network(x)
        logits = self.out_head(x) # -> [Batch, NumClasses, Time]
        
        # 如果是二分类（num_classes=1），则压缩维度以匹配BCE Loss
        if self.num_classes == 1:
            return logits.squeeze(1) # -> [Batch, Time]
        
        # 如果是多分类，保持 [B, C, T] 格式以匹配 CrossEntropyLoss
        return logits