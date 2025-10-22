import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    一个适用于多类别分类的Focal Loss实现。
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        """
        Args:
            weight (torch.Tensor, optional): 一个给每个类别的手动重缩放权重。
                                             就像您在CrossEntropyLoss中用的那样。
            gamma (float, optional): 聚焦参数。默认为 2.0。
            reduction (str, optional): 指定应用于输出的缩减方式：
                                       'none' | 'mean' | 'sum'。默认为 'mean'。
        """
        super(FocalLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): 模型的原始输出 (logits)，形状为 (N, C)。
            targets (torch.Tensor): 真实标签，形状为 (N,)。
        """
        # 计算标准的交叉熵损失，但不进行缩减，以便我们可以获取每个样本的损失
        # cross_entropy_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 为了数值稳定性，最好手动计算log_softmax和nll_loss
        log_softmax = F.log_softmax(inputs, dim=1)
        
        # 从log_softmax结果中收集对应真实类别的预测概率的对数
        # targets.view(-1, 1) 将 (N,) 变为 (N, 1) 以便gather操作
        log_pt = log_softmax.gather(1, targets.view(-1, 1))
        log_pt = log_pt.view(-1)
        
        # 计算实际的概率 pt
        pt = log_pt.exp().clamp(min=1e-7, max=1.0)

    
        # 如果有类别权重，则从权重张量中获取对应每个样本的权重
        if self.weight is not None:
            # self.weight必须在与targets相同的设备上
            weights = self.weight.gather(0, targets)
            # CE Loss部分： -weights * log_pt
            cross_entropy_loss = -weights * log_pt
        else:
            # CE Loss部分： -log_pt
            cross_entropy_loss = -log_pt

        # 计算Focal Loss的调制因子：(1 - pt)^gamma
        focal_term = (1 - pt)**self.gamma
        
        # 最终的Focal Loss
        loss = focal_term * cross_entropy_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss