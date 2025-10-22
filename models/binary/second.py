class TwoStageModel(nn.Module):
    def __init__(self, segmenter, refiner):
        super().__init__()
        self.segmenter = segmenter  # 第一阶段：粗分割模型
        self.refiner = refiner      # 第二阶段：精细回归模型

    def forward(self, x):
        coarse_out = self.segmenter(x)             # [B, T]
        coarse_prob = torch.sigmoid(coarse_out)    # 概率
        
        topk_ids = get_topk_predictions(coarse_prob, topk=6)  # [B, topk]
        refined_preds = self.refiner(x, topk_ids)             # [B, topk] refined frame positions or confidence
        
        return coarse_prob, refined_preds, topk_ids
