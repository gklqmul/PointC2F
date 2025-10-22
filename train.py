
import json

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from glob import glob
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader


from models.Pointattention import FramePointCloudEncoder
from models.multi.cnn import CNN1DClassifier
# from models.newpointnet import ImprovedPointNetExtractor, NewPointNetClassifier
from models.multi.dgcnn import DGCNN
from models.multi.pointmlp import PointMLP
from models.multi.pointtransformer import PointTransformer
from models.multi.pointnetsimple import PointNetPPFrameClassifier
from dataset import TUGFeatureDataset
from loss.focalbceloss import FocalLoss, WeightedFocalLoss
# from multi.point_transformerold import PointTransformerClassifier
from models.multi.mlp import MLPClassifier
from models.multi.transformer import MultiClassTransformer
from models.multi.bi_lstm import MultiClassBiLSTMNEW, ShuffledBiLSTM
from models.multi.bi_gru import MultiClassGRU, MultiClassGRUOLD
from models.multi.tcn import TCNClassifier
from models.pointnet import PointNetClassifier, PointNetFeatureExtractor, SimplePointNetFeatureExtractor
from models.newpointnet import ImprovedPointNetExtractorNEW
from utils.datautils import generate_and_save_scaler, my_collate_fn, pad_collate_fn
from utils.tools import compare_encoders


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    total_loss = 0
    
    for idx, batch in enumerate(dataloader):
        x = batch['input'].to(device)      # [B, T, feature_dim]
        z = batch['points'].to(device)  # [B, T, 3]，如果有点云特征
        y = batch['label'].to(device)      # [B, T]
        padding_mask = batch['padding_mask'].to(device)  # [B, T], 用于处理变长序列

        # out = model(x, z, src_key_padding_mask=padding_mask)  # 模型输出形状: [B, T, C]

        # # out = model(z)
        # loss_input = out.contiguous().view(-1, num_classes)

        # # 将 y 从 [B, T] -> [B*T]
        # loss_target = y.view(-1)

        # 假设 padding_mask 是 bool 类型，True 表示需要忽略
        # valid_mask = ~padding_mask  # 有效位置为 True
        # loss_input = out[valid_mask, :]  # 形状 [有效点数, C]
        # loss_target = y[valid_mask]      # 形状 [有效点数]
        
        # loss = criterion(loss_input, loss_target, loss_target)
            # 识别有效的标签
        valid_mask = (y != -1)
        
        # 动态计算有效帧的边界权重
        # 这部分是新加的，它直接在有效的标签上工作
        y_flat = y.view(-1)
        y_valid = y_flat[valid_mask.view(-1)]
        
        # 如果当前批次没有有效帧，则跳过
        if y_valid.numel() == 0:
            continue
            
        label_changes = (torch.diff(y_valid) != 0).nonzero().squeeze(1)
        weights = torch.full_like(y_valid, 1.0, dtype=torch.float32)

        # 为变化点附近的窗口赋予更高的权重
        for change_idx in label_changes:
            start_idx = max(0, change_idx - 5 + 1)
            end_idx = min(len(y_valid), change_idx + 5 + 1)
            weights[start_idx:end_idx] = 5.0
        
        # 模型前向传播，确保模型能处理填充
        out = model(x, z, src_key_padding_mask=padding_mask)
        
        # 展平输出并筛选掉填充帧的预测
        out_flat = out.view(-1, num_classes)
        out_valid = out_flat[valid_mask.view(-1)]

        # 将有效帧的输出、标签和权重传递给损失函数
        loss = criterion(out_valid, y_valid, weights)

        # 4. 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    total_correct_frames = 0
    total_frames = 0

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            x = batch['input'].to(device)
            z = batch['points'].to(device)  # 如果有点云特征
            y = batch['label'].to(device)
            padding_mask = batch['padding_mask'].to(device)

            out = model(x,z, src_key_padding_mask=padding_mask)
            # out = model(z)
            min_T = min(out.shape[1], y.shape[1])
            out = out[:, :min_T, :]  # [B, min_T, C]
            y = y[:, :min_T]         # [B, min_T]
            valid_mask = ~padding_mask[:, :min_T]  # [B, min_T]

            # 计算正确预测
            preds = torch.argmax(out, dim=2)  # [B, min_T]
            correct_preds = (preds == y) & valid_mask

            # 统计
            total_correct_frames += correct_preds.sum().item()
            total_frames += valid_mask.sum().item()

            B, T, C = out.shape
            out_flat = out.reshape(B * T, C)
            y_flat = y.reshape(B * T)
            mask_flat = valid_mask.reshape(B * T)
            
            valid_out = out_flat[mask_flat]
            valid_y = y_flat[mask_flat]

            # valid_mask = (y != -1)
        
            # 动态计算有效帧的边界权重
            # 这部分是新加的，它直接在有效的标签上工作

            label_changes = (torch.diff(valid_y) != 0).nonzero().squeeze(1)
            weights = torch.full_like(valid_y, 1.0, dtype=torch.float32)

            # 为变化点附近的窗口赋予更高的权重
            for change_idx in label_changes:
                start_idx = max(0, change_idx - 5 + 1)
                end_idx = min(len(valid_y), change_idx + 5 + 1)
                weights[start_idx:end_idx] = 5.0
            if len(valid_y) > 0:
                    loss = criterion(valid_out, valid_y, weights=weights)
                    total_loss += loss.item()

    # 计算最终指标
    avg_loss = total_loss / len(dataloader) if total_loss > 0 else 0
    accuracy = total_correct_frames / total_frames if total_frames > 0 else 0

    return avg_loss, accuracy




def main():

    ROOT_DIR = "allpoints" 
    SCALER_PATH = "scaler.gz"

    all_sample_ids = sorted([os.path.splitext(os.path.basename(f))[0].rsplit('-', 1)[0] 
                        for f in glob(os.path.join(ROOT_DIR, '*-motion.csv'))])

    files = glob(os.path.join(ROOT_DIR, '*-motion.csv'))

    # all_sample_ids = sorted({
    #     sid
    #     for f in files
    #     for sid in [os.path.splitext(os.path.basename(f))[0].rsplit('-', 1)[0]]
    #     if len(sid.split('-')) >= 2 and sid.split('-')[1].isdigit() and int(sid.split('-')[1]) >= 4
    # })
    with open(os.path.join(ROOT_DIR, 'newlabel.json'), 'r') as f:
        label_dict = json.load(f)


    
    train_val_ids, test_ids = train_test_split(all_sample_ids, test_size=0.1, random_state=42)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.11, random_state=42)

    if not os.path.exists(SCALER_PATH):     
        generate_and_save_scaler(ROOT_DIR, train_ids, SCALER_PATH)

    scaler = joblib.load(SCALER_PATH)

    train_dataset = TUGFeatureDataset(
        root_dir=ROOT_DIR, 
        sample_ids=train_ids, 
        label_dict=label_dict, 
        scaler=scaler, 
        augment=True 
    )
    
    val_dataset = TUGFeatureDataset(
        root_dir=ROOT_DIR, 
        sample_ids=val_ids, 
        label_dict=label_dict, 
        scaler=scaler, 
        augment=False
    )
    
    test_dataset = TUGFeatureDataset(
        root_dir=ROOT_DIR, 
        sample_ids=test_ids, 
        label_dict=label_dict, 
        scaler=scaler, 
        augment=False
    )

    print(f"dataset size: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=my_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=my_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=my_collate_fn)
    
    INPUT_DIM_1D = 36
    POINT_FEATURE_DIM = 256
    NUM_CLASSES = 6 

    point_cloud_extractor = ImprovedPointNetExtractorNEW(output_dim=256)

    model_weights_path = "./softedgecleanpoints5dseprate.pth"
    if os.path.exists(model_weights_path):
            print(f"正在从 {model_weights_path} 加载已适配的PointNet++权重...")
            try:
                state_dict = torch.load(model_weights_path, map_location='cpu')
                point_cloud_extractor.load_state_dict(state_dict)
                point_cloud_extractor = point_cloud_extractor.to(device)
                print("✅ 权重加载成功！")
            except Exception as e:
                print(f"❌ 加载权重失败: {e}")
    else:
            print(f"⚠️ 未找到 PointNet 权重文件，使用随机初始化")

    # normal_pointnet = SimplePointNetFeatureExtractor(output_dim=POINT_FEATURE_DIM).to(device)
    # extractor_ckpt_path = "best_point_cloud_extractor5D.pt"
    # if os.path.exists(extractor_ckpt_path):
    #     ckpt = torch.load(extractor_ckpt_path, map_location=device)
    #     # 兼容：既支持直接state_dict也支持包装
    #     if isinstance(ckpt, dict) and 'extractor_state_dict' in ckpt:
    #         state_dict = ckpt['extractor_state_dict']
    #         print(f"加载提取器权重 (epoch={ckpt.get('epoch')}, val_f1={ckpt.get('val_f1')})")
    #     else:
    #         state_dict = ckpt
    #         print("加载直接 state_dict 权重")

    #     # 严格加载；若结构不同可改 strict=False
    #     missing, unexpected = normal_pointnet.load_state_dict(state_dict, strict=False)
    #     if missing:
    #         print("缺失参数:", missing)
    #     if unexpected:
    #         print("多余参数:", unexpected)
    # else:
    #     print("未找到提取器权重文件，使用随机初始化")
    # # 使用示例
    # compare_encoders(point_cloud_extractor, normal_pointnet, test_dataset)

    for param in point_cloud_extractor.parameters():
            param.requires_grad = False
    print("已冻结 PointNet 权重")

    # model = MultiClassBiLSTM(
    #     num_classes=NUM_CLASSES,
    #     input_dim=INPUT_DIM_1D,  # 原始1D特征维度
    #     point_cloud_extractor=point_cloud_extractor,
    #     hidden_dim=128,  # LSTM隐藏层大小，可以调整
    #     num_layers=2,    # LSTM层数，可以调整
    #     dropout=0.2
    # ).to(device)

    model = MultiClassTransformer(
        num_classes=NUM_CLASSES,
        input_dim=INPUT_DIM_1D,  # 原始1D特征维度
        point_cloud_extractor=point_cloud_extractor
    ).to(device)

    # model = TCNClassifier(
    #      num_classes=NUM_CLASSES,
    #         input_dim=INPUT_DIM_1D,  # 原始1D特征维度
    #         point_cloud_extractor=point_cloud_extractor
    # ).to(device)

    # model = MultiClassGRU(
    #     num_classes=NUM_CLASSES,
    #     input_dim=INPUT_DIM_1D,  # 原始1D特征维度
    #     point_cloud_extractor=point_cloud_extractor,  # 使用 PointNet 特征提取器
    # ).to(device)

    # model = PointNetClassifier(
    #     num_classes=NUM_CLASSES,
    #     input_dim=POINT_FEATURE_DIM+INPUT_DIM_1D,  # PointNet++输出的点云特征维度
    #     point_cloud_extractor=normal_pointnet
    # ).to(device)

    # model = NewPointNetClassifier(
    #     num_classes=NUM_CLASSES,
    #     input_dim=POINT_FEATURE_DIM+INPUT_DIM_1D,  # PointNet++输出的点云特征维度
    #     point_cloud_extractor=point_cloud_extractor
    # ).to(device)

    # model = MLPClassifier(
    #     num_classes=NUM_CLASSES,
    #     input_dim=INPUT_DIM_1D + POINT_FEATURE_DIM,  # 原始1D特征维度
    #     point_cloud_extractor=point_cloud_extractor,
    #     hidden_dim=256,  # MLP隐藏层维度
    #     dropout=0.5
    # ).to(device)

    model = ShuffledBiLSTM(
        num_classes=NUM_CLASSES,
        input_dim=INPUT_DIM_1D + POINT_FEATURE_DIM,  # 原始1D特征维度
        point_cloud_extractor=point_cloud_extractor,  # 使用 PointNet 特征提取器
        hidden_dim=128,  # LSTM隐藏层大小，可以调整
        num_layers=2,    # LSTM层数，可以调整
        dropout=0.1,
        shuffle_time=True  # 启用时间维度随机打乱
    ).to(device)

    # model = CNN1DClassifier(
    #     num_classes=NUM_CLASSES,
    #     input_dim=INPUT_DIM_1D+POINT_FEATURE_DIM,  # 原始1D特征维度
    #     point_cloud_extractor=point_cloud_extractor,  # 使用 PointNet 特征提取器
    #     dropout=0.2  # Dropout比例
    # ).to(device)

    # model = DGCNN().to(device)
    # model = PointMLP(input_channels=9).to(device)
    # model = PointTransformer().to(device)
    # model = PointNetPPFrameClassifier().to(device)

    class_weights = 1.0/torch.tensor([ 10, 1, 4, 1, 4, 4], dtype=torch.float)
    # criterion = FocalLoss(gamma=4.0, weight=class_weights).to(device)
    criterion = WeightedFocalLoss(gamma=2.0, weight=class_weights).to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)  # 忽略填充标签
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_val_f1 = 0
    epochs = 100
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, NUM_CLASSES)
        avg_loss, acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{epochs}:")
        print(f"  Train Loss: {train_loss:.8f}")
        print(f"  Val Loss: {avg_loss:.8f}, Acc: {acc:.4f}")

        if acc > best_val_f1:
            best_val_f1 = acc
            torch.save(model.state_dict(), "GRUsoftedgecleanpoints5dsepratereal.pt")

    #         # 单独保存特征提取器
    #         torch.save({
    #             'extractor_state_dict': model.get_extractor_state_dict(),
    #             'val_f1': best_val_f1,
    #             'epoch': epoch
    #         }, "best_point_cloud_extractor5D.pt")
            
    #         print(f"Saved best model and extractor with F1: {best_val_f1:.4f}")

    
if __name__ == '__main__':
    main()