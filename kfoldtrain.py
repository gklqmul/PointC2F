import pandas as pd
from sklearn.model_selection import KFold
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
# from models.multi.transformer import MultiClassTransformer
from models.multi.bi_lstm import  MultiClassBiLSTMNEW, ShuffledBiLSTM
from models.multi.bi_gru import MultiClassGRU, MultiClassGRUOLD
from models.multi.tcn import TCNClassifier
from models.pointnet import PointNetClassifier, PointNetFeatureExtractor, SimplePointNetFeatureExtractor
from models.newpointnet import ImprovedPointNetExtractorNEW, LightweightPointCloudEncoderWithGlobalAttention, PointCloudEncoderGeomRadar

from test import sliding_window_voting
from train import evaluate, train_one_epoch
from utils.datautils import generate_and_save_scaler, my_collate_fn, pad_collate_fn
from utils.tools import compare_encoders, compute_speed_metrics, compute_speeds, compute_speeds_with_bias_correction, evaluate_comprehensive_metrics, get_model_predictions, plot_gait_speeds_violin

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = "allpoints" 
SCALER_PATH = "scaler.gz"
all_sample_ids = sorted([os.path.splitext(os.path.basename(f))[0].rsplit('-', 1)[0] 
                        for f in glob(os.path.join(ROOT_DIR, '*-motion.csv'))])

# files = glob(os.path.join(ROOT_DIR, '*-motion.csv'))
# all_sample_ids = sorted({
#         sid
#         for f in files
#         for sid in [os.path.splitext(os.path.basename(f))[0].rsplit('-', 1)[0]]
#         if len(sid.split('-')) >= 2 and sid.split('-')[1].isdigit() and int(sid.split('-')[1]) < 4
# })

kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_fold_results = []

fold_metrics = []
with open(os.path.join(ROOT_DIR, 'newlabel.json'), 'r') as f:
        label_dict = json.load(f)
# if not os.path.exists(SCALER_PATH):     
#         generate_and_save_scaler(ROOT_DIR, train_ids, SCALER_PATH)

scaler = joblib.load(SCALER_PATH)
INPUT_DIM_1D = 36
POINT_FEATURE_DIM = 256
NUM_CLASSES = 6 
all_dfs = []
for fold, (train_idx, test_idx) in enumerate(kf.split(all_sample_ids)):
    print(f"Fold {fold+1}")
    
    train_ids = [all_sample_ids[i] for i in train_idx]
    test_ids  = [all_sample_ids[i] for i in test_idx]
    
    # 可以再在 train_ids 中划分 val_ids（比如 10%）
    train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=42)
    
    # 创建 Dataset 和 DataLoader
    train_dataset = TUGFeatureDataset(ROOT_DIR, train_ids, label_dict, scaler, augment=True)
    val_dataset   = TUGFeatureDataset(ROOT_DIR, val_ids,   label_dict, scaler, augment=False)
    test_dataset  = TUGFeatureDataset(ROOT_DIR, test_ids,  label_dict, scaler, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=my_collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, collate_fn=my_collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=4, shuffle=False, collate_fn=my_collate_fn)
    

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
    for param in point_cloud_extractor.parameters():
            param.requires_grad = False
    print("已冻结 PointNet 权重")
    model = MultiClassGRU(NUM_CLASSES, INPUT_DIM_1D, point_cloud_extractor).to(device)
    # model = MultiClassBiLSTMNEW(
    #     num_classes=NUM_CLASSES,
    #     input_dim=INPUT_DIM_1D,  # 原始1D特征维度
    #     point_cloud_extractor=point_cloud_extractor,
    #     hidden_dim=128,  # LSTM隐藏层大小，可以调整
    #     num_layers=2,    # LSTM层数，可以调整
    #     dropout=0.2
    # ).to(device)
    
    # model = MultiClassTransformer(
    #     num_classes=NUM_CLASSES,
    #     input_dim=INPUT_DIM_1D,  # 原始1D特征维度
    #     point_cloud_extractor=point_cloud_extractor
    # ).to(device)

    # model = TCNClassifier(
    #      num_classes=NUM_CLASSES,
    #         input_dim=INPUT_DIM_1D,  # 原始1D特征维度
    #         point_cloud_extractor=point_cloud_extractor
    # ).to(device)

    # model = PointNetClassifier(
    #     num_classes=NUM_CLASSES,
    #     input_dim=INPUT_DIM_1D,  # PointNet++输出的点云特征维度
    #     point_cloud_extractor=point_cloud_extractor
    # ).to(device)

    # model = MLPClassifier(
    #     num_classes=NUM_CLASSES,
    #     input_dim=INPUT_DIM_1D,  # 原始1D特征维度
    #     point_cloud_extractor=point_cloud_extractor,
    #     hidden_dim=512,  # MLP隐藏层维度
    #     dropout=0.5
    # ).to(device)
    # model = CNN1DClassifier(
    #     num_classes=NUM_CLASSES,
    #     input_dim=INPUT_DIM_1D,  # 原始1D特征维度
    #     point_cloud_extractor=point_cloud_extractor,  # 使用 PointNet 特征提取器
    #     dropout=0.2  # Dropout比例
    # ).to(device)
    # model = ShuffledBiLSTM(
    #     num_classes=NUM_CLASSES,
    #     input_dim=INPUT_DIM_1D,  # 原始1D特征维度
    #     point_cloud_extractor=point_cloud_extractor,  # 使用 PointNet 特征提取器
    #     hidden_dim=128,  # LSTM隐藏层大小，可以调整
    #     num_layers=2,    # LSTM层数，可以调整
    #     dropout=0.1,
    #     shuffle_time=True  # 启用时间维度随机打乱
    # ).to(device)
    class_weights = 1.0/torch.tensor([ 10, 1, 4, 1, 4, 4], dtype=torch.float)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = WeightedFocalLoss(gamma=2.0, weight=class_weights).to(device)
    
    # best_val_f1 = 0
    # epochs = 64
    # for epoch in range(epochs):
    #     train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, NUM_CLASSES)
    #     val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    #     print(f"Epoch {epoch}/{epochs}:")
    #     print(f"  Train Loss: {train_loss:.8f}")
    #     print(f"  Val Loss: {val_loss:.8f}, Acc: {val_acc:.4f}")
    #     if val_acc > best_val_f1:
    #         best_val_f1 = val_acc
    #         best_model_state = model.state_dict()
    #     model_path = f"./dark/fold{fold+1}_model.pth"
    #     torch.save(best_model_state, model_path)
    model_path = f"./GRU/fold{fold+1}_model.pth"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    test_metrics = evaluate(model, test_loader, criterion, device)
    fold_metrics.append(test_metrics)
    metrics_path = f"./GRU/fold{fold+1}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f)
        final_results = get_model_predictions(
        model,
        test_loader,
        device,
        post_process=True,
        correction_fn=sliding_window_voting
    )
    
    # 分段指标
    segmentation_metrics = evaluate_comprehensive_metrics(
        all_true_labels=final_results['true_labels'],
        all_pred_labels=final_results['processed_predictions'],
        num_classes=5
    )
    
    # 速度指标
    speed = compute_speeds_with_bias_correction(final_results['true_labels'], 
                           final_results.get('processed_predictions', None), 
                           test_ids)
    speed_metrics = compute_speed_metrics(speed)
    all_dfs.append(speed)
    # 汇总每折结果
    fold_result = {
        "fold": fold+1,
        **{f"raw_{k}": v for k,v in final_results['raw_metrics'].items()},
        **{f"processed_{k}": v for k,v in final_results['processed_metrics'].items()},
        **{f"seg_{k}": v for k,v in segmentation_metrics.items()},
        **{f"speed_{k}": v for k,v in speed_metrics.items()}
    }
    all_fold_results.append(fold_result)

combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df.to_excel("all_folds_speed.xlsx", index=False)
compute_speed_metrics(combined_df)    
plot_gait_speeds_violin(combined_df)

df_results = pd.DataFrame(all_fold_results)
df_results.to_excel("./GRU/light.xlsx", index=False)
print("✅ 所有折测试结果已保存到 light.xlsx")

avg_metrics = np.mean(fold_metrics, axis=0)
std_metrics = np.std(fold_metrics, axis=0)
print("Average metrics:", avg_metrics)
print("Std metrics:", std_metrics)
