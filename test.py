from collections import Counter
import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from glob import glob
import os
from tqdm import tqdm
import json
from loss.focalbceloss import FocalLoss, WeightedFocalLoss
from models.multi.cnn import CNN1DClassifier
from models.multi.mlp import MLPClassifier
from models.newpointnet import ImprovedPointNetExtractorNEW
from models.multi.pointnetsimple import PointNetPPFrameClassifier
from models.multi.dgcnn import DGCNN
from models.multi.pointmlp import PointMLP
from models.multi.pointtransformer import PointTransformer

from train import evaluate

from dataset import TUGFeatureDataset
from loss.focalbceloss import FocalLoss
from models.multi.transformer import MultiClassTransformer
from models.multi.bi_lstm import MultiClassBiLSTMNEW, ShuffledBiLSTM
from models.multi.bi_gru import MultiClassGRU, MultiClassGRUOLD
from models.multi.tcn import TCNClassifier
from models.pointnet import PointNetClassifier, PointNetFeatureExtractor
from utils.datautils import generate_and_save_scaler, my_collate_fn, pad_collate_fn
from utils.tools import compute_speed_metrics, compute_speeds, evaluate_and_process, evaluate_comprehensive_metrics, get_model_predictions, plot_confusion_matrix, plot_gait_speeds_violin, plot_labels_over_time

# from utils import plot_prob_and_label

def sliding_window_voting(logits_seq, window_size=10, stride=5, num_classes=6):
    """
    滑窗投票后处理函数示例：
    logits_seq: np.array, shape=(seq_len, num_classes), 模型原始输出概率或logits
    返回: np.array, shape=(seq_len,), 每帧投票后的类别预测

    逻辑：用滑动窗口覆盖序列，窗口内按预测类别计票，窗口中心帧得票最高的类别为投票结果。
    """
    logits_seq = np.array(logits_seq)
    seq_len = logits_seq.shape[0]
    preds = np.argmax(logits_seq, axis=1)
    voted_preds = np.zeros(seq_len, dtype=int)

    half_window = window_size // 2

    # 边界处理：对序列开头和结尾做简单复制投票
    for i in range(seq_len):
        start = max(0, i - half_window)
        end = min(seq_len, i + half_window + 1)
        window_preds = preds[start:end]
        vote_count = Counter(window_preds)
        voted_preds[i] = vote_count.most_common(1)[0][0]

    return voted_preds


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    
    test_dataset = TUGFeatureDataset(
        root_dir=ROOT_DIR, 
        sample_ids=test_ids, 
        label_dict=label_dict, 
        scaler=scaler, 
        augment=False
    )

   
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=my_collate_fn)

    INPUT_DIM_1D = 36
    POINT_FEATURE_DIM = 256
    NUM_CLASSES = 6 

    point_cloud_extractor = ImprovedPointNetExtractorNEW(output_dim=256)

    # model_weights_path = "./edgeattnpoint5D.pth"
   
    # model = MultiClassBiLSTM(
    #     num_classes=NUM_CLASSES,
    #     input_dim=INPUT_DIM_1D,  # 原始1D特征维度
    #     point_cloud_extractor=point_cloud_extractor,
    #     hidden_dim=128,  # LSTM隐藏层大小，可以调整
    #     num_layers=2,    # LSTM层数，可以调整
    #     dropout=0.2
    # ).to(device)

    # model = MultiClassTransformer(
    #     num_classes=NUM_CLASSES,
    #     input_dim=INPUT_DIM_1D + POINT_FEATURE_DIM,  # 原始1D特征维度
    #     point_cloud_extractor=point_cloud_extractor
    # ).to(device)


    # model = TCNClassifier(
    #     num_classes=NUM_CLASSES,
    #     input_dim=INPUT_DIM_1D + POINT_FEATURE_DIM,  # 原始1D特征维度
    #     point_cloud_extractor=point_cloud_extractor,  # 使用 PointNet 特征提取器
    #     hidden_dim=128,  # GRU隐藏层大小，可以调整
    #     num_layers=2,    # GRU层数，可以调整
    #     dropout=0.2
    # ).to(device)


    model = MultiClassGRU(
        num_classes=NUM_CLASSES,
        input_dim=INPUT_DIM_1D,  # 原始1D特征维度
        point_cloud_extractor=point_cloud_extractor,  # 使用 PointNet 特征提取器
        hidden_dim=128,  # GRU隐藏层大小，可以调整
        num_layers=2,    # GRU层数，可以调整
        dropout=0.2
    ).to(device)

    # model = PointNetClassifier(
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

    # model = ShuffledBiLSTM(
    #     num_classes=NUM_CLASSES,
    #     input_dim=INPUT_DIM_1D + POINT_FEATURE_DIM,  # 原始1D特征维度
    #     point_cloud_extractor=point_cloud_extractor,  # 使用 PointNet 特征提取器
    #     hidden_dim=128,  # LSTM隐藏层大小，可以调整
    #     num_layers=2,    # LSTM层数，可以调整
    #     dropout=0.1,
    #     shuffle_time=True  # 启用时间维度随机打乱
    # ).to(device)

    # model = CNN1DClassifier(
    #     num_classes=NUM_CLASSES,
    #     input_dim=INPUT_DIM_1D+POINT_FEATURE_DIM,  # 原始1D特征维度
    #     point_cloud_extractor=point_cloud_extractor,  # 使用 PointNet 特征提取器
    #     dropout=0.2  # Dropout比例
    # ).to(device)

    # model = DGCNN().to(device)
    # model = PointMLP(input_channels=9).to(device)
    # model = PointTransformer().to(device)
    # model = PointNetPPFrameClassifier(
    #     num_classes=6,
    #     use_extra_features=True  # 和训练时保持一致
    # ).to(device)
    class_weights = 1.0/torch.tensor([ 10, 1, 4, 1, 4, 4], dtype=torch.float)  # 如果没有类别不平衡问题，可以直接使用均匀权重
    # criterion = FocalLoss(gamma=4.0, weight=class_weights).to(device)
    criterion = WeightedFocalLoss(gamma=2.0, weight=class_weights).to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)  
 

    state_dict = torch.load("GRUsoftedgecleanpoints5dsepratereal.pt", map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print("Test:")
    print(f"  Loss: {test_loss:.8f}, Acc: {test_acc:.4f}, ")

    final_results = get_model_predictions(
        model,
        test_loader,
        device,
        post_process=True,
        correction_fn=sliding_window_voting
    )

    print("\n--- 性能对比 ---")
    print("\n原始模型性能:")
    for metric, value in final_results['raw_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    for metric, value in final_results['processed_metrics'].items():
        print(f"  {metric} (滑窗投票后): {value:.4f}")

    y_true = final_results['true_labels']
    y_pred = final_results.get('processed_predictions',None)
    y_raw = final_results.get('raw_predictions', None)
    # speed= compute_speeds(y_true, y_pred, test_ids)
    # plot_gait_speeds_violin(speed)
    # compute_speed_metrics( speed)
    
    # segmentation_metrics = evaluate_comprehensive_metrics(
    #     all_true_labels=final_results['true_labels'],
    #     all_pred_labels=final_results['processed_predictions'], # 使用处理后的预测结果
    #     num_classes=5
    # )
    

    # print(json.dumps(segmentation_metrics, indent=4))

    # (A) 整理 final_results 到一个 DataFrame
    # 合并 'raw_metrics' 和 'processed_metrics' 以便对比
    # df_overall = pd.DataFrame({
    #     '原始模型性能': pd.Series(final_results['raw_metrics']),
    #     '滑窗投票后': pd.Series(final_results['processed_metrics'])
    # })
    # df_overall.index.name = 'Metric'

    # all_phase_rows = []
    # phase_order = [p for p in results.keys() if p.startswith('Phase')] + ['Overall_Action'] # 保证顺序

    # for phase_name in phase_order:
    #     phase_data = results[phase_name]
        
    #     # 统一键名，去除特殊字符和空格
    #     clean_data = {
    #         key.replace('(s)', '').replace('(BA)', '').strip(): value
    #         for key, value in phase_data.items()
    #     }
        
    #     # 为每个Phase创建两行数据：一行Duration，一行Segment
    #     row_template = {'Phase': 'Total' if phase_name == 'Overall_Action' else phase_name}
        
    #     # Duration 行
    #     duration_row = row_template.copy()
    #     duration_row['Category'] = 'Duration'
    #     duration_metrics = ['RMSE', 'MAE', 'Correlation', 'ICC', 'Bias', 'LoA Lower', 'LoA Upper', 'SEM', 'MDC']
    #     for m in duration_metrics:
    #         duration_row[m] = clean_data.get(m)
    #     all_phase_rows.append(duration_row)

    #     # Segment 行
    #     segment_row = row_template.copy()
    #     segment_row['Category'] = 'Segment'
    #     segment_metrics = ['Precision', 'Recall', 'F1_Score', 'Start_MAE', 'End_MAE', 'IoU']
    #     for m in segment_metrics:
    #         segment_row[m] = clean_data.get(m)
    #     all_phase_rows.append(segment_row)

    #     # 创建包含所有Phase数据的DataFrame
    #     df_phases = pd.DataFrame(all_phase_rows)
    #     df_phases.set_index(['Phase', 'Category'], inplace=True)

    #     # 为了得到最终的Excel列布局，进行数据透视
    #     # 这会将Duration和Segment的指标放到不同的列组中
    #     df_phases = df_phases.stack().unstack(level=[1, 2])

    #     # 重命名列以匹配最终期望的格式
    #     df_phases.rename(columns={'F1_Score': 'F1', 'LoA Lower': 'LoA lower', 'Start_MAE': 'Start MAE (s)', 'End_MAE': 'End MAE (s)'}, inplace=True)

    #     # 调整列的顺序以匹配您之前的图片格式
    #     col_order = [
    #         ('Duration', 'RMSE'), ('Duration', 'MAE'),('Duration', 'MAPE'), ('Duration', 'Correlation'), ('Duration', 'ICC'),
    #         ('Duration', 'Bias'), ('Duration', 'LoA lower'), ('Duration', 'LoA Upper'),
    #         ('Duration', 'SEM'), ('Duration', 'MDC'),
    #         ('Segment', 'Precision'), ('Segment', 'Recall'), ('Segment', 'F1'),
    #         ('Segment', 'Start MAE (s)'), ('Segment', 'End MAE (s)'), ('Segment', 'IoU')
    #     ]
    #     # 过滤列，只保留存在的列，以防原始数据中某些列不存在
    #     existing_cols = [c for c in col_order if c in df_phases.columns]
    #     df_phases = df_phases[existing_cols]


    #     # --- 3. 将所有整理好的 DataFrames 写入同一个 Excel 文件 ---
    #     output_filename = 'consolidated_results.xlsx'
    #     with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    #         # 首先写入模型整体性能
    #         df_overall.to_excel(writer, sheet_name='Model Results')
            
    #         # 在下方空出两行后，写入各阶段详细指标
    #         df_phases.to_excel(writer, sheet_name='Model Results', startrow=df_overall.shape[0] + 2)

    #     print(f"数据整理完毕，已成功写入Excel文件: {output_filename}")

    class_names = ['0','1','2','3','4','5']

    for i in range(len(y_true)):
            plot_labels_over_time(
            true_label=y_true[i], 
            pred_label=y_pred[i],
            sample_id=test_ids[i],
            filename=f"predict_{test_ids[i]}.png",
            save_dir='prediction_plots_pre'
            )
            plot_labels_over_time(
            true_label=y_true[i], 
            pred_label=y_raw[i],
            sample_id=test_ids[i],
            filename=f"predict_{test_ids[i]}.png",
            save_dir='prediction_plots_raw'
            )
            plot_confusion_matrix(
                y_true[i], y_pred[i],
                class_names=class_names, 
                sample_id = test_ids[i],
                filename=f'confusion_matrix_{test_ids[i]}.png',
                save_dir='confusion_matrices_pre'
            )
    # all_true_flat = np.concatenate([np.array(x).flatten() for x in y_true if x is not None])
    # all_pred_flat = np.concatenate([np.array(x).flatten() for x in y_pred if x is not None])

    # # 确保长度一致
    # min_len = min(len(all_true_flat), len(all_pred_flat))
    # all_true_flat = all_true_flat[:min_len]
    # all_pred_flat = all_pred_flat[:min_len]

    # plot_confusion_matrix(
    #     all_true_flat, all_pred_flat,  # 使用展平的数组
    #     class_names=class_names, 
    #     sample_id='test',
    #     filename=f'confusion_matrix_test_dark.png',
    #     save_dir='confusion_matrices_pre'
    # )
if __name__ == '__main__':
    main()

