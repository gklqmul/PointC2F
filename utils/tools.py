import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import seaborn as sns





import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch


def extract_embeddings(encoder, dataloader, device="cuda"):
    """
    提取每个样本的编码特征，包括点云特征。
    
    encoder: 可接受 (x, points) 并输出样本级特征的模型
    dataloader: 返回 batch 的 dict, 包含 'input', 'points', 'label'
    device: 使用的设备
    """
    encoder.eval()
    embeddings, labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['input'].to(device)        # 1D特征
            z = batch['points'].to(device)       # 点云特征 [B, N, 3] 或 [B, N, D]
            y = batch['label'].cpu().numpy()     # 标签
            
            # 编码器返回样本级嵌入
            feats = encoder(z)                # [B, embedding_dim]
            
            embeddings.append(feats.cpu().numpy())
            labels.append(y)
    
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    
    return embeddings, labels
def evaluate_embeddings(embeddings, labels, k=5):
    """Compute KNN-F1, CH, DB indices."""
    # --- KNN F1 ---
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeddings, labels)
    preds = knn.predict(embeddings)
    knn_f1 = f1_score(labels, preds, average="macro")

    # --- clustering scores ---
    ch_score = calinski_harabasz_score(embeddings, labels)
    db_score = davies_bouldin_score(embeddings, labels)

    return knn_f1, ch_score, db_score

def compare_encoders(encoder1, encoder2, dataloader, device="cuda", save_path="tsn.png"):
    """
    Compare two encoders by t-SNE and clustering metrics.
    Args:
        encoder1, encoder2: two feature extractors (torch.nn.Module)
        dataloader: torch DataLoader (must be same dataset)
        device: str, "cuda" or "cpu"
        save_path: optional, save figure path
    """
    # --- Extract embeddings ---
    emb1, labels = extract_embeddings(encoder1, dataloader, device)
    emb2, _ = extract_embeddings(encoder2, dataloader, device)

    # --- T-SNE visualization ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb1_tsne = tsne.fit_transform(emb1)
    emb2_tsne = tsne.fit_transform(emb2)

    # 定义六个颜色
    colors = [
        "#1f77b4",  # 蓝 - 稳重，常用主色
        "#ff7f0e",  # 橙 - 鲜明但不刺眼
        "#2ca02c",  # 绿 - 柔和自然
        "#d62728",  # 红 - 深红，少许庄重
        "#9467bd",  # 紫 - 高级灰紫
        "#8c564b",  # 棕 - 暖色调
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Encoder1
    for i in range(6):
        idx = labels == i
        axes[0].scatter(emb1_tsne[idx,0], emb1_tsne[idx,1], 
                        c=colors[i], s=10, label=f"Class {i}")
    axes[0].set_title("Encoder 1 (T-SNE)")
    axes[0].legend(markerscale=2, fontsize=8)

    # Encoder2
    for i in range(6):
        idx = labels == i
        axes[1].scatter(emb2_tsne[idx,0], emb2_tsne[idx,1], 
                        c=colors[i], s=10, label=f"Class {i}")
    axes[1].set_title("Encoder 2 (T-SNE)")
    axes[1].legend(markerscale=2, fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    # --- Metrics ---
    knn_f1_1, ch1, db1 = evaluate_embeddings(emb1, labels)
    knn_f1_2, ch2, db2 = evaluate_embeddings(emb2, labels)

    results = {
        "encoder1": {"KNN_F1": knn_f1_1, "CH": ch1, "DB": db1},
        "encoder2": {"KNN_F1": knn_f1_2, "CH": ch2, "DB": db2}
    }
    print("All results:", results)

    return results


def evaluate_and_process(model, dataloader, device, post_process=True, correction_fn=None):
    """
    Evaluate model performance and optionally apply post-processing.
    Returns:
        dict: {
            'raw_metrics': {...},
            'processed_metrics': {...},
            'true_labels': ndarray,
            'raw_predictions': ndarray,
            'processed_predictions': ndarray (optional)
        }
    """
    model.eval()
    all_preds, all_labels = [], []
    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['input'].to(device)
            z = batch['points'].to(device)
            y = batch['label'].to(device)
            padding_mask = batch['padding_mask'].to(device)

            out = model(x, z, src_key_padding_mask=padding_mask)  # [B, T, num_classes]
            min_T = min(out.shape[1], y.shape[1])
            out = out[:, :min_T, :]        # [B, min_T, num_classes]
            y = y[:, :min_T]               # [B, min_T]
            valid_mask = ~padding_mask[:, :min_T]  # [B, min_T]

            preds = torch.argmax(out, dim=2)      # [B, min_T]
            correct_preds = (preds == y) & valid_mask

            # 收集有效帧的预测、标签和 logits
            for b in range(out.shape[0]):
                all_preds.append(preds[b][valid_mask[b]].cpu().numpy().tolist())
                all_labels.append(y[b][valid_mask[b]].cpu().numpy().tolist())
                all_logits.append(out[b][valid_mask[b]].cpu().numpy().tolist())  # logits: [num_classes] per frame

    y_true = np.array(all_labels)
    y_raw = np.array(all_preds)
    logits_arr = np.array(all_logits)  # shape: [N, num_classes]

    raw_metrics = {
        "Precision": precision_score(y_true, y_raw, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_raw, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_raw, average="macro", zero_division=0),
        "Accuracy": accuracy_score(y_true, y_raw)
    }

    results = {
        "raw_metrics": raw_metrics,
        "true_labels": y_true,
        "raw_predictions": y_raw
    }

    # Post-processed metrics
    if post_process and correction_fn is not None:
        y_proc = correction_fn(logits_arr)  # 用 logits 而不是类别
        processed_metrics = {
            "Precision": precision_score(y_true, y_proc, average="macro", zero_division=0),
            "Recall": recall_score(y_true, y_proc, average="macro", zero_division=0),
            "F1": f1_score(y_true, y_proc, average="macro", zero_division=0),
            "Accuracy": accuracy_score(y_true, y_proc)
        }
        results["processed_metrics"] = processed_metrics
        results["processed_predictions"] = y_proc

    return results

def get_model_predictions(model, dataloader, device, post_process=True, correction_fn=None):
    """
    更新后的函数，具备双重功能：
    1. 计算并返回逐帧的总体性能指标 (raw_metrics, processed_metrics)。
    2. 返回用于后续阶段性分析的标签和预测的【序列列表】。
    """
    model.eval()
    all_true_seqs, all_raw_pred_seqs, all_logits_seqs = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['input'].to(device)
            z = batch['points'].to(device)
            y = batch['label'] # 保留在CPU
            padding_mask = batch['padding_mask']
            
            out = model(x, z, src_key_padding_mask=padding_mask.to(device))
            # out = model(z).to(device)
            preds = torch.argmax(out, dim=2)
            valid_mask = ~padding_mask

            # 逐个样本提取有效序列，并存入列表
            for i in range(y.shape[0]):
                valid_len = valid_mask[i].sum().item()
                all_true_seqs.append(y[i, :valid_len].numpy())
                all_raw_pred_seqs.append(preds[i, :valid_len].cpu().numpy())
                all_logits_seqs.append(out[i, :valid_len, :].cpu().numpy())
    
    # --- ✅ 新增：计算逐帧指标 ---
    # 1. 将序列列表“压平”成一维数组以计算总体指标
    y_true_flat = np.concatenate(all_true_seqs)
    y_raw_flat = np.concatenate(all_raw_pred_seqs)

    # 2. 计算原始模型的性能指标
    raw_metrics = {
        "Precision": precision_score(y_true_flat, y_raw_flat, average="macro", zero_division=0),
        "Recall": recall_score(y_true_flat, y_raw_flat, average="macro", zero_division=0),
        "F1": f1_score(y_true_flat, y_raw_flat, average="macro", zero_division=0),
        "Accuracy": accuracy_score(y_true_flat, y_raw_flat)
    }

    # 3. 初始化最终返回的字典
    results = {
        "raw_metrics": raw_metrics,
        "true_labels": all_true_seqs, # 返回序列列表
        "raw_predictions": all_raw_pred_seqs # 返回序列列表
    }

    # 4. 如果需要，进行后处理并计算相应指标
    if post_process and correction_fn:
        # 假设 correction_fn 可以处理一个logits序列列表
        all_proc_pred_seqs = [correction_fn(seq) for seq in all_logits_seqs]
        y_proc_flat = np.concatenate(all_proc_pred_seqs)
        
        processed_metrics = {
            "Precision": precision_score(y_true_flat, y_proc_flat, average="macro", zero_division=0),
            "Recall": recall_score(y_true_flat, y_proc_flat, average="macro", zero_division=0),
            "F1": f1_score(y_true_flat, y_proc_flat, average="macro", zero_division=0),
            "Accuracy": accuracy_score(y_true_flat, y_proc_flat)
        }
        results["processed_metrics"] = processed_metrics
        results["processed_predictions"] = all_proc_pred_seqs # 返回序列列表
    
    return results

def compute_speeds(y_true, y_proc_pred, test_ids):
    """
    根据序列ID，计算并记录每个样本的真实和预测速度（m/s）。
    
    参数:
    y_true (list of np.array): 真实标签序列列表。
    y_proc_pred (list of np.array): 后处理预测标签序列列表。
    test_ids (list of str): 序列ID列表，例如 '01-01'。
    
    返回:
    pd.DataFrame: 包含每个样本真实速度和预测速度的 DataFrame。
    """

    # 常量
    TOTAL_DISTANCE = 5.5   # 米
    FRAME_RATE = 18      # 帧/秒
    
    # 速度类别映射
    speed_map = {}
    speed_map.update({f'0{i}': 'normal' for i in [1, 4]})
    speed_map.update({f'0{i}': 'fast' for i in [2, 5]})
    speed_map.update({f'0{i}': 'slow' for i in [3, 6]})
    
    records = []
    
    for i, test_id in enumerate(test_ids):
        try:
            speed_level = test_id.split('-')[1]
            speed_type = speed_map.get(speed_level)
            if not speed_type:
                print(f"⚠️ 警告: 未知速度级别 '{speed_level}' (ID: {test_id})，跳过。")
                continue
        except IndexError:
            print(f"⚠️ 警告: ID 格式错误 '{test_id}'，跳过。")
            continue
        
        
        true_seq = y_true[i]
        proc_seq = y_proc_pred[i]
        
        true_frames = np.sum(np.isin(true_seq, [2, 4]))
        proc_frames = np.sum(np.isin(proc_seq, [2, 4]))
        
        true_time = true_frames / FRAME_RATE if true_frames > 0 else 0
        proc_time = proc_frames / FRAME_RATE if proc_frames > 0 else 0
        
        true_speed = TOTAL_DISTANCE / true_time if true_time > 0 else 0
        pred_speed = TOTAL_DISTANCE / proc_time if proc_time > 0 else 0
        
        records.append({
            "Test_ID": test_id,
            "Speed_Type": speed_type,
            "True": true_speed,
            "Pred": pred_speed
        })
    
    df = pd.DataFrame(records)
    
    # 保存 Excel
    # file_path = "all_sample_speeds.xlsx"
    # df.to_excel(file_path, index=False)
    # print(f"✅ 所有样本的速度已保存到 {file_path}")
    
    return df
def compute_speeds_with_bias_correction(y_true, y_proc_pred, test_ids):
    """
    根据序列ID，计算并记录每个样本的真实和预测速度（m/s），
    并使用平移校正对预测速度进行校正。
    
    参数:
    y_true (list of np.array): 真实标签序列列表。
    y_proc_pred (list of np.array): 后处理预测标签序列列表。
    test_ids (list of str): 序列ID列表，例如 '01-01'。
    
    返回:
    pd.DataFrame: 包含每个样本真实速度、原始预测速度和校正后预测速度的 DataFrame。
    """

    # 常量
    TOTAL_DISTANCE = 5.5   # 米
    FRAME_RATE = 18      # 帧/秒
    
    # 速度类别映射
    speed_map = {}
    speed_map.update({f'0{i}': 'normal' for i in [1, 4]})
    speed_map.update({f'0{i}': 'fast' for i in [2, 5]})
    speed_map.update({f'0{i}': 'slow' for i in [3, 6]})
    
    records = []
    
    for i, test_id in enumerate(test_ids):
        try:
            speed_level = test_id.split('-')[1]
            speed_type = speed_map.get(speed_level)
            if not speed_type:
                print(f"⚠️ 警告: 未知速度级别 '{speed_level}' (ID: {test_id})，跳过。")
                continue
        except IndexError:
            print(f"⚠️ 警告: ID 格式错误 '{test_id}'，跳过。")
            continue
        
        # 当前序列
        true_seq = y_true[i]
        proc_seq = y_proc_pred[i]
        
        # 取出“行走帧”的数量 (类别2和4表示行走)
        true_frames = np.sum(np.isin(true_seq, [2, 4]))
        proc_frames = np.sum(np.isin(proc_seq, [2, 4]))
        
        # 转换为时间
        true_time = true_frames / FRAME_RATE if true_frames > 0 else 0
        proc_time = proc_frames / FRAME_RATE if proc_frames > 0 else 0
        
        # 转换为速度 (m/s)
        true_speed = TOTAL_DISTANCE / true_time if true_time > 0 else 0
        pred_speed = TOTAL_DISTANCE / proc_time if proc_time > 0 else 0
        
        records.append({
            "Test_ID": test_id,
            "Speed_Type": speed_type,
            "True": true_speed,
            "Pred": pred_speed
        })
    
    df = pd.DataFrame(records)
    
    # --- 添加平移校正 ---
    # 过滤掉速度为0的样本，避免对平均偏差的计算产生影响
    valid_data = df[(df['True'] > 0) & (df['Pred'] > 0)]

    if len(valid_data) > 0:
        # 计算平均偏差
        average_true_speed = valid_data['True'].mean()
        average_pred_speed = valid_data['Pred'].mean()
        bias = average_true_speed - average_pred_speed
        
        print(f"\n✅ 计算得到的平均偏差 (bias): {bias:.4f} m/s\n")
        
        # 对所有预测速度进行平移校正
        df['Corrected_Pred'] = df['Pred'] + bias
    else:
        print("⚠️ 警告: 样本数过少，无法进行平移校正。")
        df['Corrected_Pred'] = df['Pred']

    # --- 校正结束 ---
    
    # 保存 Excel
    file_path = "all_sample_speeds.xlsx"
    df.to_excel(file_path, index=False)
    print(f"✅ 所有样本的速度已保存到 {file_path}")
    
    return df

def plot_gait_speeds_violin(df):
    """
    Plot violin plots of True vs Predicted speed distributions 
    for different Speed_Type categories.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: ['Speed_Type', 'True_Speed_m_s', 'Pred_Speed_m_s']
    save_path : str or None
        If provided, save the figure to this path instead of showing.
    """
    # Reshape dataframe for easier plotting
    df_long = pd.melt(
        df,
        id_vars=["Test_ID", "Speed_Type"],
        value_vars=["True", "Corrected_Pred"],
        var_name="Source",
        value_name="Speed_m_s"
    )
    source_mapping = {
        "Corrected_Pred": "Pred"
    }

    # 使用 replace() 方法替换 'Source' 列的值
    df_long['Source'].replace(source_mapping, inplace=True)
    plt.figure(figsize=(8, 6))
    custom_palette = {
        'True': "#F09D6D",
        'Pred': "#6B9FB8" 
    }
    sns.violinplot(
        x='Speed_Type',
        y='Speed_m_s',
        hue='Source',
        data=df_long,
        split=True,
        inner='quart',
        palette=custom_palette,
    )

    plt.title("True vs Predicted Speed Distributions by Speed Type")
    plt.xlabel("Speed Type")
    plt.ylabel("Speed (m/s)")
    plt.legend(title="Source")
    plt.tight_layout()
    plt.savefig('speed_distribution_violin_plot.png', dpi=300)
   
def compute_icc(y_true, y_pred):
    """
    Compute ICC(2,1) for absolute agreement.
    This function is a direct replacement for pingouin's ICC calculation.
    """
    # Combine true and predicted values into a single array
    y = np.vstack([y_true, y_pred]).T
    n, k = y.shape
    mean_per_target = np.mean(y, axis=1)
    mean_per_rater = np.mean(y, axis=0)
    grand_mean = np.mean(y)

    # Sums of squares
    SSR = np.sum((mean_per_rater - grand_mean) ** 2) * n
    SSC = np.sum((mean_per_target - grand_mean) ** 2) * k
    SSE = np.sum((y - mean_per_target[:, None] - mean_per_rater + grand_mean) ** 2)

    MSC = SSC / (n - 1)
    MSR = SSR / (k - 1)
    MSE = SSE / ((k - 1) * (n - 1))

    # ICC(2,1) formula
    icc = (MSC - MSE) / (MSC + (k - 1) * MSE + k * (MSR - MSE) / n)
    return icc


def compute_speed_metrics(df):
    """
    根据包含真实和预测速度的长表格 DataFrame (Test_ID, Speed_Type, True_Speed_m_s, Pred_Speed_m_s)，
    分别计算 normal, fast, slow 三种速度类型的多种度量指标。

    参数:
    df (pd.DataFrame): 输入数据
    """
    
    metrics = []
    speed_types = ['slow', 'normal', 'fast']
    
    # 先计算总体指标
    true_all = df["True"].values
    pred_all = df["Corrected_Pred"].values
    
    if len(true_all) >= 2:
        mae_all = np.mean(np.abs(pred_all - true_all))
        mape_all = np.mean(np.abs((pred_all - true_all) / (true_all + 1e-6))) * 100
        r_all = np.corrcoef(true_all, pred_all)[0, 1]
        icc_all = compute_icc(true_all, pred_all)
        sem_all = np.std(pred_all - true_all, ddof=1) / np.sqrt(2)
        mdc_all = 1.96 * sem_all * np.sqrt(2)
        
        metrics.append({
            'Speed_Type': 'overall',
            'MAE(m/s)': mae_all,
            'MAPE(%)': mape_all,
            'r': r_all,
            'ICC': icc_all,
            'SEM': sem_all,
            'MDC': mdc_all,
            'Sample_Size': len(df)
        })
    
    # 再分别计算每种速度类型的指标
    for speed_type in speed_types:
        group = df[df["Speed_Type"] == speed_type]
        true_speeds = group["True"].values
        pred_speeds = group["Corrected_Pred"].values

        if len(true_speeds) < 2:
            metrics.append({
                'Speed_Type': speed_type,
                'MAE(m/s)': np.nan, 
                'MAPE(%)': np.nan, 
                'r': np.nan,
                'ICC': np.nan, 
                'SEM': np.nan, 
                'MDC': np.nan,
                'Sample_Size': len(group)
            })
            continue

        # 1. MAE
        mae = np.mean(np.abs(pred_speeds - true_speeds))

        # 2. MAPE
        mape = np.mean(np.abs((pred_speeds - true_speeds) / (true_speeds + 1e-6))) * 100

        # 3. 皮尔逊相关系数
        r = np.corrcoef(true_speeds, pred_speeds)[0, 1]

        # 4. ICC
        icc = compute_icc(true_speeds, pred_speeds)

        # 5. SEM
        sem = np.std(pred_speeds - true_speeds, ddof=1) / np.sqrt(2)

        # 6. MDC
        mdc = 1.96 * sem * np.sqrt(2)

        metrics.append({
            'Speed_Type': speed_type,
            'MAE(m/s)': mae,
            'MAPE(%)': mape,
            'r': r,
            'ICC': icc,
            'SEM': sem,
            'MDC': mdc,
            'Sample_Size': len(group)
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # 打印详细结果
    print("=" * 80)
    print("速度指标计算结果 (按速度类型分组)")
    print("=" * 80)
    
    for _, row in metrics_df.iterrows():
        print(f"\n📊 {row['Speed_Type'].upper()} (n={row['Sample_Size']}):")
        print(f"   MAE: {row['MAE(m/s)']:.4f} m/s")
        print(f"   MAPE: {row['MAPE(%)']:.2f}%")
        print(f"   相关系数 r: {row['r']:.4f}")
        print(f"   ICC: {row['ICC']:.4f}")
        print(f"   SEM: {row['SEM']:.4f} m/s")
        print(f"   MDC: {row['MDC']:.4f} m/s")
    
    print("=" * 80)
    
    return metrics_df


def get_phase_data(label_seq, num_classes):
    data = {}
    label_seq = np.atleast_1d(label_seq)
    for phase_id in range(1, num_classes + 1):
        indices = np.where(label_seq == phase_id)[0]
        if len(indices) > 0:
            start, end = indices[0], indices[-1]
            data[phase_id] = {
                'boundary': (start, end),
                'duration': len(indices)
            }
    return data


def evaluate_comprehensive_metrics(all_true_labels, all_pred_labels, num_classes=5, fps=18):

    if not all_true_labels or not all_pred_labels:
        return {}
        
    # --- 1. 计算【逐帧指标】(基于混淆矩阵) ---
    y_true_flat = np.concatenate(all_true_labels)
    y_pred_flat = np.concatenate(all_pred_labels)
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(num_classes + 1))
    
    frame_wise_metrics = {}
    for phase_id in range(1, num_classes + 1):
        tp = cm[phase_id, phase_id]
        fn = np.sum(cm[phase_id, :]) - tp
        fp = np.sum(cm[:, phase_id]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        frame_wise_metrics[phase_id] = {
            "Accuracy": recall, # 阶段内部准确率就是召回率
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }

    # --- 2. 收集【时长和边界】数据 ---
    collected_data = defaultdict(lambda: {
        'true_durations': [], 'pred_durations': [],
        'start_errors': [], 'end_errors': [], 'ious': []
    })
    for true_seq, pred_seq in zip(all_true_labels, all_pred_labels):
        true_data = get_phase_data(true_seq, num_classes)
        pred_data = get_phase_data(pred_seq, num_classes)

        # 收集每个子阶段的数据
        for phase_id in range(1, num_classes + 1):
            true_phase = true_data.get(phase_id)
            pred_phase = pred_data.get(phase_id)
            # 只有当真实和预测都存在时，才计算误差
            if true_phase and pred_phase:
                collected_data[phase_id]['true_durations'].append(true_phase['duration'])
                collected_data[phase_id]['pred_durations'].append(pred_phase['duration'])
                t_start, t_end = true_phase['boundary']; p_start, p_end = pred_phase['boundary']
                collected_data[phase_id]['start_errors'].append(abs(p_start - t_start))
                collected_data[phase_id]['end_errors'].append(abs(p_end - t_end))
                intersection = max(0, min(t_end, p_end) - max(t_start, p_start) + 1)
                union = max(t_end, p_end) - min(t_start, p_start) + 1
                collected_data[phase_id]['ious'].append(intersection / union if union > 0 else 0)

        # 收集整个动作序列的数据
        true_s, true_e = true_data.get(1), true_data.get(num_classes)
        pred_s, pred_e = pred_data.get(1), pred_data.get(num_classes)
        if true_s and true_e and pred_s and pred_e:
            collected_data['Overall_Action']['true_durations'].append(true_e['boundary'][1] - true_s['boundary'][0] + 1)
            collected_data['Overall_Action']['pred_durations'].append(pred_e['boundary'][1] - pred_s['boundary'][0] + 1)
            collected_data['Overall_Action']['start_errors'].append(abs(pred_s['boundary'][0] - true_s['boundary'][0]))
            collected_data['Overall_Action']['end_errors'].append(abs(pred_e['boundary'][1] - true_e['boundary'][1]))
            intersection = max(0, min(true_e['boundary'][1], pred_e['boundary'][1]) - max(true_s['boundary'][0], pred_s['boundary'][0]) + 1)
            union = max(true_e['boundary'][1], pred_e['boundary'][1]) - min(true_s['boundary'][0], pred_s['boundary'][0]) + 1
            collected_data['Overall_Action']['ious'].append(intersection / union if union > 0 else 0)

    # --- 3. 汇总所有指标 ---
    summary = {}
    phase_keys = list(range(1, num_classes + 1)) + ['Overall_Action']
    for key in phase_keys:
        data = collected_data.get(key)
        # 只有在有数据可以比较时才生成报告
        if not data or not data['true_durations']: continue
        
        t, p = np.array(data['true_durations']), np.array(data['pred_durations'])
        
        # Duration Metrics
        rmse = np.sqrt(mean_squared_error(t, p)) / fps
        mae = mean_absolute_error(t, p) / fps
        non_zero_mask = t != 0
        mape = np.mean(np.abs((t[non_zero_mask] - p[non_zero_mask]) / t[non_zero_mask])) * 100 if np.any(non_zero_mask) else 0
        r, icc = np.nan, compute_icc(t, p)
        if len(t) > 1 and t.std() > 0 and p.std() > 0: r, _ = pearsonr(t, p)

        # Boundary Metrics
        s_mae = np.mean(data['start_errors']) / fps if data['start_errors'] else 0
        e_mae = np.mean(data['end_errors']) / fps if data['end_errors'] else 0
        iou = np.mean(data['ious']) if data['ious'] else 0
        
        summary_key = f'Phase_{key}' if isinstance(key, int) else key
        summary[summary_key] = {
            "RMSE(s)": rmse, "MAE(s)": mae, "MAPE(%)": mape, "r": r, "ICC": icc,
            "S_MAE(s)": s_mae, "E_MAE(s)": e_mae, "IoU": iou
        }
        
        # 合并预先计算好的逐帧指标
        if isinstance(key, int) and key in frame_wise_metrics:
            summary[summary_key].update(frame_wise_metrics[key])
            
    return summary

def plot_confusion_matrix(y_true, y_pred, class_names, sample_id, filename, save_dir):
    """
    Plot and save a confusion matrix.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        class_names (list): List of class names.
        sample_id (str): Identifier for the sample (e.g., 'test').
        filename (str): Filename for saving the plot.
        save_dir (str): Directory to save the plot.
        normalize (bool): Whether to normalize the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)


    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", 
                cmap="Blues", xticklabels=class_names, yticklabels=class_names)

    plt.title(f'Confusion Matrix ({sample_id})')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()


# def plot_labels_over_time(true_label, pred_label, sample_id, filename, save_dir="prediction_plots"):
#     """
#     Plot the change of true and predicted labels over time (discrete categories).
#     Use step plots for clarity.
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     frames = range(len(true_label))

#     plt.figure(figsize=(12, 4))
#     plt.step(frames, true_label, where="post", label="True Label", linewidth=2, color="black")
#     plt.step(frames, pred_label, where="post", label="Predicted Label", linewidth=2, color="red", linestyle="--")

#     plt.xlabel("Frame", fontsize=16)
#     plt.ylabel("Label", fontsize=16)
#     plt.title(f"Sample {sample_id} - True vs Predicted Labels", fontsize=16)
#     plt.legend(loc="upper left")
#     plt.grid(alpha=0.3)
#     plt.tight_layout()

#     save_path = os.path.join(save_dir, filename)
#     plt.savefig(save_path, dpi=300)
#     plt.close()

def plot_labels_over_time(true_label, pred_label, sample_id, filename, save_dir="prediction_plots"):
    """
    Plot the change of true and predicted labels over time (discrete categories).
    Use step plots for clarity.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 定义标签映射
    label_mapping = {
        0: "Background",
        1: "Stand Up", 
        2: "Walk Forward",
        3: "Turn",
        4: "Walk Back",
        5: "Sit Down"
    }

    frames = range(len(true_label))

    # 设置字体大小
    plt.rcParams.update({'font.size': 12})
    
    plt.figure(figsize=(14, 6))
    plt.step(frames, true_label, where="post", label="True Label", linewidth=3, color="black")
    plt.step(frames, pred_label, where="post", label="Predicted Label", linewidth=3, color="red", linestyle="--")

    # 设置y轴刻度和标签（倾斜45度）
    unique_labels = sorted(set(true_label) | set(pred_label))
    y_ticks = list(unique_labels)
    y_tick_labels = [label_mapping.get(label, f"Unknown({label})") for label in y_ticks]
    
    plt.yticks(y_ticks, y_tick_labels, rotation=45)  # 添加 rotation=45
    
    plt.xlabel("Frame", fontsize=14)
    plt.ylabel("Action Phase", fontsize=14)
    plt.title(f"Sample {sample_id} - True vs Predicted Labels", fontsize=14, fontweight='bold')
    plt.legend(loc="upper left", fontsize=14, frameon=True, edgecolor='black', framealpha=1)
    plt.grid(alpha=0.3)
    
    # 调整布局，为倾斜的标签留出更多空间
    plt.tight_layout()

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()