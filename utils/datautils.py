import numpy as np
import torch
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm


def pad_collate_fn(batch):
    """
    batch: List[dict]，每个 dict 包含 'input', 'points', 'label', 'padding_mask' 等
    自动裁剪/补齐到 max_len 帧，pad 到统一 shape
    """
    max_len = 400
    out_batch = {}
    for key in batch[0].keys():
        items = [item[key] for item in batch]
        # 只对需要 pad 的字段处理
        if isinstance(items[0], torch.Tensor) and items[0].ndim >= 2:
            # shape: [T, ...]
            padded = []
            for x in items:
                # 先裁剪
                x = x[:max_len]
                pad_shape = [max_len - x.shape[0]] + list(x.shape[1:])
                if pad_shape[0] > 0:
                    pad_tensor = torch.zeros(pad_shape, dtype=x.dtype)
                    x = torch.cat([x, pad_tensor], dim=0)
                padded.append(x)
            out_batch[key] = torch.stack(padded)
        elif isinstance(items[0], torch.Tensor) and items[0].ndim == 1:
            # label/padding_mask: [T]
            padded = []
            for x in items:
                x = x[:max_len]
                pad_shape = [max_len - x.shape[0]]
                if pad_shape[0] > 0:
                    pad_tensor = torch.zeros(pad_shape, dtype=x.dtype)
                    x = torch.cat([x, pad_tensor], dim=0)
                padded.append(x)
            out_batch[key] = torch.stack(padded)
        else:
            # 其它类型直接打包
            out_batch[key] = items

    # 自动补齐 padding_mask 字段
    if 'padding_mask' not in out_batch:
        masks = []
        for item in batch:
            T = min(item['input'].shape[0], max_len)
            mask = torch.zeros(max_len, dtype=torch.bool)
            if T < max_len:
                mask[T:] = True  # 后面为填充
            masks.append(mask)
        out_batch['padding_mask'] = torch.stack(masks)

    return out_batch


import torch
import numpy as np

# 这是一个辅助函数，用来处理单个数据序列的补齐/截断和mask生成
def pad_trim_and_create_mask(arr, fixed_len):
    # 这个函数现在只需要处理特征，不再需要返回mask
    cur_len = arr.shape[0]
    if cur_len >= fixed_len:
        return arr[:fixed_len, :]
    else:
        pad_len = fixed_len - cur_len
        pad = np.zeros((pad_len, arr.shape[1]), dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

def my_collate_fn(batch, fixed_len=450):
    """
    为手动屏蔽损失计算而设计的 Collate Function。
    它确保 input, points, label, 和 padding_mask 都具有相同的、固定的序列长度。
    标签填充值使用 -1，更加安全。
    
    Args:
        batch (list): Dataset返回的样本字典的列表。
        fixed_len (int): 所有序列都将被处理到的目标长度。
    """
    processed_inputs = []
    processed_points = []
    processed_labels = []
    masks = []

    # 遍历批次中的每一个样本
    for sample_dict in batch:
        # 假设Dataset返回的是NumPy数组或可以转换为数组的格式
        input_arr = np.array(sample_dict['input'], dtype=np.float32)
        points_arr = np.array(sample_dict['points'], dtype=np.float32)
        label_arr = np.array(sample_dict['label'], dtype=np.int64)

        # 获取处理前的真实长度，这是所有操作的基准
        original_len = input_arr.shape[0]

        # --- 对所有序列进行统一的填充或截断 ---

        # 1. 处理 Input 特征
        if original_len >= fixed_len:
            processed_inputs.append(input_arr[:fixed_len, :])
        else:
            pad_len = fixed_len - original_len
            pad = np.zeros((pad_len, input_arr.shape[1]), dtype=np.float32)
            processed_inputs.append(np.concatenate([input_arr, pad], axis=0))

        # 2. 处理 Points 特征
        if original_len >= fixed_len:
            processed_points.append(points_arr[:fixed_len, :, :])
        else:
            pad_len = fixed_len - original_len
            pad = np.zeros((pad_len, points_arr.shape[1], points_arr.shape[2]), dtype=np.float32)
            processed_points.append(np.concatenate([points_arr, pad], axis=0))

        # 3. 处理 Label 序列 (同样需要填充)
        if original_len >= fixed_len:
            processed_labels.append(label_arr[:fixed_len])
        else:
            pad_len = fixed_len - original_len
            # --- 主要修改在这里 ---
            # 使用 -1 来填充，避免与类别 0 混淆
            pad = np.full(pad_len, -1, dtype=np.int64) 
            processed_labels.append(np.concatenate([label_arr, pad], axis=0))

        # 4. 根据真实长度生成正确的 Padding Mask
        mask = np.zeros(fixed_len, dtype=bool)
        if original_len < fixed_len:
            mask[original_len:] = True  # 填充区域为 True
        masks.append(mask)

    # --- 将所有处理好的列表堆叠成最终的批次张量 ---
    return {
        'input': torch.from_numpy(np.stack(processed_inputs, axis=0)),
        'points': torch.from_numpy(np.stack(processed_points, axis=0)),
        'label': torch.from_numpy(np.stack(processed_labels, axis=0)),
        'padding_mask': torch.from_numpy(np.stack(masks, axis=0))
    }

def channel_wise_noise_single(seq, noise_scale=0.05):
    """
    Add Gaussian noise independently to each feature channel.
    seq: (seq_len, feature_dim)
    """
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=seq.shape)
    return seq + noise

def magnitude_scaling(seq, scale_range=(0.9, 1.1)):
    """
    Scale the magnitude of the whole sequence by a random factor.
    seq: (seq_len, feature_dim)
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return seq * scale

def feature_dropout(seq, dropout_ratio=0.1):
    """
    Randomly drop (set to zero) some feature dimensions across the whole sequence.
    seq: (seq_len, feature_dim)
    """
    feature_dim = seq.shape[1]
    num_drop = int(feature_dim * dropout_ratio)
    drop_idx = np.random.choice(feature_dim, num_drop, replace=False)
    seq_copy = seq.copy()
    seq_copy[:, drop_idx] = 0
    return seq_copy

def time_masking(seq, max_mask_ratio=0.15):
    """
    Mask out a continuous time segment in the sequence.
    seq: (seq_len, feature_dim)
    """
    seq_len = seq.shape[0]
    max_mask_len = int(seq_len * max_mask_ratio)
    if max_mask_len < 1:
        return seq

    mask_len = np.random.randint(1, max_mask_len + 1)
    start = np.random.randint(0, seq_len - mask_len + 1)

    seq_copy = seq.copy()
    seq_copy[start:start + mask_len, :] = 0
    return seq_copy


def smooth_signal(x, window_size=5):
    """Simple moving average smoothing."""
    if window_size < 2:
        return x
    window = np.ones(window_size) / window_size
    return np.convolve(x, window, mode='same')

def compute_derivatives(x):
    """Compute first and second order derivatives of a 1D array."""
    first = np.gradient(x)
    second = np.gradient(first)
    return first, second

def enhance_features(raw_seq):
    """
    Enhance raw features by adding motion/distance derivatives, 
    active mask, and height/distance ratio.
    
    Args:
        raw_seq (ndarray): shape (seq_len, feature_dim)
                           first 3 features = [motion, distance, height]
    Returns:
        enhanced_seq (ndarray): (seq_len, feature_dim + extra_features)
    """
    seq_len, feat_dim = raw_seq.shape
    
    motion = raw_seq[:, 0]
    distance = raw_seq[:, 1]
    height = raw_seq[:, 2]

    # motion derivatives
    motion_d1, motion_d2 = compute_derivatives(motion)

    # distance derivatives
    dist_d1, dist_d2 = compute_derivatives(distance)

    # smooth distance
    smoothed_dist = smooth_signal(distance, window_size=5)

    # active mask
    active_mask = (smoothed_dist < 3.8).astype(float)

    # height/distance ratio (use smoothed distance to avoid spikes)
    height_dist_ratio = height / np.clip(smoothed_dist, a_min=1e-6, a_max=None)

    # stack all together
    extra_features = np.stack([
        motion_d1, motion_d2,
        dist_d1, dist_d2,
        active_mask,
        height_dist_ratio
    ], axis=1)

    enhanced_seq = np.hstack([raw_seq, extra_features])
    return enhanced_seq


def _load_raw_features_for_scaler(root_dir, sample_id):
    """
    一个独立的辅助函数，用于加载单个样本的所有原始特征。
    此逻辑与您Dataset中的版本完全相同。
    """
    motion = np.loadtxt(os.path.join(root_dir, f"{sample_id}-motion.csv"), delimiter=',')
    distance = np.loadtxt(os.path.join(root_dir, f"{sample_id}-distance.csv"), delimiter=',')
    height = np.loadtxt(os.path.join(root_dir, f"{sample_id}-height.csv"), delimiter=',')
    maxD = np.loadtxt(os.path.join(root_dir, f"{sample_id}-maxD.csv"), delimiter=',')
    meanD = np.loadtxt(os.path.join(root_dir, f"{sample_id}-meanD.csv"), delimiter=',')
    medD = np.loadtxt(os.path.join(root_dir, f"{sample_id}-medD.csv"), delimiter=',')
    points = np.loadtxt(os.path.join(root_dir, f"{sample_id}-points.csv"), delimiter=',')

    # 保证维度正确
    if motion.ndim == 1: motion = motion[:, np.newaxis]
    if distance.ndim == 1: distance = distance[:, np.newaxis]
    if height.ndim == 1: height = height[:, np.newaxis]
    if maxD.ndim == 1: maxD = maxD[:, np.newaxis]
    if meanD.ndim == 1: meanD = meanD[:, np.newaxis]
    if medD.ndim == 1: medD = medD[:, np.newaxis]
    
    # 找到最小长度
    min_length = min(len(motion), len(distance), len(height), len(maxD), len(meanD), len(medD), len(points))
    
    motion = motion[:min_length]
    distance = distance[:min_length]
    height = height[:min_length]
    maxD = maxD[:min_length]
    meanD = meanD[:min_length]
    medD = medD[:min_length]
    points = points[:min_length]
        
    # 先连接所有单列特征
    basic_features = np.concatenate([motion, distance, height, maxD, meanD, medD], axis=1)
    
    # 单独处理 points 特征 - 检查其维度
    if points.ndim == 1:
        points = points[:, np.newaxis]
    
    # 合并所有特征
    return np.hstack([basic_features, points])


def generate_and_save_scaler(root_dir, train_ids, scaler_path='scalernew.gz'):
    """
    遍历所有训练样本，使用与Dataset完全相同的加载逻辑收集数据，
    计算并保存用于标准化的Scaler对象。
    """
    print(f"开始从 {len(train_ids)} 个训练样本中收集数据以生成Scaler...")
    
    all_train_features = []
    # 遍历所有训练样本ID
    for sample_id in tqdm(train_ids, desc="收集中..."):
        raw_seq = _load_raw_features_for_scaler(root_dir, sample_id)
        enhanced_seq = enhance_features(raw_seq)
        all_train_features.append(enhanced_seq)

    if not all_train_features:
        raise ValueError("未能从训练ID中收集到任何特征数据，请检查文件路径和ID。")

    # 将所有增强后的训练帧拼接成一个巨大的数组 [N_total_frames, num_features]
    full_feature_matrix = np.concatenate(all_train_features, axis=0)
    print(f"数据收集完毕，总共 {full_feature_matrix.shape[0]} 帧，{full_feature_matrix.shape[1]} 个特征。")

    # 创建StandardScaler并拟合
    print("正在拟合Scaler...")
    scaler = StandardScaler()
    scaler.fit(full_feature_matrix)

    # 保存拟合好的scaler到文件
    joblib.dump(scaler, scaler_path)
    print(f"Scaler已成功生成并保存到: {scaler_path}")
    print(f"  - 特征数量: {scaler.n_features_in_}")