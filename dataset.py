import torch
from torch.utils.data import Dataset
import os
import json
import numpy as np
from glob import glob
from scipy.io import loadmat
import joblib

from utils.datautils import channel_wise_noise_single, enhance_features, feature_dropout, magnitude_scaling, time_masking

class TUGFeatureDataset(Dataset):
    def __init__(self, root_dir, sample_ids, label_dict, scaler, fixed_len=450, augment=False):
        """
        标准化的Dataset类。

        Args:
            self.root_dir (str): 数据根目录。
            sample_ids (list): 这个数据集实例包含的样本ID列表 (训练/验证/测试)。
            label_dict (dict): 从label.json加载的标签字典。
            scaler (StandardScaler): 预先拟合好的scaler对象。
            fixed_len (int): 序列的固定长度。
            augment (bool): 是否应用数据增强。
        """
        super().__init__()
        self.root_dir = root_dir
        self.sample_ids = sample_ids
        self.label_dict = label_dict
        self.scaler = scaler
        self.fixed_len = fixed_len
        self.augment = augment
        self.num_classes = 0


    def __len__(self):
        return len(self.sample_ids)

    def _load_raw_feature_seq(self, sample_id):
    
        motion = np.loadtxt(os.path.join(self.root_dir, f"{sample_id}-motion.csv"), delimiter=',')
        distance = np.loadtxt(os.path.join(self.root_dir, f"{sample_id}-distance.csv"), delimiter=',')
        height = np.loadtxt(os.path.join(self.root_dir, f"{sample_id}-height.csv"), delimiter=',')
        maxD = np.loadtxt(os.path.join(self.root_dir, f"{sample_id}-maxD.csv"), delimiter=',')
        meanD = np.loadtxt(os.path.join(self.root_dir, f"{sample_id}-meanD.csv"), delimiter=',')
        medD = np.loadtxt(os.path.join(self.root_dir, f"{sample_id}-medD.csv"), delimiter=',')
        points = np.loadtxt(os.path.join(self.root_dir, f"{sample_id}-points.csv"), delimiter=',')

        # 保证维度正确
        if motion.ndim == 1: motion = motion[:, np.newaxis]
        if distance.ndim == 1: distance = distance[:, np.newaxis]
        if height.ndim == 1: height = height[:, np.newaxis]
        if maxD.ndim == 1: maxD = maxD[:, np.newaxis]
        if meanD.ndim == 1: meanD = meanD[:, np.newaxis]
        if medD.ndim == 1: medD = medD[:, np.newaxis]
        # 找到最小长度
        min_length = min(len(motion), len(distance), len(height), len(maxD), len(meanD), len(medD))
        
        motion = motion[:min_length]
        distance = distance[:min_length]
        height = height[:min_length]
        maxD = maxD[:min_length]
        meanD = meanD[:min_length]
        medD = medD[:min_length]
        points = points[:min_length]  # 确保 points 也被截断
           
        # 先连接所有单列特征
        basic_features = np.concatenate([motion, distance, height, maxD, meanD, medD], axis=1)
        
        # 单独处理 points 特征 - 检查其维度
        if points.ndim == 1:
            points = points[:, np.newaxis]
        
        # 合并所有特征
        return np.hstack([basic_features, points])
       

    def _pad_or_trim(self, arr, fixed_len):
        masks = []
        cur_len = arr.shape[0]
        if cur_len >= fixed_len:
            return arr[:fixed_len, :]
        else:
            pad_len = fixed_len - cur_len
            pad = np.zeros((pad_len, arr.shape[1]), dtype=np.float32)
            return np.concatenate([arr, pad], axis=0)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        raw_seq = self._load_raw_feature_seq(sample_id)
        
        if self.augment:
        # 以一定概率应用每种增强
            if np.random.rand() < 0.5: # 50%的概率
                raw_seq = channel_wise_noise_single(raw_seq, noise_scale=0.05)
            
            if np.random.rand() < 0.5: # 50%的概率
                raw_seq = magnitude_scaling(raw_seq, scale_range=(0.9, 1.1))
                
            if np.random.rand() < 0.3: # 30%的概率
                raw_seq = feature_dropout(raw_seq, dropout_ratio=0.1)
                
            if np.random.rand() < 0.5: # 50%的概率
                raw_seq = time_masking(raw_seq, max_mask_ratio=0.15)

        seg_ids = self.label_dict.get(sample_id, [])
        sorted_segs = sorted(seg_ids)
        # raw_seq = raw_seq[sorted_segs[0]:sorted_segs[-1], :] 
        enhanced_seq = enhance_features(raw_seq)
        
        scaled_seq = self.scaler.transform(enhanced_seq)

        point_cloud = self.load_pointcloud_sequence(sample_id)
        # point_cloud = point_cloud[sorted_segs[0]:sorted_segs[-1], :, :]  # 截取对应的点云帧
        pc_mean = point_cloud.mean(axis=1, keepdims=True)  # [frames, 1, 3]
        pc_std = point_cloud.std(axis=1, keepdims=True) + 1e-6  # 防止除0
        normed_point_cloud = (point_cloud - pc_mean) / pc_std  # [frames, 512, 3]
    
        # FOR MULI-classification task
        len_frame = raw_seq.shape[0]
        label = np.zeros(len_frame, dtype=np.int32)
       
        if len(sorted_segs) > 1:
            for i in range(len(sorted_segs) - 1):
                # 4. 定义当前阶段的开始和结束帧
                start_frame = sorted_segs[i]
                end_frame = sorted_segs[i+1] 
                
                phase_label = i+1

                s_idx = np.clip(start_frame, 0, self.fixed_len)
                e_idx = np.clip(end_frame, 0, self.fixed_len)
                
                if s_idx < e_idx:
                    label[s_idx:e_idx] = phase_label

            binary_label = np.zeros(len_frame, dtype=np.int32)
            binary_label[sorted_segs[0]:sorted_segs[-1]] = 1  # 标记动作阶段为1，背景为0

        return {
            'input': torch.tensor(scaled_seq, dtype=torch.float32),
            'points': torch.tensor(normed_point_cloud, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'binary_label': torch.tensor(binary_label, dtype=torch.long),           
        }
    

    def load_pointcloud_sequence(self, sample_id, num_points=160):


        mat_path = os.path.join(self.root_dir, f"{sample_id}-allclouds.mat")
        mat_data = loadmat(mat_path)
        pc_struct = mat_data['pointclouds'].squeeze()  # shape: (num_frames,)

        frames = []
        for i in range(len(pc_struct)):
            x = np.array(pc_struct[i]['X']).reshape(-1)
            y = np.array(pc_struct[i]['Y']).reshape(-1)
            z = np.array(pc_struct[i]['Z']).reshape(-1)
            dopp = np.array(pc_struct[i]['dopp']).reshape(-1)
            # el = np.array(pc_struct[i]['el']).reshape(-1)
            # az = np.array(pc_struct[i]['az']).reshape(-1)
            # rng = np.array(pc_struct[i]['rng']).reshape(-1)
            snr = np.array(pc_struct[i]['snr']).reshape(-1)
            # size = np.array(pc_struct[i]['size']).reshape(-1)

            mask = (x >= -1) & (x <= 1) & (y >= 0) & (y <= 4) & (z >= -1) & (z <= 1)

    
            x, y, z, dopp, snr = x[mask], y[mask], z[mask], dopp[mask], snr[mask]

            points = np.stack([x, y, z, dopp, snr], axis=1)  # shape: (N, 5)
            # points = np.stack([x, y, z], axis=1)  # shape: (N, 3)

            # 截断或补零
            if points.shape[0] >= num_points:
                points = points[:num_points]
            else:
                pad_len = num_points - points.shape[0]
                pad = np.zeros((pad_len, 5))
                points = np.concatenate([points, pad], axis=0)

            frames.append(points)

        return np.stack(frames, axis=0)  # [num_frames, 128, 5]



