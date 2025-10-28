from glob import glob
import json
import joblib
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from models.newpointnet import ImprovedPointNetExtractorNEW
from dataset import TUGFeatureDataset
from utils.datautils import generate_and_save_scaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. 模型架构：PointNet++骨干网络 + 投影头 ---
class PointNetContrastive(nn.Module):
    def __init__(self, point_feature_dim=256, projection_dim=128):
        super().__init__()
        self.backbone = ImprovedPointNetExtractorNEW(output_dim=point_feature_dim)
        self.projection_head = nn.Sequential(nn.Linear(point_feature_dim, projection_dim))

    def forward(self, x):
        feature = self.backbone(x)
        projection = self.projection_head(feature)
        return F.normalize(projection, dim=1)
    def encode(self, x):
        return self.backbone(x)

def augment_point_cloud(points_np):
    """
    points_np: (N, 3) 或 (N, 3+K). 仅对前三维坐标做空间增强。
    返回同形状 (N, D)。
    """
    if not isinstance(points_np, np.ndarray):
        points_np = np.asarray(points_np)

    assert points_np.ndim == 2, f"期望二维数组, 得到形状 {points_np.shape}"
    D = points_np.shape[1]
    if D < 3:
        raise ValueError(f"点维度必须>=3 (含xyz), 得到 {D}")

    coords = points_np[:, :3].astype(np.float32)
    attrs = points_np[:, 3:] if D > 3 else None  # (N, K)

    # 旋转（绕 y 轴示例，可视需求扩展）
    angle = np.random.uniform(-15, 15) * np.pi / 180.0
    cy, sy = np.cos(angle), np.sin(angle)
    rot_y = np.array([[ cy, 0, sy],
                      [  0, 1,  0],
                      [-sy, 0, cy]], dtype=np.float32)
    coords = coords @ rot_y.T

    # 缩放
    scale = np.random.uniform(0.9, 1.1)
    coords *= scale

    # 抖动
    coords += np.random.normal(0, 0.02, coords.shape).astype(np.float32)

    if attrs is not None:
        attrs = attrs.astype(np.float32)

    if attrs is not None:
        out = np.concatenate([coords, attrs], axis=1)
    else:
        out = coords
    return out

# --- 3. SCL损失函数 ---
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, projections, targets):
        device = projections.device
        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        exp_dot_tempered = (torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5)
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        cardinality_per_samples[cardinality_per_samples == 0] = 1e-5
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        return loss_per_sample.mean()

# --- 4. 新的数据集类，整合了数据提取和增强 ---
class SCLPointCloudsDataset(Dataset):

    def __init__(self, source_dataset):
        self.all_pc_frames = []
        self.all_labels = []
        
        print(f"正在从源数据集 {type(source_dataset).__name__} 中提取帧数据...")
        # 直接遍历源数据集，它每次返回一个完整的序列字典
        for i in tqdm(range(len(source_dataset)), desc="处理序列"):
            sample_dict = source_dataset[i]
            
            points_seq = sample_dict['points']
            labels_seq = sample_dict['label']
            
            # 将序列中的每一帧添加到列表中
            for frame_idx in range(len(labels_seq)):
                self.all_pc_frames.append(points_seq[frame_idx].numpy())
                self.all_labels.append(labels_seq[frame_idx].numpy())
                
        print(f"数据准备完毕，总有效帧数: {len(self.all_labels)}")

    def __len__(self):
        return len(self.all_pc_frames)

    def __getitem__(self, idx):
        point_cloud = self.all_pc_frames[idx]
        label = self.all_labels[idx]
        
        # 数据增强
        view_1 = augment_point_cloud(point_cloud)
        view_2 = augment_point_cloud(point_cloud)

        return (torch.tensor(view_1, dtype=torch.float32), 
                torch.tensor(view_2, dtype=torch.float32), 
                torch.tensor(label, dtype=torch.long))


class FramesDatasetForProbe(Dataset):
    """为线性评估准备数据，逻辑与上面类似，但getitem不同。"""
    def __init__(self, source_dataset):
        self.all_pc_frames = []
        self.all_labels = []
        print(f"正在为线性评估从源数据集 {type(source_dataset).__name__} 中提取帧数据...")
        for i in tqdm(range(len(source_dataset)), desc="处理序列"):
            sample_dict = source_dataset[i]
            points_seq = sample_dict['points']
            labels_seq = sample_dict['label']
            for frame_idx in range(len(labels_seq)):
                self.all_pc_frames.append(points_seq[frame_idx].numpy())
                self.all_labels.append(labels_seq[frame_idx].numpy())
        print(f"数据准备完毕，总有效帧数: {len(self.all_labels)}")

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        # 线性评估不需要数据增强
        points = self.all_pc_frames[idx]
        label = self.all_labels[idx]
        return torch.tensor(points, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def evaluate_linear_probe(backbone, train_loader_frames, val_loader_frames, config):
    device = config['device']
    backbone.eval()
    
    linear_classifier = nn.Linear(config['point_feature_dim'], config['num_classes']).to(device)
    optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    
    for _ in range(30):
        for frames, labels in train_loader_frames: 
            frames, labels = frames.to(device), labels.to(device)
            with torch.no_grad():
                features = backbone.encode(frames)
            optimizer.zero_grad()
            logits = linear_classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for frames, labels in val_loader_frames:
            frames, labels = frames.to(device), labels.to(device)
            features = backbone.encode(frames) 
            logits = linear_classifier(features)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    backbone.train()
    return accuracy


def main():
  
    config = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'point_feature_dim': 256,
        'projection_dim': 128,
        'num_classes': 6,
        'temperature': 0.1,
        'lr': 1e-4,
        'epochs':64,
        'batch_size': 64, 
        'probe_batch_size': 64, 
        'save_path': './softedge.pth',
    }

    ROOT_DIR = "allpoints" 
    SCALER_PATH = "scaler.gz"

    all_sample_ids = sorted([os.path.splitext(os.path.basename(f))[0].rsplit('-', 1)[0] 
                             for f in glob(os.path.join(ROOT_DIR, '*-motion.csv'))])
    
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

    print(f"数据集大小: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")
    
    
    scl_train_dataset = SCLPointCloudsDataset(source_dataset=train_dataset)
    scl_loader = DataLoader(scl_train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    
    probe_train_dataset = FramesDatasetForProbe(source_dataset=train_dataset)
    probe_val_dataset = FramesDatasetForProbe(source_dataset=val_dataset)
    probe_train_loader = DataLoader(probe_train_dataset, batch_size=config['probe_batch_size'], shuffle=True)
    probe_val_loader = DataLoader(probe_val_dataset, batch_size=config['probe_batch_size'])


    model = PointNetContrastive(config['point_feature_dim'], config['projection_dim']).to(config['device'])
    criterion = SupervisedContrastiveLoss(temperature=config['temperature']).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    print("\n--- 开始有监督对比学习训练 ---")
    best_scl_loss = float('inf')  
    best_val_accuracy = 0.0  
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        loop = tqdm(scl_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for view1, view2, labels in loop:
            images = torch.cat([view1, view2], dim=0).to(config['device'])
            labels = torch.cat([labels, labels], dim=0).to(config['device'])
            
            optimizer.zero_grad()
            projections = model(images)
            loss = criterion(projections, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())
            total_loss += loss.item()
        
        avg_scl_loss = total_loss / len(loop)
        
        val_accuracy = evaluate_linear_probe(model, probe_train_loader, probe_val_loader, config)

        print(f"Epoch {epoch+1}/{config['epochs']} | SCL Loss: {avg_scl_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.backbone.state_dict(), config['save_path'])
            print(f"💾 模型保存：Val Accuracy 最高 {best_val_accuracy:.4f}，已保存到 {config['save_path']}")

if __name__ == '__main__':
    main()