"""
Stage A: Training 3-layer CNN on First-Order Residual Stream (PGC Single-Stream).
Uses random scaling factors, JPEG/Blur augmentations, and PyTorch cosine scheduler.
"""
import os
import io
import math
import glob
import random
import tarfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

class ResidualOp(nn.Module):
    """水平与垂直一阶差分，截断在 [-3, 3] 区间并归一化"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] in range [0, 255]
        diff_h = x[:, :, :, 1:] - x[:, :, :, :-1]
        diff_h = F.pad(diff_h, (0, 1, 0, 0), mode='constant', value=0)
        diff_v = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_v = F.pad(diff_v, (0, 0, 0, 1), mode='constant', value=0)
        res = torch.cat([diff_h, diff_v], dim=1)
        res = torch.clamp(res, -3.0, 3.0) / 3.0
        return res

class ResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = ResidualOp()
        self.features = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 接收 uint8 RGB (0-255) 经残差算子提取特征
        r = self.residual(x)
        feat = self.features(r)
        out = self.fc(torch.flatten(feat, 1))
        return out

class ShardDataset(Dataset):
    def __init__(self, tar_paths):
        self.samples = []
        for p in tar_paths:
            with tarfile.open(p, "r") as tf:
                for member in tf.getmembers():
                    if member.name.endswith(('.png', '.jpg')):
                        lbl = int(member.name.split('_')[0])
                        self.samples.append((p, member.name, lbl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tar_path, member_name, label = self.samples[idx]
        with tarfile.open(tar_path, "r") as tf:
            b = tf.extractfile(member_name).read()
        img = Image.open(io.BytesIO(b)).convert("RGB")
        
        # 随机缩放比因子 (log-uniform 0.5 - 2.0) 消除尺寸捷径
        scale = math.exp(random.uniform(math.log(0.5), math.log(2.0)))
        nw, nh = max(32, int(img.width * scale)), max(32, int(img.height * scale))
        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        
        # 中心裁剪 224x224
        if nw < 224 or nh < 224:
            img = img.resize((max(224, nw), max(224, nh)), Image.Resampling.BILINEAR)
        img = TF.center_crop(img, [224, 224])
        
        tensor = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1)
        return tensor, torch.tensor(label, dtype=torch.float32)

def train():
    os.makedirs("_ckpt/stageA", exist_ok=True)
    all_shards = sorted(glob.glob("/content/shards/*.tar"))
    train_shards = all_shards[:-2] # 保留最后 2 个分片作为 Dev 集合
    
    loader = DataLoader(ShardDataset(train_shards), batch_size=128, shuffle=True, num_workers=2)
    model = ResidualCNN().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(3):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.cuda(), y.cuda().unsqueeze(1)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/3 Loss: {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), f"_ckpt/stageA/epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), "_ckpt/stageA/best.pt")

if __name__ == "__main__":
    import numpy as np
    train()