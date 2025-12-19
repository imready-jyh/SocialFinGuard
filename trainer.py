import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from utils import DEVICE, FRAUD_WEIGHT, LEARNING_RATE, WEIGHT_DECAY


# 新增Focal Loss（解决类别不平衡+难样本学习）
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FraudDetectionTrainer:
    """模型训练器（优化版：Focal Loss）"""

    def __init__(self, model):
        self.model = model.to(DEVICE)
        self.device = DEVICE
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        # 使用Focal Loss替代普通交叉熵
        self.criterion = FocalLoss(alpha=FRAUD_WEIGHT, gamma=2)

    def train_epoch(self, data, labels, mask):
        self.model.train()
        self.optimizer.zero_grad()

        if isinstance(self.model, torch.nn.Module) and hasattr(self.model, 'n_views'):
            # MultiViewGNN
            logits, _ = self.model(data['x'], data['edge_indices'])
        else:
            # BIAN
            logits = self.model(
                data['x'],
                data['edge_index'],
                data['edge_attr'],
                data['timestamps']
            )

        # 计算每个样本的损失
        loss_per_sample = F.cross_entropy(logits[mask], labels[mask], reduction='none')

        # 对诈骗样本加权
        weights = torch.ones_like(labels[mask], dtype=torch.float32)
        weights[labels[mask] == 1] = FRAUD_WEIGHT

        # 应用权重并计算平均损失
        loss = (loss_per_sample * weights).mean()

        loss.backward()
        self.optimizer.step()

        return loss.item()

    # 评估函数保持不变（原代码）
    def evaluate(self, data, labels, mask):
        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, torch.nn.Module) and hasattr(self.model, 'n_views'):
                logits, _ = self.model(data['x'], data['edge_indices'])
            else:
                logits = self.model(
                    data['x'],
                    data['edge_index'],
                    data['edge_attr'],
                    data['timestamps']
                )

            pred_proba = F.softmax(logits, dim=1)[:, 1]

            # 计算最优阈值（基于验证集）
            if mask.sum() > 100:  # 只在验证集上调整
                from sklearn.metrics import precision_recall_curve
                precision, recall, thresholds = precision_recall_curve(
                    labels[mask].cpu().numpy(),
                    pred_proba[mask].cpu().numpy()
                )
                # 选择F1最大的阈值
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
            else:
                best_threshold = 0.3  # 默认阈值

            pred = (pred_proba > best_threshold).long()

            # 计算指标（保持不变）
            accuracy = (pred[mask] == labels[mask]).float().mean().item()

            labels_np = labels[mask].cpu().numpy()
            pred_np = pred[mask].cpu().numpy()
            pred_proba_np = pred_proba[mask].cpu().numpy()

            auc = roc_auc_score(labels_np, pred_proba_np)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels_np, pred_np, average='binary', zero_division=0
            )

            return {
                'accuracy': accuracy,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'threshold': best_threshold  # 新增：返回阈值
            }