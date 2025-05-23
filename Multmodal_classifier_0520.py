import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import logging
from datetime import datetime

# ---------- 日志配置 ----------
current_date = datetime.now().strftime('%m%d')
LOG_DIR = './log'
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f'FCSoftmax_CV5x10_{current_date}.log')
logging.basicConfig(
    filename=log_file,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - Line: %(lineno)d'
)
logger = logging.getLogger(__name__)

# ---------- 模型 ----------
class FCSoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)  # 输出 logits

# ---------- 评估 ----------
def evaluate_model(model, loader, device):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            p = torch.softmax(logits, dim=1)[:, 1]
            preds.extend((p >= 0.5).long().cpu().numpy())
            probs.extend(p.cpu().numpy())
            labels.extend(yb.cpu().numpy())
    return np.array(preds), np.array(labels), np.array(probs)

# ---------- 指标 ----------
def compute_metrics(preds, labels, probs):
    acc  = (preds == labels).mean()
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    auc  = roc_auc_score(labels, probs) if len(np.unique(labels))>1 else 0.0
    fpr, tpr, _ = roc_curve(labels, probs, pos_label=1)
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc, 'fpr': fpr, 'tpr': tpr}

# ---------- 设置 ----------
weidu = 1280
files = [
    ("Dynamic",         f"../PCAG_fusion_file/0501/Dynamic/all_comments_fusion_{weidu}_PU.xlsx"),
    ("Dynamic_all",     f"../PCAG_fusion_file/0501/Dynamic_all/all_comments_fusion_{weidu}_PU.xlsx"),
    ("Dynamic_CAG",     f"../PCAG_fusion_file/0501/Dynamic_CAG/all_comments_fusion_{weidu}_PU.xlsx"),
    ("Dynamic_PG",      f"../PCAG_fusion_file/0501/Dynamic_PG/all_comments_fusion_{weidu}_PU.xlsx"),
    ("PCAG",            f"../PCAG_fusion_file/0501/PCAG/all_comments_fusion_{weidu}_PU.xlsx"),
    ("Cross_Attention", f"../PCAG_fusion_file/0501/Cross_Attention/all_comments_fusion_{weidu}_PU.xlsx"),
]
feature_dim = weidu
hidden_dim  = 256
num_epochs  = 10
batch_size  = 32
lr          = 1e-3
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 主循环：10次五折交叉验证 ----------
all_records = []  # 存储所有重复和折次的指标
for name, path in files:
    df = pd.read_excel(path)
    y_all = df.iloc[:, -1]
    min_cnt = y_all.value_counts().min()

    # 平衡并乱序
    balanced = pd.concat([
        df[df.iloc[:, -1]==lbl].sample(min_cnt, random_state=42)
        for lbl in y_all.unique()
    ], ignore_index=True).sample(frac=1, random_state=42)
    X = balanced.iloc[:, :feature_dim].values
    y = balanced.iloc[:, -1].values

    # 10 次重复
    for rep in range(1, 11):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
        fold = 1
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                                   torch.tensor(y_tr, dtype=torch.long)),
                                       batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                                   torch.tensor(y_val, dtype=torch.long)),
                                       batch_size=batch_size)

            model = FCSoftmaxClassifier(feature_dim, hidden_dim, 2).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # 训练
            model.train()
            for epoch in range(num_epochs):
                for Xb, yb in train_loader:
                    optimizer.zero_grad()
                    logits = model(Xb.to(device))
                    loss = F.cross_entropy(logits, yb.to(device))
                    loss.backward()
                    optimizer.step()

            # 评估
            preds, labels_true, probs = evaluate_model(model, val_loader, device)
            m = compute_metrics(preds, labels_true, probs)
            logger.info(f"{name} Rep{rep} Fold{fold} — Acc={m['acc']:.4f}, Prec={m['prec']:.4f}, Rec={m['rec']:.4f}, F1={m['f1']:.4f}, AUC={m['auc']:.4f}")

            all_records.append({
                'file': name,
                'repeat': rep,
                'fold': fold,
                'acc': m['acc'],
                'prec': m['prec'],
                'rec': m['rec'],
                'f1': m['f1'],
                'auc': m['auc']
            })
            fold += 1

# ---------- 汇总与保存 ----------
records_df = pd.DataFrame(all_records)
# 保存所有重复-折次的指标
os.makedirs('./file', exist_ok=True)
records_df.to_excel('./file/cv5x10_all_metrics.xlsx', index=False)

# 计算平均值
mean_df = records_df.groupby('file')[['acc','prec','rec','f1','auc']].mean().reset_index()
mean_df.to_excel('./file/cv5x10_mean_metrics.xlsx', index=False)
logger.info("10次五折交叉验证所有指标及平均值已保存。")

print("已完成 10 次五折交叉验证，结果保存在 './file/' 目录下。")
