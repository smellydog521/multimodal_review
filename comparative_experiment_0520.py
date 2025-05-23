#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
多模态 FCSoftmax 二分类脚本，支持两种策略：
 1. 全特征 (all_features)
 2. 模态投票 (modality_voting)
并在外层进行 10 次重复，每次内部执行 5 折交叉验证，记录每次重复-折次的指标，最后保存为 XLSX。
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
import logging
from datetime import datetime

# ---------- 日志配置 ----------
current_date = datetime.now().strftime('%m%d')
LOG_DIR = './log'
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f'Multimodal_CV5x10_{current_date}.log')
logging.basicConfig(
    filename=log_file,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - Line: %(lineno)d'
)
logger = logging.getLogger(__name__)

# ---------- 超参数 ----------
text_dim = 768
image_dim = 4096
video_dim = 2048
feature_total = text_dim + image_dim + video_dim
proj_dim = 1280       # 投影目标维度
proj_method = 'pca'    # 'fc','pca','slice'
hidden_dim = 256
batch_size = 32
lr = 1e-3
num_epochs = 10
n_folds = 5
n_repeats = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = '../PCAG_fusion_file/0501/all_comment_0501_PU.xlsx'
output_dir = './file'
os.makedirs(output_dir, exist_ok=True)
all_metrics_path = os.path.join(output_dir, 'multimodal_cv5x10_all_metrics.xlsx')
mean_metrics_path = os.path.join(output_dir, 'multimodal_cv5x10_mean_metrics.xlsx')

# ---------- 模型 ----------
class FCSoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, proj_dim, hidden_dim, proj_method='fc', num_classes=2):
        super().__init__()
        if proj_method == 'fc':
            self.proj = nn.Linear(input_dim, proj_dim)
        else:
            self.proj = nn.Identity()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(proj_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.proj(x))
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ---------- 评估 ----------
def evaluate_model(model, loader):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            p = F.softmax(logits, dim=1)[:, 1]
            preds.extend((p >= 0.5).long().cpu().numpy())
            probs.extend(p.cpu().numpy())
            labels.extend(yb.cpu().numpy())
    return np.array(preds), np.array(labels), np.array(probs)

# ---------- 数据加载与平衡 ----------
def load_and_balance(path):
    df = pd.read_excel(path)
    y = df.iloc[:, -1]
    min_count = y.value_counts().min()
    balanced = pd.concat([
        df[df.iloc[:, -1] == lbl].sample(min_count, random_state=42)
        for lbl in y.unique()
    ], ignore_index=True).sample(frac=1, random_state=42)
    return balanced

# ---------- 训练与评估两个策略 ----------
def train_all_features(X, y):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    records = []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        orig_dim = X_tr.shape[1]
        # 投影 / 补零 / 截断
        if proj_method == 'pca' and proj_dim <= orig_dim:
            pca = PCA(n_components=proj_dim, random_state=42)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        elif proj_method == 'pca':
            pad = proj_dim - orig_dim
            X_tr = np.hstack([X_tr, np.zeros((len(X_tr), pad))])
            X_te = np.hstack([X_te, np.zeros((len(X_te), pad))])
        elif proj_method == 'slice':
            X_tr = X_tr[:, :proj_dim]
            X_te = X_te[:, :proj_dim]
        # 'fc' 留给模型内部
        # DataLoader
        tr_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.long)
            ), batch_size=batch_size, shuffle=True
        )
        te_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_te, dtype=torch.float32),
                torch.tensor(y_te, dtype=torch.long)
            ), batch_size=batch_size
        )
        # 模型
        input_dim = feature_total if proj_method == 'fc' else proj_dim
        model = FCSoftmaxClassifier(input_dim, proj_dim, hidden_dim, proj_method).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        # 训练
        for _ in range(num_epochs):
            model.train()
            for Xb, yb in tr_loader:
                opt.zero_grad()
                loss = F.cross_entropy(model(Xb.to(device)), yb.to(device))
                loss.backward()
                opt.step()
        # 评估
        preds, labels, probs = evaluate_model(model, te_loader)
        auc = roc_auc_score(labels, probs)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        records.append({'method': 'all_features', 'fold': fold, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc})
        logger.info(f"All feat Fold{fold}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    return records


def train_modality_voting(X, y):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    records = []
    # 划分三模态特征
    X_text = X[:, :text_dim]
    X_image = X[:, text_dim:text_dim + image_dim]
    X_video = X[:, text_dim + image_dim:]
    for fold, (tr, te) in enumerate(skf.split(X_text, y), start=1):
        loaders = {}
        for name_modal, X_all in [('text', X_text), ('image', X_image), ('video', X_video)]:
            X_tr, X_te = X_all[tr].copy(), X_all[te].copy()
            orig_dim = X_tr.shape[1]
            if proj_method == 'pca' and proj_dim <= orig_dim:
                pca = PCA(n_components=proj_dim, random_state=42)
                X_tr = pca.fit_transform(X_tr)
                X_te = pca.transform(X_te)
            elif proj_method == 'pca':
                pad = proj_dim - orig_dim
                X_tr = np.hstack([X_tr, np.zeros((len(X_tr), pad))])
                X_te = np.hstack([X_te, np.zeros((len(X_te), pad))])
            elif proj_method == 'slice':
                X_tr = X_tr[:, :proj_dim]
                X_te = X_te[:, :proj_dim]
            loaders[name_modal] = (
                DataLoader(
                    TensorDataset(
                        torch.tensor(X_tr, dtype=torch.float32),
                        torch.tensor(y[tr], dtype=torch.long)
                    ), batch_size=batch_size, shuffle=True
                ),
                DataLoader(
                    TensorDataset(
                        torch.tensor(X_te, dtype=torch.float32),
                        torch.tensor(y[te], dtype=torch.long)
                    ), batch_size=batch_size
                )
            )
        probs_sum, labels = None, None
        for name_modal, (tr_loader, te_loader) in loaders.items():
            model = FCSoftmaxClassifier(proj_dim, proj_dim, hidden_dim, proj_method).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(num_epochs):
                model.train()
                for Xb, yb in tr_loader:
                    opt.zero_grad()
                    loss = F.cross_entropy(model(Xb.to(device)), yb.to(device))
                    loss.backward()
                    opt.step()
            _, labels, probs = evaluate_model(model, te_loader)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        avg_probs = probs_sum / 3
        preds = (avg_probs >= 0.5).astype(int)
        auc = roc_auc_score(labels, avg_probs)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        records.append({'method': 'modality_voting', 'fold': fold, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc})
        logger.info(f"Mod vote Fold{fold}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    return records

# ---------- 主流程：重复 & CV ----------
if __name__ == '__main__':
    df = load_and_balance(dataset_path)
    X = df.iloc[:, :feature_total].values
    y = df.iloc[:, -1].values
    all_records = []
    for rep in range(1, n_repeats + 1):
        torch.manual_seed(rep)
        np.random.seed(rep)
        rec_all = train_all_features(X, y)
        for r in rec_all:
            r['repeat'] = rep
        rec_mod = train_modality_voting(X, y)
        for r in rec_mod:
            r['repeat'] = rep
        all_records.extend(rec_all + rec_mod)
    df_records = pd.DataFrame(all_records)
    df_records.to_excel(all_metrics_path, index=False)
    mean_df = df_records.groupby(['method'])[['acc','prec','rec','f1','auc']].mean().reset_index()
    mean_df.to_excel(mean_metrics_path, index=False)
    logger.info("CV5x10 所有指标和平均值已保存。")
    print(f"已保存所有指标: {all_metrics_path}\n平均指标: {mean_metrics_path}")
