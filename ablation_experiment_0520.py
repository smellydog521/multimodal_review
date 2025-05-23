#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
多模态 FCSoftmax 二分类脚本，支持单模态与双模态六种输入。
增加外层 10 次重复，每次内部执行 5 折交叉验证，记录每次重复-折次的指标，最后保存为 XLSX。
可选降维：FC、PCA（安全补零/截断）、Slice，通过 proj_method 和 proj_dim 控制。
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
import logging
from datetime import datetime

# ---------- 日志配置 ----------
current_date = datetime.now().strftime('%Y%m%d_%H%M')
LOG_DIR = './log'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f'modal_split_cv5x10_{current_date}.log')
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- 超参数及路径 ----------
TEXT_DIM      = 768
IMAGE_DIM     = 4096
VIDEO_DIM     = 2048
FEATURE_TOTAL = TEXT_DIM + IMAGE_DIM + VIDEO_DIM
HIDDEN_DIM    = 1280
BATCH_SIZE    = 32
LR            = 1e-3
NUM_EPOCHS    = 10
FOLDS         = 5
REPEATS       = 10
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_PATH  = '../PCAG_fusion_file/0501/all_comment_0501_PU.xlsx'
OUTPUT_DIR    = './file'
os.makedirs(OUTPUT_DIR, exist_ok=True)
ALL_METRICS_XLSX = os.path.join(OUTPUT_DIR, 'modal_split_cv5x10_all_metrics.xlsx')
MEAN_METRICS_XLSX = os.path.join(OUTPUT_DIR, 'modal_split_cv5x10_mean_metrics.xlsx')

# 降维配置
proj_dim    = 1280
proj_method = 'pca'  # 可选 'fc','pca','slice'

# ---------- 模型定义 ----------
class FCSoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, proj_dim, hidden_dim=256, proj_method='fc', num_classes=2):
        super().__init__()
        if proj_method == 'fc':
            self.proj = nn.Linear(input_dim, proj_dim)
            clf_in = proj_dim
        else:
            self.proj = nn.Identity()
            clf_in = input_dim
        self.relu = nn.ReLU()
        self.fc1  = nn.Linear(clf_in, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.proj(x))
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ---------- 评估函数 ----------
def evaluate_metrics(model, loader):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            p = F.softmax(logits, dim=1)[:, 1]
            preds.extend((p>=0.5).long().cpu().numpy())
            probs.extend(p.cpu().numpy())
            labels.extend(yb.cpu().numpy())
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    auc  = roc_auc_score(labels, probs)
    return {'acc':acc,'prec':prec,'rec':rec,'f1':f1,'auc':auc}, labels, probs

# ---------- 数据加载与平衡 ----------
def load_and_balance(path):
    df = pd.read_excel(path)
    y  = df.iloc[:, -1]
    mn = y.value_counts().min()
    bal = pd.concat([df[df.iloc[:,-1]==lbl].sample(mn,random_state=42)
                     for lbl in y.unique()],ignore_index=True)
    return bal.sample(frac=1, random_state=42)

# ---------- 主流程：重复 & 五折 CV ----------
if __name__=='__main__':
    df = load_and_balance(DATASET_PATH)
    y  = df.iloc[:, -1].values
    # 构建六种输入方式
    X_text  = df.iloc[:, :TEXT_DIM].values
    X_image = df.iloc[:, TEXT_DIM:TEXT_DIM+IMAGE_DIM].values
    X_video = df.iloc[:, TEXT_DIM+IMAGE_DIM:FEATURE_TOTAL].values
    methods = [
        ('text', X_text), ('image', X_image), ('video', X_video),
        ('text+image', np.concatenate([X_text,X_image],axis=1)),
        ('text+video', np.concatenate([X_text,X_video],axis=1)),
        ('image+video',np.concatenate([X_image,X_video],axis=1))
    ]
    all_records = []
    for name, X in methods:
        for rep in range(1, REPEATS+1):
            skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=rep)
            for fold,(tr,te) in enumerate(skf.split(X,y), start=1):
                X_tr, X_te = X[tr], X[te]
                y_tr, y_te = y[tr], y[te]
                # 降维/扩维
                odim = X_tr.shape[1]
                if proj_method=='pca':
                    if proj_dim<=odim:
                        pca = PCA(n_components=proj_dim, random_state=42)
                        X_tr = pca.fit_transform(X_tr); X_te = pca.transform(X_te)
                    else:
                        pad = proj_dim-odim
                        X_tr = np.hstack([X_tr, np.zeros((len(X_tr),pad))])
                        X_te = np.hstack([X_te, np.zeros((len(X_te),pad))])
                elif proj_method=='slice':
                    X_tr = X_tr[:,:proj_dim]; X_te = X_te[:,:proj_dim]
                # DataLoader
                tr_ld = DataLoader(TensorDataset(torch.tensor(X_tr,dtype=torch.float32), torch.tensor(y_tr,dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True)
                te_ld = DataLoader(TensorDataset(torch.tensor(X_te,dtype=torch.float32), torch.tensor(y_te,dtype=torch.long)), batch_size=BATCH_SIZE)
                # 模型
                inp_dim = FEATURE_TOTAL if proj_method=='fc' else proj_dim
                model = FCSoftmaxClassifier(inp_dim, proj_dim, HIDDEN_DIM, proj_method).to(DEVICE)
                opt   = torch.optim.Adam(model.parameters(), lr=LR)
                # 训练
                for _ in range(NUM_EPOCHS):
                    model.train()
                    for Xb,yb in tr_ld:
                        opt.zero_grad(); loss=F.cross_entropy(model(Xb.to(DEVICE)), yb.to(DEVICE)); loss.backward(); opt.step()
                # 评估
                m, labels, probs = evaluate_metrics(model, te_ld)
                m.update({'method':name,'repeat':rep,'fold':fold})
                all_records.append(m)
                logger.info(f"{name} Rep{rep} Fold{fold}: {m}")
    # 保存 Excel
    df_all = pd.DataFrame(all_records)
    df_all.to_excel(ALL_METRICS_XLSX, index=False)
    df_mean = df_all.groupby('method')[['acc','prec','rec','f1','auc']].mean().reset_index()
    df_mean.to_excel(MEAN_METRICS_XLSX, index=False)
    logger.info("10次五折CV所有指标及平均值已保存。")
    print(f"全部指标: {ALL_METRICS_XLSX}\n平均指标: {MEAN_METRICS_XLSX}")
