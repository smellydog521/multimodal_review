#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
PU 学习脚本，包含：
- 动态伪负标签生成（KMeans + 方差筛选 + PCA + 可选大/小簇映射）
- 一次性 Platt/Isotonic 校准
- 动态阈值调整（基于概率分布分位数）
- 记录每次迭代的概率分布和标签分布
- 固定标签映射（-1 → 0）
"""

import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import xgboost as xgb
from lightgbm import LGBMClassifier
from tensorflow import sigmoid

# ----- 日志配置 -----
current_date = datetime.now().strftime('%m%d')
LOG_DIR = '../log/PU'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, f'PU_{current_date}.log')
logging.basicConfig(
    filename=LOG_FILENAME,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - 行号: %(lineno)d'
)
logger = logging.getLogger(__name__)


# ----- 伪负标签生成：KMeans + 方差筛选 + PCA + 动态映射 -----
def pseudo_negative_class_kmeans(
    X_unlabeled,
    X_pos=None,
    y_pos=None,
    var_threshold: float = 0.01,
    pca_components: int = 10,
    alpha: float = 1,
    beta: float = 0,
    cluster_size_preference: str = 'large',
    random_state: int = 42
):
    # 方差筛选
    selector = VarianceThreshold(threshold=var_threshold)
    X_var = selector.fit_transform(X_unlabeled)
    X_pos_var = selector.transform(X_pos) if X_pos is not None else None

    # PCA 降维
    pca = PCA(n_components=min(pca_components, X_var.shape[1]), random_state=random_state)
    X_pca = pca.fit_transform(X_var)
    X_pos_pca = pca.transform(X_pos_var) if X_pos_var is not None else None

    # KMeans 聚类
    kmeans = KMeans(
        n_clusters=2, n_init=20, max_iter=300,
        tol=1e-4, init='k-means++', random_state=random_state
    )
    labels = kmeans.fit_predict(X_pca)
    centers = kmeans.cluster_centers_

    # 簇大小得分
    counts = np.bincount(labels, minlength=2)
    total = counts.sum()
    if cluster_size_preference == 'large':
        size_score = counts / total
    else:
        size_score = 1.0 - counts / total

    # 簇中心距离得分
    if X_pos_pca is not None and len(X_pos_pca) > 0:
        pos_center = X_pos_pca.mean(axis=0)
        dist = np.linalg.norm(centers - pos_center, axis=1)
        dist_score = dist / dist.max()
    else:
        dist_score = np.zeros(2)

    # 综合得分并选负类簇
    scores = alpha * size_score + beta * dist_score
    neg_cluster = np.argmax(scores)

    # 生成伪标签：负类簇→0，其他→1
    y_pseudo = np.where(labels == neg_cluster, 0, 1)

    # 日志输出
    logger.info(f"簇大小: {counts.tolist()}, 大小得分: {size_score.tolist()}")
    logger.info(f"距离得分: {dist_score.tolist()}, 综合得分: {scores.tolist()}")
    logger.info(f"选定负类簇编号: {neg_cluster} (偏好={cluster_size_preference})")

    return y_pseudo


# ----- 自训练迭代：记录概率分布 + 动态阈值筛选 -----
def pseudo_negative_class_self_training(
    X_unlabeled, X_pos, model,
    base_threshold=0.7, dynamic_quantile=75, iteration=1
):
    # 1. 预测校准后正类概率
    probas_pos = model.predict_proba(X_unlabeled)[:, 1]

    # 2. 记录概率分布的分位数
    p0, p25, p50, p75_val, p100 = np.percentile(
        probas_pos, [0, 25, 50, dynamic_quantile, 100]
    )
    logger.info(
        f"迭代 {iteration}：probas_pos 分布 [min={p0:.3f}, 25%={p25:.3f}, "
        f"50%={p50:.3f}, {dynamic_quantile}%={p75_val:.3f}, max={p100:.3f}]"
    )

    # 3. 动态阈值
    threshold = max(base_threshold, p75_val)
    logger.info(f"迭代 {iteration}：动态阈值 = {threshold:.3f}")

    # 4. 筛选高置信度正类
    mask = probas_pos > threshold
    logger.info(f"迭代 {iteration}：高于阈值样本数 = {mask.sum()}")

    # 5. 更新 X_pos 与 X_unlabeled
    X_new_pos = X_unlabeled[mask]
    X_pos = np.vstack((X_pos, X_new_pos))
    X_unlabeled = X_unlabeled[~mask]

    # 6. 剩余未标记样本伪负标签全部设为 -1
    y_pseudo_neg_new = np.full(X_unlabeled.shape[0], -1)

    return y_pseudo_neg_new, X_pos, X_unlabeled


# ----- 主训练流程：一次性校准 + 自训练迭代，并记录标签分布 -----
def train_pu_model_with_self_training(
    X, y, pseudo_negative_class_func,
    model_type='random_forest',
    max_iter=5, base_threshold=0.7,
    dynamic_quantile=75, cv_method='isotonic'
):
    # 划分正类与未标记
    X_pos = X[y == 1]
    X_unlabeled = X[y == -1]

    # 初始化基模型
    if model_type == 'random_forest':
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        base_model = xgb.XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42
        )
    elif model_type == 'lightgbm':
        base_model = LGBMClassifier(
            learning_rate=0.05, num_leaves=31, random_state=42
        )
    elif model_type == 'extratrees':
        base_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"不支持的 model_type: {model_type}")

    # 初次伪负预测
    y_pseudo_neg = pseudo_negative_class_func(X_unlabeled, X_pos, y[y == 1])

    # 构建初次训练集并映射 -1→0
    X_train = np.vstack((X_pos, X_unlabeled))
    y_train = np.concatenate((np.ones(X_pos.shape[0]), y_pseudo_neg))
    y_train = np.where(y_train == -1, 0, y_train)

    # 记录迭代 0 标签分布
    labels0, counts0 = np.unique(y_train, return_counts=True)
    logger.info(f"迭代 0：标签分布 = {dict(zip(labels0.tolist(), counts0.tolist()))}")

    # 标准化并训练
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    base_model.fit(X_train_scaled, y_train)

    model = base_model
    # 自训练迭代
    for i in range(1, max_iter + 1):
        start = time.time()
        y_pseudo_neg, X_pos, X_unlabeled = pseudo_negative_class_self_training(
            X_unlabeled, X_pos, model,
            base_threshold=base_threshold,
            dynamic_quantile=dynamic_quantile,
            iteration=i
        )
        if X_unlabeled.size == 0 or y_pseudo_neg.size == 0:
            logger.info(f"迭代 {i}：无新增高置信度样本，提前结束。")
            break

        # 重建训练集并映射 -1→0
        X_train = np.vstack((X_pos, X_unlabeled))
        y_train = np.concatenate((np.ones(X_pos.shape[0]), y_pseudo_neg))
        y_train = np.where(y_train == -1, 0, y_train)

        # 记录本轮标签分布
        labels_i, counts_i = np.unique(y_train, return_counts=True)
        logger.info(f"迭代 {i}：标签分布 = {dict(zip(labels_i.tolist(), counts_i.tolist()))}")

        # 标准化并重新训练基学习器
        X_train_scaled = scaler.transform(X_train)
        # model.estimator.fit(X_train_scaled, y_train)
        model.fit(X_train_scaled, y_train)

        logger.info(f"迭代 {i} 耗时 {time.time() - start:.4f} 秒")

    return model, scaler, X_train, y_train


# ----- 数据加载 -----
def load_data_from_excel(file_path):
    df = pd.read_excel(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    logger.info(f"原始样本总数 = {len(X)}")
    return df, X, y


# ----- 主程序入口 -----
if __name__ == "__main__":
    weidu = "1280"
    # folder = "PCAG"
    # folder = "Dynamic"
    # folder = "Dynamic_all"
    # folder = "Dynamic_CAG"
    folder = "Dynamic_PG"
    # folder = "Cross_Attention"
    file_path = f'../PCAG_fusion_file/0501/{folder}/all_comments_fusion_{weidu}.xlsx'
    # file_path = f'../PCAG_fusion_file/0426/all_comment_0424.xlsx'

    df_original, X, y = load_data_from_excel(file_path)

    model, scaler, X_train, y_train = train_pu_model_with_self_training(
        X, y,
        pseudo_negative_class_kmeans,
        model_type='lightgbm',
        max_iter=5,
        base_threshold=0.8,
        dynamic_quantile=97,
        cv_method='sigmoid'
    )

    # 全量数据预测
    X_all_scaled = scaler.transform(X)
    new_labels = model.predict(X_all_scaled)
    new_labels = np.where(new_labels == -1, 0, new_labels)

    # 保存结果
    df_modified = df_original.iloc[:, :-1].copy()
    df_modified['trained_label'] = new_labels
    output_path = f'../PCAG_fusion_file/0501/{folder}/all_comments_fusion_{weidu}_PU.xlsx'
    # output_path = f'../PCAG_fusion_file/0426/all_comment_0424_PU.xlsx'
    df_modified.to_excel(output_path, index=False)
    logger.info(f"已完成并保存结果至: {output_path}")
##对比实验：1.直接拼接 2.单模态投票法，
##消融实验：1.单模态或者双模态 2.PCAG 3.动态分类及变体