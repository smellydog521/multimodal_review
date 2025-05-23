#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 顾和益
# @time    : 2024/12/21 16:43 (modified)
# @function: 多模态向量化（无 PCA）
# @version : V2
from datetime import datetime
import pandas as pd
import os
import numpy as np
from mutiple_vector.text_vector import process_text
from mutiple_vector.image_vector import process_images
from mutiple_vector.video_vector import process_video_to_vector
import logging

# 设置日志文件路径
current_date = datetime.now().strftime('%m%d')
LOG_DIR = '../log'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, f'vectorize_{current_date}.log')

# 配置日志记录
logging.basicConfig(
    filename=LOG_FILENAME,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - Line: %(lineno)d'
)
logger = logging.getLogger(__name__)


def process_file(file_path,
                 output_path,
                 model_choice='resnet50',
                 device=None,
                 image_cols=None  # 图片所在列的列表
                 ):
    """
    处理文件，支持按行处理多列图片地址，并对每一行进行向量化合并，
    同时保留原xlsx文件最后一列的标签值，将其追加到最终向量化结果的最后一列。

    假设 Excel 文件格式：
    - 第1列：文本数据
    - 中间列：图片地址（多列）
    - 倒数第二列：视频路径
    - 最后一列：标签
    """
    try:
        logger.info(f"开始读取文件: {file_path}")
        if image_cols is None:
            image_cols = [1]

        df = pd.read_excel(file_path, sheet_name="Sheet1")
        logger.info("文件读取完成。")

        # --------------------- 数据提取 ---------------------
        texts = df.iloc[:, 0].astype(str).tolist()
        video_paths = df.iloc[:, -2].astype(str).tolist()
        labels = df.iloc[:, -1].tolist()

        # --------------------- 文本处理 ---------------------
        logger.info("开始处理文本嵌入...")
        text_embeddings = process_text(texts, device)
        logger.info(f"文本嵌入形状: {text_embeddings.shape}")

        # --------------------- 图片处理（按行） ---------------------
        logger.info("开始按行处理图片嵌入...")
        row_image_embeddings = []
        all_failed_images = []
        each_image_dim = text_embeddings.shape[1] if False else 2048  # 若动态获取可改此处

        for idx, row in df.iterrows():
            row_embs = []
            for col in image_cols:
                img_path = row.iloc[col]
                if pd.isna(img_path) or not str(img_path).strip():
                    row_embs.append(np.zeros(each_image_dim))
                    logger.warning(f"行{idx}列{col}图片无效，补零向量")
                else:
                    emb, failed = process_images([str(img_path)], device)
                    row_embs.append(emb[0] if emb.shape[0] > 0 else np.zeros(each_image_dim))
                    all_failed_images.extend(failed)
            row_image_embeddings.append(np.concatenate(row_embs))
        image_embeddings = np.stack(row_image_embeddings, axis=0)
        logger.info(f"图片嵌入形状: {image_embeddings.shape}")

        # --------------------- 视频处理 ---------------------
        logging.info("开始处理视频嵌入...")
        video_embeddings = []
        for idx, video_path in enumerate(video_paths):
            if video_path and str(video_path).lower() != 'nan':
                try:
                    # 移除了 frame_rate 参数
                    video_embedding = process_video_to_vector(
                        video_path,
                        method="mean",
                        device=device
                    )
                    if video_embedding.size > 0:
                        video_embeddings.append(video_embedding)
                    else:
                        video_embeddings.append(np.zeros(256))
                except Exception as e:
                    video_embeddings.append(np.zeros(256))
                    logging.error(f"处理视频 '{video_path}' 出错: {e}", exc_info=True)
            else:
                video_embeddings.append(np.zeros(256))
                logging.warning(f"视频路径为空或无效: 索引 {idx}")

        video_embeddings = np.array(video_embeddings)
        logging.info(f"视频嵌入的形状: {video_embeddings.shape}")

        # --------------------- 数据对齐 ---------------------
        n = min(len(texts), image_embeddings.shape[0], video_embeddings.shape[0])
        texts = texts[:n]
        text_embeddings = text_embeddings[:n]
        image_embeddings = image_embeddings[:n]
        video_embeddings = video_embeddings[:n]
        labels = labels[:n]
        logger.info(f"对齐后样本数: {n}")

        # --------------------- 合并嵌入 & 保存 ---------------------
        merged = np.concatenate([text_embeddings, image_embeddings, video_embeddings], axis=1)
        logger.info(f"合并后特征形状: {merged.shape}")

        df_out = pd.DataFrame(merged)
        df_out['label'] = labels
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_out.to_excel(output_path, index=False, header=False)
        logger.info(f"结果已保存: {output_path}")

        if all_failed_images:
            logger.warning(f"图片处理失败共 {len(all_failed_images)} 个：{all_failed_images}")
        else:
            logger.info("所有图片均处理成功。")

    except Exception as e:
        logger.error(f"处理出错: {e}", exc_info=True)


if __name__ == "__main__":
    import torch
    cols = [1, 2]
    inp = "所有评论文本及附件地址.xlsx"
    out = "../process_file/all_comment_0501.xlsx"
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    process_file(inp, out, device=dev, image_cols=cols)
