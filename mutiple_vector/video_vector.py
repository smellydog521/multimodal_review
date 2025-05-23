# video_vector.py

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import logging
from typing import List


# 自定义 ResNet50 特征提取类
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.base_model = models.resnet50(pretrained=True)
            for param in self.base_model.parameters():
                param.requires_grad = False  # 冻结模型参数
            self.features = nn.Sequential(*list(self.base_model.children())[:-1])
            logging.info("ResNet50 特征提取模型加载成功。")
        except Exception as e:
            logging.error(f"加载 ResNet50 模型时出错: {e}")
            raise e  # 重新抛出异常以便在外层处理

    def forward(self, x):
        try:
            x = self.features(x)  # 输出 (batch_size, 2048, 1, 1)
            return x.view(x.size(0), -1)  # 展平为 (batch_size, 2048)
        except Exception as e:
            logging.error(f"提取特征时出错: {e}")
            raise e  # 重新抛出异常以便在外层处理


# 1. 固定抽取视频中的16帧（自适应采样）
def extract_fixed_frames(video_path: str, num_frames: int = 16) -> List[np.ndarray]:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {video_path}")

        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logging.error("无法获取视频总帧数")
            return []

        # 均匀采样 num_frames 个帧索引
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                logging.warning(f"无法读取第 {idx} 帧")
        cap.release()
        logging.info(f"固定抽取了 {len(frames)} 帧来自视频 '{video_path}'")
        return frames
    except Exception as e:
        logging.error(f"提取帧时出错: {e}")
        return []


# 2. 提取帧特征（批量处理）
def extract_features(frames: List[np.ndarray], model: nn.Module, preprocess: transforms.Compose, device: str = 'cpu',
                     batch_size: int = 32) -> np.ndarray:
    if not frames:
        logging.warning("没有帧可供提取特征。")
        return np.array([])  # 返回空数组表示未提取到任何特征
    try:
        features = []
        batch = []
        for frame in frames:
            try:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                input_tensor = preprocess(img)
                batch.append(input_tensor)
            except Exception as e:
                logging.error(f"处理帧时出错: {e}")
                continue  # 跳过无法处理的帧
            if len(batch) == batch_size:
                input_batch = torch.stack(batch).to(device)
                with torch.no_grad():
                    batch_features = model(input_batch).cpu().numpy()
                features.append(batch_features)
                batch = []
        if batch:
            input_batch = torch.stack(batch).to(device)
            with torch.no_grad():
                batch_features = model(input_batch).cpu().numpy()
            features.append(batch_features)
        if features:
            features = np.concatenate(features, axis=0)
            logging.info(f"提取了 {features.shape[0]} 帧的特征")
            return features
        else:
            logging.warning("未提取到任何帧特征。")
            return np.array([])  # 返回空数组表示未提取到任何特征
    except Exception as e:
        logging.error(f"提取帧特征时出错: {e}")
        return np.array([])  # 返回空数组表示提取特征失败


# 3. 聚合特征
def aggregate_features(features: np.ndarray, method: str = "mean") -> np.ndarray:
    if features.size == 0:
        logging.warning("输入特征为空，无法进行聚合。")
        return np.array([])  # 返回空数组表示无法聚合
    try:
        if method == "mean":  # 平均值聚合
            aggregated = np.mean(features, axis=0)
        elif method == "max":  # 最大值聚合
            aggregated = np.max(features, axis=0)
        else:
            raise ValueError("Unsupported aggregation method")
        logging.info(f"聚合后的特征向量维度: {aggregated.shape}")
        return aggregated
    except Exception as e:
        logging.error(f"聚合特征时出错: {e}")
        return np.array([])  # 返回空数组表示聚合失败


# 4. 整合为一个方法
def process_video_to_vector(video_path: str, num_frames: int = 16, method: str = "mean",
                            device: str = 'cpu') -> np.ndarray:
    """
    将视频文件处理为特征向量
    :param video_path: 输入视频文件路径
    :param num_frames: 固定抽取的帧数（默认16帧）
    :param method: 聚合方法 ("mean" 或 "max")
    :param device: 计算设备 ("cpu" 或 "cuda")
    :return: 聚合后的视频特征向量，或空数组如果处理失败
    """
    logging.info(f"开始处理视频: {video_path}")
    try:
        # 1. 固定抽取视频中的 num_frames 帧
        frames = extract_fixed_frames(video_path, num_frames=num_frames)
        if not frames:
            logging.error("未提取到任何帧，无法生成视频特征向量")
            return np.array([])  # 返回空数组表示未提取到任何帧

        # 2. 初始化模型
        model = ResNet50FeatureExtractor().to(device)
        model.eval()
        logging.info(f"使用设备: {device}")

        # 3. 定义预处理流程
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 4. 提取帧特征
        frame_features = extract_features(frames, model, preprocess, device=device, batch_size=32)
        if frame_features.size == 0:
            logging.error("帧特征提取失败，无法生成视频特征向量")
            return np.array([])  # 返回空数组表示帧特征提取失败

        # 5. 聚合特征
        video_vector = aggregate_features(frame_features, method=method)
        if video_vector.size == 0:
            logging.error("特征聚合失败，无法生成视频特征向量")
            return np.array([])  # 返回空数组表示特征聚合失败

        logging.info("视频特征向量生成成功。")
        return video_vector
    except Exception as e:
        logging.error(f"处理视频时出错: {e}")
        return np.array([])  # 返回空数组表示整个处理流程失败
