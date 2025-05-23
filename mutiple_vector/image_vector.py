import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import logging
from typing import List, Tuple

class ImageVectorizer:
    def __init__(self, model_choice: str = 'resnet50', device: str = 'cpu'):
        if model_choice == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            feature_dim = 2048
        elif model_choice == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            feature_dim = 512
        else:
            raise ValueError("Invalid model_choice. Choose either 'resnet50' or 'resnet18'.")

        # 去除最后的全连接层
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1]).to(device)
        self.model.eval()  # 设置为评估模式
        self.device = device

        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        logging.info(f"{model_choice} 模型加载并准备就绪。")

    def get_image_embedding(self, image_path: str) -> None:
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature = self.model(image)
            embedding = feature.cpu().numpy().flatten()  # 返回2048或512维向量
            logging.info(f"成功提取图像 '{image_path}' 的特征向量。")
            return embedding
        except Exception as e:
            logging.error(f"处理图像 '{image_path}' 时出错: {e}")
            return None

def process_images(image_paths: List[str], device: str = 'cpu') -> Tuple[np.ndarray, List[str]]:
    image_vectorizer = ImageVectorizer(model_choice='resnet50', device=device)
    embeddings = []
    failed_images = []
    for image_path in image_paths:
        embedding = image_vectorizer.get_image_embedding(image_path)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            failed_images.append(image_path)
    embeddings_array = np.array(embeddings)
    logging.info(f"处理了 {len(embeddings)} 张图像，嵌入形状: {embeddings_array.shape}")
    return embeddings_array, failed_images