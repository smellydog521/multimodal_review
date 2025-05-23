# text_vector.py
import logging

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class TextVectorizer:
    def __init__(self, model_name='bert-base-chinese', device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()  # 设置为评估模式
        self.device = device
        logging.info(f"Loaded {model_name} model on device {self.device}")

    def get_text_embedding(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # 获取 [CLS] token 的嵌入作为句子的表示
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()  # 返回768维向量
            logging.info(f"Text embedding shape: {embedding.shape}")  # 调试信息
            return embedding
        except Exception as e:
            logging.error(f"Error processing text '{text}': {e}")
            return None

# 用于调用的统一函数
def process_text(texts, device):
    text_vectorizer = TextVectorizer(device=device)
    embeddings = []
    for text in texts:
        embedding = text_vectorizer.get_text_embedding(text)
        if embedding is not None:
            embeddings.append(embedding)
    embeddings_array = np.array(embeddings)
    logging.info(f"Processed {len(embeddings)} texts with embedding shape: {embeddings_array.shape}")
    return embeddings_array
