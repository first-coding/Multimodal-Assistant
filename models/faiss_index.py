import faiss
import numpy as np
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# 加载 CLIP 模型和处理器
model = CLIPModel.from_pretrained("./clip-vit-base-patch32/")
processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32/")

# 加载图像并提取特征
def extract_features_from_image(image_path):
    """
    从图像路径提取图像特征
    :param image_path: 图像文件路径
    :return: 图像特征向量
    """
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.detach().numpy().flatten()  # 转换为一维数组

# 从目录中加载所有图像，并提取特征
def create_faiss_index(image_dir, index_file="./faiss.index"):
    """
    创建FAISS索引并保存
    :param image_dir: 图像文件夹路径
    :param index_file: 保存FAISS索引的文件名
    :return: None
    """
    # 列出目录下所有图像文件
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 提取所有图像的特征
    features = []
    for image_path in image_paths:
        features.append(extract_features_from_image(image_path))
    
    # 将特征列表转换为NumPy数组
    features = np.array(features).astype('float32')
    
    # 创建FAISS索引
    dimension = features.shape[1]  # 特征的维度
    index = faiss.IndexFlatL2(dimension)  # 使用L2距离计算相似度
    index.add(features)  # 添加图像特征到索引中
    
    # 保存FAISS索引
    faiss.write_index(index, index_file)
    print(f"FAISS index saved to {index_file}")

# 加载FAISS索引
def load_faiss_index(index_file="./faiss.index"):
    """
    加载已保存的FAISS索引
    :param index_file: FAISS索引文件名
    :return: FAISS索引对象
    """
    return faiss.read_index(index_file)

# 使用FAISS索引进行图像检索
def search_similar_images(query_features, index, k=5):
    """
    从FAISS索引中检索与查询特征最相似的图像
    :param query_features: 查询图像的特征
    :param index: FAISS索引
    :param k: 返回最相似的k个图像
    :return: 最相似的k个图像索引
    """
    query_features = np.array(query_features).astype('float32').reshape(1, -1)  # 确保查询是二维的
    distances, indices = index.search(query_features, k)
    return indices[0], distances[0]  # 返回图像索引和相似度距离

