from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
from transformers import pipeline

# 加载 CLIP 模型和处理器
model = CLIPModel.from_pretrained("./clip-vit-base-patch32/")
processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32/")

# 创建情感分析管道
sentiment_analyzer = pipeline("sentiment-analysis",model="./distilbert-base-uncased-finetuned-sst-2-english/")

# 情感分析函数
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    sentiment = result[0]['label']
    return sentiment

# 查询与文本相关的图像
def query_image_from_text(text):
    # 对文本进行处理，生成特征向量
    inputs = processor(text=text, return_tensors="pt", padding=True)
    text_features = model.get_text_features(**inputs)
    
    # 假设我们有一组预存的图像和对应的文本描述
    images = [Image.open("./data/images/1.jpg"), Image.open("./data/images/2.jpg"), Image.open("./data/images/3.jpg")]
    texts = ["a photo of a cat", "a photo of a dog", "a picture of a sunset"]
    
    # 对所有图像生成特征向量
    inputs_images = processor(images=images, return_tensors="pt")
    image_features = model.get_image_features(**inputs_images)

    # 计算文本与图像特征之间的相似性（使用余弦相似度）
    text_features = text_features.detach().numpy()
    image_features = image_features.detach().numpy()

    similarities = np.dot(image_features, text_features.T)
    best_match_idx = np.argmax(similarities)  # 找到最相似的图像
    return images[best_match_idx]

# 获取 CLIP 的文本特征
def get_text_features(text):
    inputs = processor(text=text, return_tensors="pt", padding=True)
    text_features = model.get_text_features(**inputs)
    return text_features.detach().numpy()

# 获取 CLIP 的图像特征
def get_image_features(image):
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    return image_features.detach().numpy()
