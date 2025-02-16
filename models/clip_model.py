from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import numpy as np
from transformers import pipeline
from models.faiss_index import load_faiss_index, search_similar_images,create_faiss_index

# 加载 CLIP 模型和处理器
model = CLIPModel.from_pretrained("./clip-vit-base-patch32/")
processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32/")

# 加载 BLIP 模型和处理器
blip_processor = BlipProcessor.from_pretrained("./blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("./blip-image-captioning-base")

# 创建情感分析管道
sentiment_analyzer = pipeline("sentiment-analysis", model="./distilbert-base-uncased-finetuned-sst-2-english/")


# 情感分析函数
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    sentiment = result[0]['label']
    return sentiment

# 查询与文本相关的图像（使用FAISS索引）
def query_image_from_text(text):
    image_path = "./data/images"
    create_faiss_index(image_path)
    index_file="./faiss.index"
    index=load_faiss_index(index_file)
    inputs = processor(text=text, return_tensors="pt", padding=True)
    text_features = model.get_text_features(**inputs)
    text_features = text_features.detach().numpy().flatten()  # 转换为一维数组
    
    # 使用FAISS索引进行查询
    indices, distances = search_similar_images(text_features, index, k=1)  # 获取最相似的图像
    
    # 根据索引加载最相似的图像
    image_paths = [f"./data/images/{i+1}.jpg" for i in range(len(indices))]
    best_image_path = image_paths[indices[0]]
    best_image = Image.open(best_image_path)
    
    return best_image

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

# 图像问答功能（BLIP）
def answer_question_with_image(image: Image, question: str) -> str:
    """
    使用BLIP模型生成图像和问题的答案
    :param image: 上传的图像
    :param question: 用户输入的问题
    :return: 问题的答案
    """
    # 处理图像
    inputs = blip_processor(images=image, return_tensors="pt")

    # 使用BLIP模型生成答案
    outputs = blip_model.generate(
        **inputs,
        max_new_tokens=700,  # 显式指定新生成token数
        do_sample=True,
        temperature=0.9,    # 提高随机性
        top_p=0.95,         # 核采样过滤低概率选项
        top_k=700,           # 限制候选词数量
        repetition_penalty=1.2,  # 抑制重复
    )

    # 获取生成的答案
    answer = blip_processor.decode(outputs[0], skip_special_tokens=True)
    
    return answer
