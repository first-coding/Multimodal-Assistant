import gradio as gr
from models.clip_model import analyze_sentiment, query_image_from_text, answer_question_with_image
import numpy as np

# Gradio回调函数
def multimodal_query(input_text, input_sentence, input_image, input_question):
    sentiment = ""  # 初始化情感分析结果
    related_image = None  # 初始化相关图像
    answer = ""  # 初始化图像问答结果

    # 情感分析部分：如果输入句子不为空，则进行情感分析
    if input_sentence:
        sentiment = analyze_sentiment(input_sentence)
    
    # 查询与输入文本相关的图像
    if input_text:
        related_image = query_image_from_text(input_text)

    # 图像问答部分：如果用户上传图像和问题
    if input_image is not None and isinstance(input_image, np.ndarray) and input_image.size > 0 and input_question:
        answer = answer_question_with_image(input_image, input_question)
        
    return sentiment, related_image, answer


# 创建Gradio界面
def launch_interface():
    interface = gr.Interface(
        fn=multimodal_query,
        inputs=[
            gr.Textbox(label="输入文本以查询图像"),  # 输入查询图像的文本框
            gr.Textbox(label="输入句子进行情感分析"),  # 输入句子进行情感分析的文本框
            gr.Image(label="上传图像"),  # 上传图像框
            gr.Textbox(label="输入问题对图像进行问答")  # 输入问题框，进行图像问答
        ],
        outputs=[
            gr.Textbox(label="情感分析结果"),  # 输出情感分析结果
            gr.Image(type='pil', label="相关图像"),  # 输出与输入文本相关的图像
            gr.Textbox(label="图像问答结果")  # 输出图像问答的答案
        ]
    )
    interface.launch()


