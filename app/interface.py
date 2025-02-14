import gradio as gr
from models.clip_model import analyze_sentiment, query_image_from_text

# Gradio回调函数
def multimodal_query(input_text, input_sentence):
    sentiment = ""  # 初始化情感分析结果
    related_image = None  # 初始化相关图像

    # 如果输入的句子不为空，则进行情感分析
    if input_sentence:
        sentiment = analyze_sentiment(input_sentence)
    
    # 如果输入文本不为空，则查询与文本相关的图像
    if input_text:
        related_image = query_image_from_text(input_text)

    return sentiment, related_image


# 创建Gradio界面
def launch_interface():
    interface = gr.Interface(
        fn=multimodal_query,
        inputs=[
            gr.Textbox(label="输入文本以查询图像"),  # 输入查询图像的文本框
            gr.Textbox(label="输入句子进行情感分析")    # 输入句子进行情感分析的文本框
        ],
        outputs=[
            gr.Textbox(label="情感分析结果"),  # 输出情感分析结果
            gr.Image(type='pil', label="相关图像")  # 输出与输入文本相关的图像
        ]
    )
    interface.launch()

if __name__ == "__main__":
    launch_interface()
