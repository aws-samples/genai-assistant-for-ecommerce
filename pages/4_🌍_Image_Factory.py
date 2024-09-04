import base64
import string
import streamlit as st
from pathlib import Path
import os
import json
from dotenv import load_dotenv
from utils.image_generation import generate_prompt_from_image
from utils.image_generation import generate_or_vary_image
from PIL import Image

import logging
logger = logging.getLogger(__name__)
STYLES = ["anime", "pixel-art"]
#more options: "anime", "analog-film" , "cinematic", "comic-book", "digital-art","enhance", "fantasy-art", "isometric", "line-art", "low-poly", "modeling-compound","neon-punk", "origami", "photographic", "pixel-art", "tile-texture","3d-model"

st.set_page_config(page_title="AI 图像工厂", page_icon="🎨", layout="wide")
def main():
    # load environment variables
    load_dotenv()
    
    # 主标题
    st.title("AI 图像工厂 🖼️")
    st.markdown("将GenAI的能力应用到电商图片制作中，激发创意，提升效率！")

    image_gen, image_variation_sdxl, image_variation_titan, image_background_removal = st.tabs(['Image Generation', 'Image Variation(sdxl)', 'Image Variation(titan)', 'Background Removal'])

    with image_gen:
        st.title("根据文字描述生成图片")
    
        # 文本输入区
        text_state = st.session_state.get('text', '')
        text = st.text_area("请输入图片描述", text_state, height=100)
    
        # 模型选择
        model_options = ["stability.stable-diffusion-xl-v1", "amazon.titan-image-generator-v2:0"]
        selected_model = st.selectbox("选择模型", model_options)
    
        # 生成按钮
        result = st.button("生成图片", key="text_submit")
    
        st.info("👆 在上方输入描述并选择模型，然后点击生成按钮")
    
        # 处理图片生成
        if result:
            if text:
                with st.spinner('正在生成图片...'):
                    status, image_result = generate_or_vary_image(model_id=selected_model, positive_prompt=text, task_type='image generation')
                    if status == 0:
                        st.success("图片生成成功!")
                        display_and_resize_image(image_result, target_size=768)
                        
                        # 下载按钮
                        st.download_button(
                            label="下载图片",
                            data=image_result,
                            file_name="generated_image.png",
                            mime="image/png"
                        )
                    else:
                        st.error(f'遇到执行错误: {image_result}')
            else:
                st.warning("请输入图片描述!")
    
    
    with image_variation_sdxl:
        st.title("图像变体生成")
        st.subheader("上传原图，选择变体的风格")
        model_id='stability.stable-diffusion-xl-v1'
        # 初始化 session state
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None
        if 'selected_style' not in st.session_state:
            st.session_state.selected_style = STYLES[0]
        if 'pre_prompts' not in st.session_state:
            st.session_state.pre_prompts = ""
        if 'generated_image' not in st.session_state:
            st.session_state.generated_image = None
        
        def process_uploaded_image():
            File = st.session_state.uploaded_file
            save_folder = os.getenv("save_folder")
            save_path = Path(save_folder, File.name)
            
            with open(save_path, mode='wb') as w:
                w.write(File.getvalue())
    
            if save_path.exists():
                file_name = save_path
                
                # 定义回调函数
                def on_style_change():
                    new_style = st.session_state.style_selector
                    with st.spinner('正在生成提示词...'):
                        st.session_state.pre_prompts = generate_prompt_from_image(file_name, style=new_style)
    
                # 风格选择
                selected_style = st.selectbox(
                    "选择风格", 
                    STYLES, 
                    index=STYLES.index(st.session_state.selected_style),
                    on_change=on_style_change,
                    key="style_selector"
                )
                st.session_state.selected_style = selected_style
    
                # 生成并编辑提示词
                if 'pre_prompts' not in st.session_state or st.session_state.pre_prompts == "":
                    with st.spinner('正在生成提示词...'):
                        st.session_state.pre_prompts = generate_prompt_from_image(file_name, style=selected_style)
                
                pre_prompts = st.text_area("提示词,可自由编辑:", value=st.session_state.pre_prompts, key="prompt_area_sdxl")
                st.session_state.pre_prompts = pre_prompts
    
                # 生成新图像按钮
                if st.button('生成新图片'):
                    with st.spinner('正在生成新图片...'):
                        status, result = generate_or_vary_image(model_id=model_id, positive_prompt=pre_prompts,   style_preset=selected_style, source_image=file_name)
                    if status == 0:
                        st.session_state.generated_image = result
                        st.success('新图片生成成功！')
                    else:
                        st.error(f'遇到执行错误: {result}')
    
                # 创建两列布局
                col1, col2 = st.columns(2)
    
                # 在左列显示原始图片
                with col1:
                    st.subheader("原始图片")
                    display_and_resize_image(file_name)
    
                # 在右列显示生成的变体图片
                with col2:
                    st.subheader("变体图片")
                    if st.session_state.generated_image:
                        display_and_resize_image(st.session_state.generated_image)
                    else:
                        st.info("生成的变体图片将显示在这里")
    
        uploaded_file = st.file_uploader('选择你的原始图片', type=["webp", "png", "jpg", "jpeg"], key="variation_img")
        
        # 检查是否有新文件上传
        if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            # 清除之前的 style 和 prompt
            st.session_state.selected_style = STYLES[0]
            st.session_state.pre_prompts = ""
            st.session_state.generated_image = None
            process_uploaded_image()
        elif uploaded_file is not None:
            process_uploaded_image()
 
    with image_variation_titan:
        st.title("图像变体生成")
        st.subheader("上传原图，输入提示词")
        model_id='amazon.titan-image-generator-v2:0'
        # 初始化 session state
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None
        if 'prompt' not in st.session_state:
            st.session_state.prompt = ""
        if 'generated_image' not in st.session_state:
            st.session_state.generated_image = None
        if 'task_type' not in st.session_state:
            st.session_state.task_type = "image conditioning"
        
        def process_uploaded_image_titan():
            File = st.session_state.uploaded_file
            save_folder = os.getenv("save_folder")
            save_path = Path(save_folder, File.name)
            
            with open(save_path, mode='wb') as w:
                w.write(File.getvalue())
    
            if save_path.exists():
                file_name = save_path
                
                # 任务类型选择
                task_type = st.selectbox(
                    "选择任务类型", 
                    ["image conditioning", "color guided content"],
                    index=["image conditioning", "color guided content"].index(st.session_state.task_type),
                    key="task_type_selector"
                )
                st.session_state.task_type = task_type
    
                # 提示词输入
                prompt = st.text_area("输入提示词:", value=st.session_state.prompt, key="prompt_area")
                st.session_state.prompt = prompt
    
                # 生成新图像按钮
                if st.button('生成新图片', key='titan_generating'):
                    with st.spinner('正在生成新图片...'):
                        status, result = generate_or_vary_image(
                            model_id=model_id, 
                            positive_prompt=prompt,   
                            source_image=file_name,
                            task_type=task_type
                        )
                    if status == 0:
                        st.session_state.generated_image = result
                        st.success('新图片生成成功！')
                    else:
                        st.error(f'遇到执行错误: {result}')
    
                # 创建两列布局
                col1, col2 = st.columns(2)
    
                # 在左列显示原始图片
                with col1:
                    st.subheader("原始图片")
                    display_and_resize_image(file_name,512)
    
                # 在右列显示生成的变体图片
                with col2:
                    st.subheader("变体图片")
                    if st.session_state.generated_image:
                        display_and_resize_image(st.session_state.generated_image, 512)
                    else:
                        st.info("生成的变体图片将显示在这里")
    
        uploaded_file = st.file_uploader('选择你的原始图片', type=["webp", "png", "jpg", "jpeg"], key="variation_img_titan")
        
        # 检查是否有新文件上传
        if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            # 清除之前的 prompt 和生成的图像
            st.session_state.prompt = ""
            st.session_state.generated_image = None
            process_uploaded_image_titan()
        elif uploaded_file is not None:
            process_uploaded_image_titan()
        

    with image_background_removal:
        st.title("图片背景移除 🖼️✂️")
        st.markdown("上传图片，我们将自动移除背景!")
        model_id='amazon.titan-image-generator-v2:0'
    
        # 文件上传器
        file = st.file_uploader('选择要处理的图片', type=["webp", "png", "jpg", "jpeg"], key="background_removal_img")
        
        # 提交按钮
        result = st.button("移除背景", key="submit_image_for_background_removal")
    
        if result:
            if file is not None:
                with st.spinner('正在处理图片...'):
                    save_folder = os.getenv("save_folder")
                    save_path = Path(save_folder, file.name)
                    
                    # 保存上传的文件
                    with open(save_path, mode='wb') as w:
                        w.write(file.getvalue())
    
                    if save_path.exists():
                        # 处理图片
                        #new_image = background_removal_titan(save_path)
                        status, result = generate_or_vary_image(
                            model_id=model_id, 
                            source_image=save_path,
                            task_type="background removal"
                        )
                        if status == 0:
                            st.session_state.generated_image = result
                        else:
                            st.error(f'遇到执行错误: {result}')
                        
                        
                        
                        # 创建两列布局来并排显示图片
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("原始图片")
                            st.image(file, caption='原始图片', use_column_width=True)
                        
                        with col2:
                            st.subheader("背景移除后")
                            st.image(result, caption='原始图片', use_column_width=True)
            else:
                st.warning('请上传图片!')
    
        # 添加一些说明信息
        st.markdown("---")
        st.markdown("📌 支持的图片格式: WEBP, PNG, JPG, JPEG")
        st.markdown("📌 图片大小限制: 最大 5MB")
    
    # 页脚
    st.markdown("---")
    st.markdown("由 AI 驱动 | 创建于 2024")


def display_and_resize_image(file_name, target_size=512):
    """
    打开图片文件，根据需要调整大小并显示。

    参数:
    file_name (str): 图片文件的路径
    target_size (int): 目标尺寸的宽度（默认为256像素）

    返回:
    None
    """
    try:
        # 打开图片
        img = Image.open(file_name)
        width, height = img.size

        # 调整图片大小（如果需要）
        if width > target_size:
            # 计算等比例缩放的高度
            new_height = int(height * (target_size / width))
            img = img.resize((target_size, new_height))
            st.image(img, caption='Image', width=target_size)
        else:
            st.image(img, caption='Image', use_column_width=True)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")


if __name__ == '__main__':
    main()
