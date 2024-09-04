import base64
import string
import streamlit as st
from pathlib import Path
import os
import json
from dotenv import load_dotenv
from utils.content_moderation import content_moderation_image
from utils.content_moderation import content_moderation_text
from PIL import Image

import logging
logger = logging.getLogger(__name__)

def main():
    # load environment variables
    load_dotenv()
    
    st.set_page_config(page_title="Content Moderation", )
    
    audit_image, audit_text = st.tabs(['Image Moderation', 'Text Moderation'])
    
    with audit_image:
        st.title("图像审核")
        st.subheader("上传图片，检测是否涉嫌侵权")
        File = st.file_uploader('请选择要审核的图片', type=["webp", "png", "jpg", "jpeg"], key="new")
        result = st.button("提交", key="image_submit")
        if result:
            if File is not None:
                save_folder = os.getenv("save_folder")
                print(save_folder)
                print('filename:' + File.name)

                save_path = Path(save_folder, File.name)
                with open(save_path, mode='wb') as w:
                    w.write(File.getvalue())

                if save_path.exists():
                    file_name = save_path
                    output = content_moderation_image(file_name)

                    # 2. 显示图片功能
                    st.subheader("上传的图片")
                    # 获取图片的宽度
                    img = Image.open(File)
                    width, height = img.size
                    
                    # 如果宽度超过 256 像素,则按比例缩小到 256 像素宽度
                    if width > 256:
                        st.image(File, caption='Uploaded Image', width=256)
                    else:
                        st.image(File, caption='Uploaded Image', use_column_width=True)

                    # 3. 显示图片审核结果功能
                    st.subheader("审核结果")
                    data = json.loads(output[0]['text'])
                    st.write('【侵权检测结果】', '是' if data['infringement'] else '否')
                    st.write('【置信度】', data['confidence'])
                    st.write('【说明】', data['reason'])
                    st.write(' 结构化输出')
                    st.write(data)

            else:
                st.write('请选择要审核的图片')
        
        with audit_text:
            st.title("文本审核")
            st.subheader("输入要审核的文本")
            text_state = st.session_state.get('text', '')
            
            text = st.text_area("请在此输入文本内容", text_state)
            result = st.button("提交",  key="text_submit")
            
            if result:
                if text:
                    output = content_moderation_text(text)

                    st.subheader("文本审核结果")
                    data = json.loads(output[0]['text'])
                
                    st.write('【违规检测】', '是' if data['Moderation'] else '否')
                    st.write('【违规内容】', data['Category'])
                    st.write('【置信度】', data['confidence_score'])
                    st.write('【说明】', data['Reason'])
                    st.write('结构化输出')
                    st.write(data)

                else:
                    st.write('请输入要审核的文本内容')

if __name__ == '__main__':
    main()
