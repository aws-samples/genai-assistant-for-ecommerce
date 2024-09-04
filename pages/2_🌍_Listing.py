import base64
import string
import streamlit as st
from pathlib import Path
import os
import json
from dotenv import load_dotenv
from utils.listing_voc_prompt import gen_listing_prompt, bedrock_converse_api, bedrock_converse_api_with_image
from utils.listing_voc_agents import create_listing

from PIL import Image

import xml.etree.ElementTree as ET
import logging
logger = logging.getLogger(__name__)


model_Id_multi_modal = 'anthropic.claude-3-sonnet-20240229-v1:0'
                    #'meta.llama3-70b-instruct-v1:0' 
                    #'anthropic.claude-3-5-sonnet-20240620-v1:0' 
                    #'anthropic.claude-3-sonnet-20240229-v1:0'
model_Id = 'meta.llama3-70b-instruct-v1:0' 

def main():
    # load environment variables
    load_dotenv()
    
    st.set_page_config(page_title="AI Listing", )
    language_options = ['English', 'Chinese']
    language_lable = st.sidebar.selectbox('Select Language', language_options)
    
    mode_lable = 'PE'
    # default listing container that houses the image upload field
    with st.container():
        # header that is shown on the web UI
        st.subheader('Listing写作')

        File = st.file_uploader('商品图片', type=["webp", "png", "jpg", "jpeg"], key="new")
        brand = st.text_input("品牌", '')
        features = st.text_input("商品关键词", '')

        st.divider()

        asin_label = ['B0BZYCJK89', 'B0BGYWPWNC', 'B0CX23V2ZK']
        asin = st.selectbox('请选择参考的热卖商品', asin_label)

        filename = './data/' + 'asin_' + asin + '_product.json'
        with open(filename, 'r', encoding='utf-8') as file:
            product_data = file.read()
            
        product_results = json.loads(product_data)
        
        as_title = product_results['results'][0]['content']['title']
        as_bullet = product_results['results'][0]['content']['bullet_points']
        as_des = product_results['results'][0]['content']['description']

        expander = st.expander('详细信息')
        expander.write('Title:')
        expander.write(as_title)

        expander.write('Bullet Points:')
        expander.write(as_bullet)

        expander.write('Description:')
        expander.write(as_des)

        result = st.button("点击生成商品Listing")

        # if the button is pressed, the model is invoked, and the results are output to the front end
        if result:
            # if an image is uploaded, a file will be present, triggering the image_to_text function
            if File is not None:

                save_folder = os.getenv("save_folder")
                print(save_folder)
                print('filename:' + File.name)

                # create a posix path of save_folder and the file name
                save_path = Path(save_folder, File.name)
                # write the uploaded image file to the save_folder you specified
                with open(save_path, mode='wb') as w:
                    w.write(File.getvalue())

                # once the save path exists...
                if save_path.exists():

                    file_name = save_path

                    if mode_lable == 'PE':
                        user_prompt = gen_listing_prompt(asin, 'com', brand, features, language_lable)
                        print('user_prompt:' + user_prompt)
                        
                        llm_output = bedrock_converse_api_with_image(model_Id_multi_modal, file_name, user_prompt)
                        #st.write(output)
                    elif mode_lable == 'Agent':
                        response = create_listing(asin, file_name, brand, features)
                        print(response)
                        rslist = str(response['output']).rsplit('>')
                        output = rslist[-1]

                    print("output:" + llm_output)

                    title, bullets, description = parse_listing_xml_response(llm_output)

                    st.write("Title:\n")
                    st.write(title)

                    st.write("Bullet Points:\n")
                    st.write(bullets)

                    st.write("Description:\n")
                    st.write(description)
    
                    # removing the image file that was temporarily saved to perform the question and answer task
                    os.remove(save_path)
            else:
                if mode_lable == 'PE':
                    user_prompt = gen_listing_prompt(asin, 'com', brand, features, language_lable)
                    print('user_prompt:' + user_prompt)
                        
                    llm_output = bedrock_converse_api(model_Id, user_prompt)
                    print(llm_output)

                    title, bullets, description = parse_listing_xml_response(llm_output)

                    st.write("Title:\n")
                    st.write(title)

                    st.write("Bullet Points:\n")
                    st.write(bullets)

                    st.write("Description:\n")
                    st.write(description)
    
def parse_listing_xml_response(xml_string):
    try:
        # 将XML字符串包装在根元素中
        wrapped_xml = f"<root>{xml_string}</root>"
        
        # 解析XML
        root = ET.fromstring(wrapped_xml)
        
        # 提取title和bullets
        title = root.find('title').text.strip() if root.find('title') is not None else ""
        bullets = root.find('bullets').text.strip() if root.find('bullets') is not None else ""
        description = root.find('description').text.strip() if root.find('description') is not None else ""
        
        return title, bullets, description
    except ET.ParseError:
        print("XML解析错误")
        return None
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None


if __name__ == '__main__':
    main()
