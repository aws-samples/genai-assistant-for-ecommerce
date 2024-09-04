import base64
import string
import streamlit as st
from pathlib import Path
import os
import json
from dotenv import load_dotenv
from utils.listing_voc_prompt import gen_listing_prompt, gen_voc_prompt, bedrock_converse_api
from utils.listing_voc_agents import create_listing

from PIL import Image

import logging
logger = logging.getLogger(__name__)

model_Id = 'meta.llama3-70b-instruct-v1:0'
            #'meta.llama3-70b-instruct-v1:0' 
            #'anthropic.claude-3-5-sonnet-20240620-v1:0' 
            #'anthropic.claude-3-sonnet-20240229-v1:0'

st.set_page_config(page_title="VoC客户之声", page_icon="🎨", layout="wide")

def main():
    language_options = ['English', 'Chinese']
    language_lable = st.sidebar.selectbox('Select Language', language_options)

    with st.container():
        #asin = st.text_input("Amazon ASIN", 'B0BZYCJK89')
        st.subheader('VOC 客户之声')

        asin_label = ['B0BZYCJK89', 'B0BGYWPWNC', 'B0CX23V2ZK']
        asin = st.selectbox('请选择 Amazon ASIN', asin_label)

        reviews = ''
        filename = './data/' + 'asin_' + asin + '_reviews.json'
        with open(filename, 'r', encoding='utf-8') as file:
            reviews = file.read()

        st.text('用户评论信息')
        st.json(json.loads(reviews)['results'][0]['content'])

        result = st.button("点击生成报告")

        if result:
            domain = "com"
            user_prompt  = gen_voc_prompt(asin, domain, language_lable)

            # print("user_prompt:" + user_prompt)

            # output = text_to_text(system_prompt, user_prompt)

            output = bedrock_converse_api(model_Id, user_prompt)

            st.write(output)

if __name__ == '__main__':
    main()
