import base64
import string
import streamlit as st
from pathlib import Path
import os
import json
from dotenv import load_dotenv
from utils.listing_voc_prompt import gen_listing_prompt, gen_voc_prompt, bedrock_converse_api
from utils.listing_voc_prompt import gen_purchase_motivation_prompt, gen_user_suggestions_prompt, gen_negative_opinions_prompt,gen_product_experience_prompt,gen_star_rating_distribution_prompt,gen_user_expectations_prompt
from utils.listing_voc_agents import create_listing

from PIL import Image

import logging
logger = logging.getLogger(__name__)

model_Id = 'anthropic.claude-3-sonnet-20240229-v1:0'

st.set_page_config(page_title="VoC客户之声", page_icon="🎨", layout="wide")

@st.cache_data
def load_reviews(asin):
    filename = f'./data/asin_{asin}_reviews.json'
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        st.error(f"Review file for ASIN {asin} not found.")
        return None
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in review file for ASIN {asin}.")
        return None

def main():
    language_options = ['English', 'Chinese']
    language_label = st.sidebar.selectbox('Select Language', language_options)

    st.title('VOC 客户之声')

    asin_label = ['B0BZYCJK89', 'B0BGYWPWNC', 'B0CX23V2ZK']
    asin = st.selectbox('请选择 Amazon ASIN', asin_label)

    reviews = load_reviews(asin)
    if reviews:
        with st.expander("用户评论信息"):
            st.json(reviews['results'][0]['content'])

    if st.button("点击生成报告"):
        with st.spinner('正在生成报告...'):
            domain = "com"
            user_prompt = gen_voc_prompt(asin, domain, language_label)
            
            # 创建两列布局
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("总结报告")
                output = bedrock_converse_api(model_Id, user_prompt)
                st.write(output)
            
            with col2:
                st.subheader("分类指标")
                analysis_aspect = ["购买动机", "用户建议", "负面观点", "产品体验", "星级分布", "用户期望"]
                prompt_generators = {
                    "购买动机": gen_purchase_motivation_prompt,
                    "用户建议": gen_user_suggestions_prompt,
                    "负面观点": gen_negative_opinions_prompt,
                    "产品体验": gen_product_experience_prompt,
                    "星级分布": gen_star_rating_distribution_prompt,
                    "用户期望": gen_user_expectations_prompt
                }

                for aspect in analysis_aspect:
                    with st.expander(aspect):
                        prompt = prompt_generators[aspect](reviews['results'])
                        voc_metrics = bedrock_converse_api(model_Id, prompt)
                        st.write(voc_metrics)

if __name__ == '__main__':
    main()
