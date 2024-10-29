import json
import logging
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.invoice_extract import InvoiceExtractor

logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI 发票信息提取", page_icon="🧾", layout="wide")


def load_prompts():
    filename = f'./data/invoice/prompts.txt'
    try:
        with open(filename, 'r', encoding='utf-8', newline="\r\n") as file:
            prompts = file.read()
            return prompts
    except FileNotFoundError:
        st.error(f"Prompts_txt file not found.")
        return None


def main():
    # load environment variables
    load_dotenv()

    # 主标题
    st.title("AI 发票信息提取 🧾️")
    st.markdown("使用Amazon Bedrock中的LLM提取发票信息，提升税务处理效率🥰")

    with st.container():
        # header that is shown on the web UI
        # 添加一些说明信息
        st.markdown("---")
        st.markdown("📌 支持的发票文件格式: PDF,WEBP,PNG, JPG, JPEG")
        st.markdown("📌 图片大小限制: 最大 5MB")

        prompts = load_prompts()
        expander = st.expander('prompts 详细信息')
        expander.text_area('', value=prompts, height=300)

        File = st.file_uploader(label='发票文件',
                                type=["pdf", "webp", "png", "jpg", "jpeg"],
                                accept_multiple_files=False,
                                key="new")

        result = st.button("提交", key="invoice_submit")
        if result:
            if File is not None:
                save_folder = os.getenv("save_folder")
                print(save_folder)
                print('filename:' + File.name)
                save_path = Path(save_folder, File.name)
                with open(save_path, mode='wb') as w:
                    w.write(File.getvalue())

                if save_path.exists():
                    file_name = str(save_path)
                    print(file_name)
                    output = InvoiceExtractor(file_name).extract()
                    data = json.loads(output)
                    st.write(data)


if __name__ == '__main__':
    main()
