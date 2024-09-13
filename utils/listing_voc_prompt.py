import os
import boto3
import json
from dotenv import load_dotenv
from botocore.exceptions import ClientError

import base64
import io
from PIL import Image


# loading in variables from .env file
load_dotenv()

data_folder = os.getenv("data_folder")

# instantiating the Bedrock client, and passing in the CLI profile
# boto3.setup_default_session(profile_name=os.getenv("profile_name"))

bedrock = boto3.client('bedrock-runtime', 'us-west-2')

def gen_listing_prompt(asin, domain, brand, features, language):
    # results = get_product(asin, domain)

    filename = './data/' + 'asin_' + asin + '_product.json'
    with open(filename, 'r', encoding='utf-8') as file:
        data = file.read()

    results = json.loads(data)

    as_title = results['results'][0]['content']['title']
    as_bullet = results['results'][0]['content']['bullet_points']
    as_des = results['results'][0]['content']['description']
    
    prompt_template = '''If you were an excellent Amazon product listing specialist.
    Your task is to create compelling and optimized product listings for Amazon based on the provided information.
    Please refer to the following examples and best seller products on Amazon to create a comprehensive product listing.
    
    Example of a good product listing on Amazon:

    <Example>
        <title>{title}</title>
        <bullets>{bullet}</title>
        <description>{des}</description>
    </Example>

    Please refer to the above image and the following production infomation fo to create product listing.

    Product Information:

    <product_information>
        <brand>{kw}</brand> 
        <keywords>{ft}</keywords> 
    </product_information>

    **Respond in valid XML format with the tags as "title", "bullets", "description"**. 
    Here is one sample:
        <title>{title}</title>
        <bullets>{bullet}</title>
        <description>{des}</description>

    please answer it in {lang}
    '''

    user_prompt = prompt_template.format(title=as_title, bullet=as_bullet, des=as_des, kw=brand, ft=features,lang=language)

    return user_prompt


def gen_voc_prompt(asin, domain, language):

    print('asin:' + asin, 'domain:' + domain)
    #results = get_reviews(asin, domain)

    filename = './data/' + 'asin_' + asin + '_reviews.json'
    with open(filename, 'r', encoding='utf-8') as file:
        reviews = file.read()

    results = json.loads(reviews)

    prompt_template = '''
    You are an analyst tasked with analyzing the provided customer review examples on an e-commerce platform and summarizing them into a comprehensive Voice of Customer (VoC) report. Your job is to carefully read through the product description and reviews, identify key areas of concern, praise, and dissatisfaction regarding the product. You will then synthesize these findings into a well-structured report that highlights the main points for the product team and management to consider.

    The report should include the following sections:
    Executive Summary - Briefly summarize the key findings and recommendations
    Positive Feedback - List the main aspects that customers praised about the product
    Areas for Improvement - Summarize the key areas of dissatisfaction and improvement needs raised by customers
    Differentiation from Competitors - Unique features or advantages that set a product apart from competitors
    Unperceived Product Features - Valuable product characteristics or benefits that customers are not fully aware of
    Core Factors for Repurchase and Recommendation - Critical elements that drive customers to repurchase and recommend a product
    Sentiment Analysis - Analyze the sentiment tendencies (positive, negative, neutral) in the reviews
    Topic Categorization - Categorize the review content by topics such as product quality, scent, effectiveness, etc.
    Recommendations - Based on the analysis, provide recommendations for product improvements and marketing strategies

    When writing the report, use concise and professional language, highlight key points, and provide reviews examples where relevant. Also, be mindful of protecting individual privacy by not disclosing any personally identifiable information.

    <Product descriptions>
    {product_description}
    <Product descriptions>

    <product reviews>
    {product_reviews}
    <product reviews>

    if output is not English, Please also ouput the reuslt in {lang}
    '''
    
    user_prompt  = prompt_template.format(product_description='', product_reviews=results['results'], lang=language)

    return user_prompt

def image_base64_encoder(image_path, max_size=1568):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    img_format = img.format.lower()

    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

    resized_bytes = io.BytesIO()
    img.save(resized_bytes, format=img_format)
    resized_bytes = resized_bytes.getvalue()

    return resized_bytes, img_format


def bedrock_converse_api(model_id, input_text):
    conversation = [
        {
            "role": "user",
            "content": [{"text": input_text}],
        }
    ]

    try:
        # Send the message to the model, using a basic inference configuration.
        response = bedrock.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
        )

        # Extract and print the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        #print(response_text)

        return response_text

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")


def bedrock_converse_api_with_image(model_id, image_filename, input_text):
    image_base64, file_type = image_base64_encoder(image_filename)
    conversation = [
        {
            "role": "user",
            "content": [
                {"text": input_text},
                {    "image": {
                        "format": file_type,
                        "source": {
                            "bytes": image_base64
                        }
                    }
                }
            ],
        }
    ]

    try:
        # Send the message to the model, using a basic inference configuration.
        response = bedrock.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
        )

        # Extract and print the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        #print(response_text)

        return response_text

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")

#购买动机

def gen_purchase_motivation_prompt(product_description=None, product_reviews=None):
    return f"""
    作为电商数据分析师，请分析以下产品评论中的购买动机。请按照以下步骤进行：

    1. 识别主要购买动机：找出5-8个最常见的购买原因。
    2. 量化分析：估算每个动机在评论中的提及比例。
    3. 动机解释：简要解释每个购买动机的背景或意义。
    4. 代表性引用：为每个动机选择一个简短但有代表性的评论。

    使用以下格式：

    购买动机：[动机名称]
    提及占比：[百分比]
    解释：[简要说明]
    评论引用："[引用内容]"

    <产品描述>
    {product_description}
    </产品描述>

    <产品评论>
    {product_reviews}
    </产品评论>

    请用简洁、专业的语言呈现分析。
    """

# 用户建议

def gen_user_suggestions_prompt(product_description=None, product_reviews=None):
    return f"""
    作为产品改进专家，请分析以下产品评论中的用户建议。请按照以下步骤进行：

    1. 识别主要建议：找出5-8个最常见的用户改进建议。
    2. 量化分析：估算每个建议在评论中的提及比例。
    3. 建议价值：评估每个建议的潜在影响和可行性。
    4. 代表性引用：为每个建议选择一个简短但有代表性的评论。

    使用以下格式：

    用户建议：[建议内容]
    提及占比：[百分比]
    潜在价值：[高/中/低] - [简要说明]
    评论引用："[引用内容]"

    <产品描述>
    {product_description}
    </产品描述>

    <产品评论>
    {product_reviews}
    </产品评论>

    请用简洁、专业的语言呈现分析。
    """

#负面观点

def gen_negative_opinions_prompt(product_description=None, product_reviews=None):
    return f"""
    作为客户满意度分析师，请分析以下产品评论中的负面观点。请按照以下步骤进行：

    1. 识别主要问题：找出5-8个最常见的负面评价点。
    2. 量化分析：估算每个问题在负面评论中的占比。
    3. 影响评估：评估每个问题对整体客户满意度的影响程度。
    4. 代表性引用：为每个问题选择一个简短但有代表性的评论。

    使用以下格式：

    负面观点：[问题描述]
    占比：[在负面评论中的百分比]
    影响程度：[高/中/低] - [简要说明]
    评论引用："[引用内容]"

    <产品描述>
    {product_description}
    </产品描述>

    <产品评论>
    {product_reviews}
    </产品评论>

    请用简洁、专业的语言呈现分析。
    """

# 产品体验

def gen_product_experience_prompt(product_description=None, product_reviews=None):
    return f"""
    作为用户体验专家，请分析以下产品评论中的产品体验。请按照以下步骤进行：

    1. 识别关键体验点：找出5-8个最常被提及的产品体验方面。
    2. 量化分析：估算每个体验点在评论中的提及比例。
    3. 体验评价：总结用户对每个体验点的整体评价（正面/中性/负面）。
    4. 代表性引用：为每个体验点选择一个简短但有代表性的评论。

    使用以下格式：

    体验方面：[体验点描述]
    提及占比：[百分比]
    整体评价：[正面/中性/负面] - [简要说明]
    评论引用："[引用内容]"

    <产品描述>
    {product_description}
    </产品描述>

    <产品评论>
    {product_reviews}
    </产品评论>

    请用简洁、专业的语言呈现分析。
    """

# 星级分布


def gen_star_rating_distribution_prompt(product_description=None, product_reviews=None):
    return f"""
    作为数据统计专家，请分析以下产品评论中的星级分布。请按照以下步骤进行：

    1. 统计分布：计算每个星级（1-5星）的评论数量和百分比。
    2. 趋势分析：识别星级分布的整体趋势和特点。
    3. 关键因素：对于每个星级，总结影响该评分的主要因素。
    4. 代表性引用：为每个星级选择一个简短但有代表性的评论。

    使用以下格式：

    星级：[1-5星]
    占比：[百分比]
    主要因素：[简要说明影响该评分的因素]
    评论引用："[引用内容]"

    总体趋势：[对星级分布的整体分析]

    <产品描述>
    {product_description}
    </产品描述>

    <产品评论>
    {product_reviews}
    </产品评论>

    请用简洁、专业的语言呈现分析。
    """


def gen_user_expectations_prompt(product_description=None, product_reviews=None):
    return f"""
    作为消费者洞察专家，请分析以下产品评论中的用户期望。请按照以下步骤进行：

    1. 识别主要期望：找出5-8个最常见的用户期望。
    2. 量化分析：估算每个期望在评论中的提及比例。
    3. 满足程度：评估产品对每个期望的满足程度。
    4. 代表性引用：为每个期望选择一个简短但有代表性的评论。

    使用以下格式：

    用户期望：[期望描述]
    提及占比：[百分比]
    满足程度：[高/中/低] - [简要说明]
    评论引用："[引用内容]"

    <产品描述>
    {product_description}
    </产品描述>

    <产品评论>
    {product_reviews}
    </产品评论>

    请用简洁、专业的语言呈现分析。
    """
