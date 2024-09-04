import os
import boto3
import json
from dotenv import load_dotenv
from botocore.exceptions import ClientError

import base64
import io
from PIL import Image

from langchain.agents import tool 
from langchain_community.chat_models import BedrockChat
from langchain.llms import Bedrock

from utils.amazon_scraper import get_product, get_reviews, get_bestsellers

# loading in variables from .env file
load_dotenv()

data_folder = os.getenv("data_folder")

# instantiating the Bedrock client, and passing in the CLI profile
# boto3.setup_default_session(profile_name=os.getenv("profile_name"))

bedrock = boto3.client('bedrock-runtime', 'us-east-1')

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
