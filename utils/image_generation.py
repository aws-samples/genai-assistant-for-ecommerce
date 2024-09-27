import base64
import io
import os
import json
import logging
import boto3
from PIL import Image
import time
from enum import Enum, unique
from botocore.exceptions import ClientError


class ImageError(Exception):
    """
    Custom exception for errors returned bedrock model.
    """

    def __init__(self, message):
        self.message = message


# Set up logging for notebook environment
logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.handlers.clear()
handler = logging.StreamHandler()
logger.addHandler(handler)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)

def generate_image_request(model_id, body):
    """
    Generate an image using bedrock invoke model.
    """
    logger.info(f"Generating image with model {model_id}")
    
    bedrock = boto3.client(service_name="bedrock-runtime", region_name='us-west-2')
    response = bedrock.invoke_model(body=body, modelId=model_id, accept="application/json", contentType="application/json")
    response_body = json.loads(response.get("body").read().decode("utf-8"))
    
    if model_id.startswith('stability'):
        finish_reasons = response_body.get('finish_reasons')
        logger.info(f"finish reasons is: {finish_reasons}")
        seeds = response_body.get('seeds')
        images = response_body.get('images')[0]
        # 检查是否有错误
        if finish_reasons and any(reason is not None for reason in finish_reasons):
            raise ImageError(f"Image generation error. Error code is {finish_reasons}")
        image_bytes = base64.b64decode(images)
    else:  # Titan model
        base64_image = response_body.get("images")[0]
        if response_body.get("error"):
            raise ImageError(f"Image generation error. Error is {response_body.get('error')}")
        image_bytes = base64.b64decode(base64_image.encode("ascii"))
        body_dict = json.loads(body)
        seeds = body_dict.get("imageGenerationConfig", {}).get("seed", 0)
    
    logger.info(f"Successfully generated image with model {model_id}")
    
    return seeds, image_bytes


def generate_or_vary_image(model_id, positive_prompt=None, negative_prompt='low quality', source_image=None, **kwargs):
    """
    Generate a new image from text or vary an existing image.
    
    Args:
        model_id (str): The ID of the model to use.
        positive_prompt (str): The positive prompt for image generation.
        negative_prompt (str): The negative prompt for image generation.
        source_image (str, optional): The path to the source image for variation.
        **kwargs: Additional parameters for customization.
    
    Returns:
        tuple: A tuple containing status code (0 for success, 1 for failure) and result (file path or error message).
    """
    try:
        if model_id.startswith('stability'):
            request_data = {
                "prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "mode": kwargs.get('mode', "text-to-image") ,
                "aspect_ratio": kwargs.get('aspect_ratio', "1:1") ,
                "seed": kwargs.get('seed', 0),
                "output_format": kwargs.get('output_format', 'png'),
            }
            if source_image and model_id == "stability.sd3-large-v1:0":
                # Image variation mode
                request_data.pop('aspect_ratio', None)
                request_data.update({
                    "mode": "image-to-image",
                    "strength": kwargs.get('strength', 1),
                })
                
                with open(source_image, "rb") as image_file:
                    request_data["image"] = base64.b64encode(image_file.read()).decode("utf-8")

            body = json.dumps(request_data)
            
        elif model_id == 'amazon.titan-image-generator-v2:0':
            print(f"task_type is {kwargs.get('task_type')}")
            print(f"prompt is: {positive_prompt}")
            print(f"color_list is: {kwargs.get('color_list')}")
            if kwargs.get('task_type') == "image generation":
                body = json.dumps({
                    "taskType": "TEXT_IMAGE",
                    "textToImageParams": {
                        "text": positive_prompt,
                        "negativeText": negative_prompt
                    },
                    "imageGenerationConfig": {
                        "numberOfImages": kwargs.get('numberOfImages', 1),
                        "height": kwargs.get('height', 1024),
                        "width": kwargs.get('width', 1024),
                        "cfgScale": kwargs.get('cfgScale', 8.0),
                        "seed": kwargs.get('seed', 0)
                    }
                })
            elif kwargs.get('task_type') == "color_guided_titan":
                print("start color guided titan11")
                print(kwargs.get('color_list'))
                color_list = kwargs.get('color_list')
                request_data = {
                    "taskType": "COLOR_GUIDED_GENERATION",
                    "colorGuidedGenerationParams": {
                        "text": positive_prompt, # sample: a jar of salad dressing in a rustic kitchen surrounded by fresh vegetables with studio lighting
                        "negativeText": negative_prompt,
                        "colors": color_list # '#ff8080', '#ffb280', '#ffe680', '#e5ff80'
                    },
                    "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "height": 512,
                    "width": 512,
                    "cfgScale": 8.0
                    }
                }
                if source_image:
                    input_image=load_and_resize_image(source_image)
                    request_data["colorGuidedGenerationParams"]["referenceImage"] = input_image
                body = json.dumps(request_data)

            elif kwargs.get('task_type') == "background removal":
                input_image=load_and_resize_image(source_image)
                body = json.dumps({
                    "taskType": "BACKGROUND_REMOVAL",
                    "backgroundRemovalParams": {
                    "image": input_image,
                    }
                })
            else:
                return 1, "parameters error, please check again!"

        else:
            raise ValueError(f"Unsupported model_id: {model_id}")

        seeds, image_bytes = generate_image_request(model_id=model_id, body=body)
        image = Image.open(io.BytesIO(image_bytes))
        
        if source_image:
            prefix = "variation"
        else:
            prefix = "text2image"
        
        file_path = save_image(image, prefix)
        return (0, file_path) if file_path else (1, "Failed to save image")
    
    except (ClientError, ImageError, ValueError) as err:
        logger.error(f"Error occurred: {str(err)}")
        return 1, f"{type(err).__name__}: {str(err)}"
    
    except Exception as err:
        logger.error(f"An unexpected error occurred: {str(err)}")
        return 1, f"Unexpected error: {str(err)}"

def load_and_resize_image(image_path, max_size=1408):
    with Image.open(image_path) as img:
        # 如果图片的宽度或高度超过max_size，就进行缩放
        if img.width > max_size or img.height > max_size:
            # 计算缩放比例
            scale = max_size / max(img.width, img.height)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        # 将图片转换为PNG格式的字节流
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")

        # 进行base64编码
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return encoded_image

def save_image(image, prefix="generated_image"):
    """
    保存图像到指定文件夹。

    参数:
    image (PIL.Image): 要保存的图像
    prefix (str): 文件名前缀

    返回:
    str: 保存的文件路径，如果保存失败则返回 None
    """
    try:
        # 从环境变量获取保存文件夹路径
        save_folder = os.getenv("save_folder", "generated_images")
        
        # 确保保存目录存在
        os.makedirs(save_folder, exist_ok=True)

        # 生成唯一的文件名
        epoch_time = int(time.time())
        file_name = f"{prefix}_{epoch_time}.png"
        file_path = os.path.join(save_folder, file_name)

        # 保存图像
        image.save(file_path)
        logger.info(f"Image saved as {file_path}")

        return file_path

    except Exception as err:
        logger.error(f"Failed to save image: {str(err)}")
        return None
