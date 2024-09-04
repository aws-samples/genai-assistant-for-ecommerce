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
    response_body = json.loads(response.get("body").read())
    
    if model_id.startswith('stability'):
        base64_image = response_body.get("artifacts")[0].get("base64")
        finish_reason = response_body.get("artifacts")[0].get("finishReason")
        if finish_reason in ["ERROR", "CONTENT_FILTERED"]:
            raise ImageError(f"Image generation error. Error code is {finish_reason}")
    else:  # Titan model
        base64_image = response_body.get("images")[0]
        if response_body.get("error"):
            raise ImageError(f"Image generation error. Error is {response_body.get('error')}")

    image_bytes = base64.b64decode(base64_image.encode("ascii"))
    logger.info(f"Successfully generated image with model {model_id}")
    
    return image_bytes


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
        if model_id == 'stability.stable-diffusion-xl-v1':
            request_data = {
                "text_prompts": [
                    {"text": positive_prompt, "weight": 1},
                    {"text": negative_prompt, "weight": -1},
                ],
                "height": kwargs.get('height', 1024),
                "width": kwargs.get('width', 1024),
                "cfg_scale": kwargs.get('cfg_scale', 12),
                "clip_guidance_preset": kwargs.get('clip_guidance_preset', "NONE"),
                "sampler": kwargs.get('sampler', "K_DPMPP_2M"),
                "samples": kwargs.get('samples', 1),
                "seed": kwargs.get('seed', 123456),
                "steps": kwargs.get('steps', 25),
            }
            
            if source_image:
                # Image variation mode
                request_data.update({
                    "init_image_mode": "IMAGE_STRENGTH",
                    "image_strength": kwargs.get('image_strength', 0.5),
                    "cfg_scale": kwargs.get('cfg_scale', 7),
                    "clip_guidance_preset": kwargs.get('clip_guidance_preset', "SLOWER"),
                    "steps": kwargs.get('steps', 30),
                })
                
                with Image.open(source_image) as image:
                    original_width, original_height = image.size
                    resized_image = image.resize((request_data["width"], request_data["height"]))
                    buffered = io.BytesIO()
                    resized_image.save(buffered, format="PNG")
                    request_data["init_image"] = base64.b64encode(buffered.getvalue()).decode("utf-8")
            else:
                # Text to image mode
                request_data["style_preset"] = kwargs.get('style_preset', "anime")
            
            body = json.dumps(request_data)
            
        elif model_id == 'amazon.titan-image-generator-v2:0':
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
            if kwargs.get('task_type') == "image conditioning":
                input_image=load_and_resize_image(source_image)
                body = json.dumps({
                    "taskType": "TEXT_IMAGE",
                    "textToImageParams": {
                        "text": positive_prompt, # sample: a cartoon deer in a fair world
                        "negativeText": negative_prompt,
                        "conditionImage": input_image,
                        "controlMode": "CANNY_EDGE", # Optional: CANNY_EDGE | SEGMENTATION
                        "controlStrength": 0.7
                    },
                    "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "height": 512,
                    "width": 512,
                    "cfgScale": 8.0
                    }
                })
            if kwargs.get('task_type') == "color guided content":
                input_image=load_and_resize_image(source_image)
                body = json.dumps({
                    "taskType": "COLOR_GUIDED_GENERATION",
                    "colorGuidedGenerationParams": {
                        "text": positive_prompt, # sample: a jar of salad dressing in a rustic kitchen surrounded by fresh vegetables with studio lighting
                        "negativeText": negative_prompt,
                        "referenceImage": input_image,
                        "colors": ['#ff8080', '#ffb280', '#ffe680', '#e5ff80'] # '#ff8080', '#ffb280', '#ffe680', '#e5ff80'
                    },
                    "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "height": 512,
                    "width": 512,
                    "cfgScale": 8.0
                    }
                })
            if kwargs.get('task_type') == "background removal":
                input_image=load_and_resize_image(source_image)
                body = json.dumps({
                    "taskType": "BACKGROUND_REMOVAL",
                    "backgroundRemovalParams": {
                    "image": input_image,
                    }
                })
            else:
                return 1, "parameters error, please check again"

        else:
            raise ValueError(f"Unsupported model_id: {model_id}")

        image_bytes = generate_image_request(model_id=model_id, body=body)
        image = Image.open(io.BytesIO(image_bytes))
        
        if source_image:
            if model_id == 'stability.stable-diffusion-xl-v1':
                image = image.resize((original_width, original_height))
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
        
def generate_prompt_from_image(source_image, style):
    user_text = f'''I'm using Stable Diffusion XL to generate variant images in different artistic styles. I'll provide an image for you to analyze. Based on that analysis, please generate a detailed text prompt in the {style} style. The prompt should:

    1. Accurately describe the key elements, subjects, and composition of the original image.
    2. Incorporate specific characteristics, techniques, and visual elements associated with the {style} style.
    3. Include relevant details about color palette, lighting, texture, and mood that would be appropriate for the chosen style.
    4. Be formatted in a way that's optimized for Stable Diffusion XL, using any relevant prompt engineering techniques.

    please give me text prompt only and no need any notes and explanation.
    '''
    max_size=1568
    # source_image is a file name
    with open(source_image, "rb") as f:
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

            img = img.resize((new_width, new_height), Image.LANCZOS)

        resized_bytes = io.BytesIO()
        img.save(resized_bytes, format=img_format)
        resized_bytes = resized_bytes.getvalue()

    bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')
    response = bedrock_client.converse(
        modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
        messages=[{"role": "user", "content": [{"text": user_text, }, {"image": {"format": img_format, "source": {"bytes": resized_bytes}}}]}],
        inferenceConfig={"temperature": 0.1},
        additionalModelRequestFields={"top_k": 200}
    )

    return response['output']['message']['content'][0]['text']
