from PIL import Image
import io
import boto3

def generate_prompt_from_image(source_image, positive_prompt=None):
    user_text = f'''Analyze the provided image and generate an optimized text prompt for Stable Diffusion image-to-image generation. Your response should:
1. Describe the image content:
   - Main subject(s) and their characteristics
   - Background and setting
   - Composition and framing
2. Specify visual elements:
   - Color palette and dominant colors
   - Lighting conditions and effects
   - Textures and materials
   - Style (e.g., photorealistic, painterly, cartoon)
3. Capture the mood and atmosphere
4. Incorporate artistic techniques or references if applicable
5. Use Stable Diffusion-specific formatting:
   - Separate elements with commas
   - Use () for emphasis and [] for de-emphasis
   - Include relevant artistic or technical terms

Initial prompt (if any):
{positive_prompt}

Based on this initial prompt and the image analysis, create an enhanced, comprehensive prompt that maintains the original intent while improving its effectiveness for Stable Diffusion.

Provide only the generated prompt, formatted for direct use in Stable Diffusion. Aim for 50-75 words. Do not include explanations or notes.

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
    model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
    response = bedrock_client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": user_text, }, {"image": {"format": img_format, "source": {"bytes": resized_bytes}}}]}],
        inferenceConfig={"temperature": 0.1},
        additionalModelRequestFields={"top_k": 200}
    )

    return response['output']['message']['content'][0]['text']


def generate_prompt_from_text(source_text):
    user_text = f'''Analyze the provided text and generate an optimized text prompt for Stable Diffusion text-to-image generation. Your response should:

1. Describe the image content:
   - Main subject(s) and their characteristics
   - Background and setting
   - Composition and framing
2. Specify visual elements:
   - Color palette and dominant colors
   - Lighting conditions and effects
   - Textures and materials
   - Style (e.g., photorealistic, painterly, cartoon)
3. Capture the mood and atmosphere
4. Incorporate artistic techniques or references if applicable
5. Use Stable Diffusion-specific formatting:
   - Separate elements with commas
   - Use () for emphasis and [] for de-emphasis
   - Include relevant artistic or technical terms

Provided text:
{source_text}

Based on this text analysis, create an enhanced, comprehensive prompt that maintains the original intent while improving its effectiveness for Stable Diffusion.

Provide only the generated prompt, formatted for direct use in Stable Diffusion. Aim for 50-75 words. Do not include explanations, notes, or variations.
'''
    bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')
    model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
    response = bedrock_client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": user_text, }, {"text": source_text}]}],
        inferenceConfig={"temperature": 0.1},
        additionalModelRequestFields={"top_k": 200}
    )

    return response['output']['message']['content'][0]['text']