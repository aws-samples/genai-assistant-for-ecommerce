import time
import boto3
import json
import base64
from PIL import Image
import io

bedrock_client = boto3.client(service_name = 'bedrock-runtime',region_name = 'us-west-2')

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


def content_moderation_image(image_filename):

    imagedata, image_type =image_base64_encoder(image_filename)

    system_text='''
任务: 检测用户上传的图片是否对现有市场上的品牌、商标、版权作品等知识产权造成侵权。

输入:
用户上传的图片

输出:
{
"infringement": <布尔值,表示是否侵权>,
"confidence": <0到1之间的浮点数,表示置信度>,
"reason": "<判断侵权或不侵权的详细原因>",
"infringing_elements": [<侵权元素列表>],
"suggested_actions": "<建议采取的行动>"
}

例子输出:
{
"infringement": true,
"confidence": 0.95,
"reason": "上传的图片中包含了与知名品牌'XYZ'极为相似的logo设计,未经授权使用可能构成商标侵权。此外,图片背景使用了版权电影'ABC'的场景,未经许可可能涉及版权侵犯。",
"infringing_elements": ["XYZ品牌logo", "ABC电影场景"],
"suggested_actions": "建议移除或重新设计相似logo,并获取电影场景的使用授权。"
}

限制条件:

"infringement"字段必须是布尔值
"confidence"字段必须是0到1之间的浮点数,精确到小数点后两位
"reason"字段必须详细说明判断依据,包括可能侵权的具体元素及其对应的知识产权
"infringing_elements"字段列出所有可能侵权的元素
"suggested_actions"字段提供避免侵权的具体建议
分析步骤:

仔细观察图片中的所有视觉元素
识别任何与现有品牌、商标相似的标志、文字或设计
检测是否存在来自版权作品(如电影、音乐、艺术品等)的元素
评估这些元素的使用是否可能构成侵权
根据分析结果生成详细的JSON输出
注意事项:

考虑商标、版权、专利等各种知识产权
注意区分合理使用和侵权行为
保持客观,给出基于事实的分析
请根据上述要求,全面分析输入图片是否存在侵权风险,并生成详细的JSON格式输出。
'''
    system_prompts = [{"text" : system_text}]
    text='判断用户上传的图片是否侵权，使用JSON格式返回，不要做任何多余解释。'

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text":text,
                },
                {    "image": {
                        "format": image_type,
                        "source": {
                            "bytes": imagedata
                        }
                    }
                }
            ]
        }
    ]
    #Base inference parameters to use.
    inference_config = {"temperature": 0.1}
    # Additional inference parameters to use.
    additional_model_fields = {"top_k": 200}
    response = bedrock_client.converse(
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields,
    )

    return(response['output']['message']['content'])

#content_result=content_moderation_image(image_data,image_type)
#print(content_result)


def content_moderation_text(text):
    system_text='''
Your task is to Identify and classify any inappropriate content in user given text, Identify inappropriate content in the provided categories in the <Categories>.

</Categories>
hate,
hate/threatening,
self-harm,
sexual,
sexual/minors,
violence,
violence/graphic
</Categories>

<description>
hate: Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. Hateful content aimed at non-protected groups (e.g., chess players) is harassment.
hate/threatening: Hateful content that also includes violence or serious harm towards the targeted group based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.
harassment: Content that expresses, incites, or promotes harassing language towards any target.
harassment/threatening:	Harassment content that also includes violence or serious harm towards any target.
self-harm: Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.
self-harm/intent: Content where the speaker expresses that they are engaging or intend to engage in acts of self-harm, such as suicide, cutting, and eating disorders.
self-harm/instructions: Content that encourages performing acts of self-harm, such as suicide, cutting, and eating disorders, or that gives instructions or advice on how to commit such acts.
</description>

<ResponseFormat>
{
	"Moderation": false,
	"Category": "hate"
	"confidence_score": 1.0,
	"Reason":"the user content is harmful."
}
</ResponseFormat>
you should respond json only, no any other explanation.
'''
    system_prompts = [{"text" : system_text}]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text":text,
                }
            ]
        }
    ]
    #Base inference parameters to use.
    inference_config = {"temperature": 0.1}
    # Additional inference parameters to use.
    additional_model_fields = {"top_k": 200}
    response = bedrock_client.converse(
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields,
    )

    return(response['output']['message']['content'])

text='I hate everyone.'


#content_result=content_moderation_text(text)
#print(content_result)