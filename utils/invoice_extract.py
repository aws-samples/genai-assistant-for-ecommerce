#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

ref:
    - https://aws.amazon.com/cn/blogs/china/evatmaster-achieves-accurate-invoice-recognition-based-on-claude3
    - https://docs.anthropic.com/en/docs/build-with-claude/vision
"""

import base64
import io
import json
from typing import Final

import boto3
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# https://docs.anthropic.com/en/docs/build-with-claude/vision#evaluate-image-size
# the max image height: 1568
IMAGE_MAX_HEIGHT: Final[int] = 1568
# the max image height: 1092
IMAGE_MAX_WIDTH: Final[int] = 1092


class ImageInvoiceExtractor:
    """
    Pdf Invoice Extractor
    """

    def __init__(self, file_path: str):
        self._file_path = file_path
        self._prompts_template: str = """
                        The unformated raw text in the image is pre parsed in messages's content,text start with 'Image N:',N represents the sequence number of images.
                        Please prioritize messages's content results when responding, and keep the exact spelling of words in uppercase and lowercase letters.
                        Please provide the following information based on the invoices provided, and output it in JSON array.
                        
                        <json_format>
                        "seller_company": I need the invoice seller's name,the seller's company name may be described as "seller", "sold from", "vendor", "supplier", "卖方", "乙方", "供应商". usually at the top center of the invoice,Prioritize company name in title or first line of the invoice,Prioritize original English Company Name,if only chinese company name, priority traditional chinese name;DO NOT translate name. 
                        "buyer_company": I need the invoice buyer's name,The buyer's name is generally marked with terms such as "Buyer", "to", "向", "甲方", "买方", "买家", "客户", "购买方", etc., representing the buyer.
                        "date": Creation Date, the value is in the YYYY-MM-DD format.
                        "invoice_number": invoice number, could also be described as purchase number, order number, serial number, if this are none of these, output an empty string.
                        "currency": currency abbreviation in ISO 4217 standard, like USD, EUR, CNY, CAD, HKD, JPY, AUD etc. DO NOT use other abbreviations.
                        "total_amount": the total amount of this order in float data type.
                        </json_format>
                            
                        below is output example,it a json array:
                        <json_example>
                        [
                            {
                              "seller_company": "seller company 1",
                              "buyer_company": "buyer company 1",
                              "date": "2024-08-30",
                              "invoice_number": "6123450",
                              "currency": "USD",
                              "total_amount": 88.88
                            },
                            {
                              "seller_company": "seller company 2",
                              "buyer_company": "buyer company 2",
                              "date": "2024-08-31",
                              "invoice_number": "6123451",
                              "currency": "CNY",
                              "total_amount": 88.89
                            }
                        ]
                        </json_example>
                        
                        Please note that the result does not need additional explanation fields,You only need output json array,do not output any conversation messages between you and me that are not relevant to the invoices content
                        Please note that the invoices I provide are 1 to multiple consecutive images, which are continuous.
                        Please note that output is a JSON array according to <json_format> structure,and it's must can be directly parsed as json array.
                        """

    def _pre_process_images(self, texts: list[str], images: list[str]) -> (list[str], list[str]):
        """
        preprocessing image for claude
            1. keep image size in proper size
            2. rotate image if necessary
            3. convert image to webp format
            4. convert image to base64 encoding

        ref: https://docs.anthropic.com/en/docs/build-with-claude/vision#evaluate-image-size

        :return: [list of pdf page text content,list of pdf page image base64 encoding]
        """

        images_base64: list[str] = []

        for i, image in enumerate(images):
            # detect orientation and rotate
            # https://notes-of-python.readthedocs.io/zh/latest/python/python-notes-for-ocr-with-tesseract/
            orientation = pytesseract.image_to_osd(
                np.array(image),
                output_type=pytesseract.Output.DICT
            )["orientation"]
            # rotate images
            if orientation != 0:
                image = Image.fromarray(np.rot90(np.array(image), k=orientation // 90))
                texts[i] = pytesseract.image_to_string(image, config="-l chi_sim+eng")

            # get image's width,height
            width, height = image.size
            # get image's max long edge
            max_size = max(width, height)
            # if image's long edge is larger than 1568, resize max long edge to 1568
            if max_size > IMAGE_MAX_WIDTH:
                # resize image's with
                width = round(width * IMAGE_MAX_WIDTH / max_size)
                # resize image's height
                height = round(height * IMAGE_MAX_WIDTH / max_size)
                # resize image
                image = image.resize((width, height))

            # reformat to webp
            buffer = io.BytesIO()
            image.save(buffer, format="webp", quality=85)
            image_data = buffer.getvalue()
            # base64 encode image
            images_base64.append(base64.b64encode(image_data).decode("utf-8"))

        return texts, images_base64

    def _pre_process(self) -> (list[str], list[str]):
        """
        preprocessing image for claude
            1. keep image size in proper size
            2. rotate image if necessary
            3. convert image to webp format
            4. convert image to base64 encoding

        ref: https://docs.anthropic.com/en/docs/build-with-claude/vision#evaluate-image-size

        :return: [list of pdf page text content,list of pdf page image base64 encoding]
        """

        images: list[Image] = [Image.open(self._file_path)]
        texts: list[str] = [pytesseract.image_to_string(image, config="-l chi_sim+eng") for image in images]

        return self._pre_process_images(texts, images)

    def extract(self) -> str:
        texts, images = self._pre_process()

        _content = []
        for i, val in enumerate(texts):
            _text = {
                "type": "text",
                "text": f"Image {i}: {val}"
            }
            _content.append(_text)

            _image = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/webp",
                    "data": images[i],
                },
            }
            _content.append(_image)

        _prompts = {
            "type": "text",
            "text": self._prompts_template,
        }
        _content.append(_prompts)

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "system": "You are a financial staff responsible for identifying and entering procurement invoices.",
            "messages": [
                {
                    "role": "user",
                    "content": _content,
                }
            ],
        }, ensure_ascii=False)

        bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId="anthropic.claude-3-haiku-20240307-v1:0"
        )
        result = json.loads(response.get('body').read())["content"][0]["text"]
        return result


class PdfInvoiceExtractor(ImageInvoiceExtractor):
    """
    Pdf Invoice Extractor
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)

    def _pre_process(self) -> (list[str], list[str]):
        """
        preprocessing image for claude
            1. keep image size in proper size
            2. rotate image if necessary
            3. convert image to webp format
            4. convert image to base64 encoding

        ref: https://docs.anthropic.com/en/docs/build-with-claude/vision#evaluate-image-size

        :return: [list of pdf page text content,list of pdf page image base64 encoding]
        """
        texts: list[str] = []
        with pdfplumber.open(self._file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                texts.append(text)

        # convert pdf to images
        images: list[Image] = convert_from_path(self._file_path)
        return self._pre_process_images(texts, images)


if __name__ == '__main__':
    pdf_extractor = PdfInvoiceExtractor("../data/invoice/invoice_sample_1.pdf")
    result = pdf_extractor.extract()
    print(result)
    # image_extractor = ImageInvoiceExtractor("../data/invoice/invoice_sample_2.png")
    # result = image_extractor.extract()
    # print(result)
    # image_extractor = ImageInvoiceExtractor("../data/invoice/invoice_sample_3.png")
    # result = image_extractor.extract()
    # print(result)
    # image_extractor = ImageInvoiceExtractor("../data/invoice/invoice_sample_4.png")
    # result = image_extractor.extract()
    # print(result)
