import os
import sys
import json
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import time


with open('./settings.json', 'r') as f:
    data = json.load(f)
    print("Read successful")

ENDPOINT = data["ENDPOINT"]
KEY = data["ACCOUNT_KEY"]
client = ImageAnalysisClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))



def process_image_with_ms_ocr(image_path): 
    extracted_text = ""
    with open(image_path, "rb") as image_file:
        result = client.analyze(
            image_data=image_file,
            visual_features=[VisualFeatures.read]
        )

        if result.read is not None:
            for block in result.read.blocks:
                for line in block.lines:
                    extracted_text += line.text + "\n" 
    return extracted_text





def process_images_in_folder():
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)

        text = process_image_with_ms_ocr(image_path)

        output_file_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(text)

        print(f"Processed {image_file}, saved text to {output_file_path}")

        if (idx + 1) % 20 == 0:
            print("Sleeping for 60 seconds to respect the rate limit...")
            time.sleep(60)




input_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', '..', 'static', 'dol_compressed')
output_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', '..','research_results','ocr_results', 'ms_azure_ocr_results')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

process_images_in_folder()