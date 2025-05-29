from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import os

# Load Donut model and processor from Hugging Face
model_name = "naver-clova-ix/donut-base"  # Pretrained model from Hugging Face
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Function to process each image using Donut


def process_image(image_path):
    try:
        # Open the image
        img = Image.open(image_path)

        # Preprocess the image
        inputs = processor(images=img, return_tensors="pt")

        # Generate the output using Donut
        generated_ids = model.generate(inputs["pixel_values"])

        # Decode the generated output to text
        text = processor.decode(generated_ids[0], skip_special_tokens=True)

        return text
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Function to loop through images in a folder and save results in a text file


def process_images_in_folder(image_folder, output_file):
    with open(output_file, 'w', encoding='utf-8') as output:
        # Loop through all images in the folder
        for image_file in os.listdir(image_folder):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, image_file)
                print(f"Processing image: {image_path}")

                # Process the image and get the text result
                text_result = process_image(image_path)

                if text_result:
                    output.write(f"--- {image_file} ---\n")
                    output.write(text_result + "\n\n")
                else:
                    output.write(
                        f"--- {image_file} ---\nError: Could not process this image.\n\n")

    print(f"Donut OCR results saved to {output_file}")


# Set folder paths
# Replace with your image folder path
image_folder = 'path_to_your_image_folder'
output_file = 'donut_ocr_result.txt'       # Output file to save OCR results

# Process images in the folder
process_images_in_folder(image_folder, output_file)
