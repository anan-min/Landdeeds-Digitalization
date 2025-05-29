import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load the TrOCR model and processor
model_name = "microsoft/trocr-base-printed"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Function to process image and extract text using TrOCR


def process_image(image_path):
    try:
        # Open the image
        img = Image.open(image_path)

        # Preprocess the image and get predictions
        inputs = processor(images=img, return_tensors="pt")
        generated_ids = model.generate(inputs["pixel_values"])
        text = processor.decode(generated_ids[0], skip_special_tokens=True)

        return text
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


# Folder containing the images
image_folder = '../images'  # Update with your image folder path
output_file = 'TrOCR_result.txt'

# Open the output text file
with open(output_file, 'w', encoding='utf-8') as output:
    # Loop through all images in the folder
    for image_file in os.listdir(image_folder):
        # Check if the file is an image (you can add more extensions if needed)
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_file)
            print(f"Processing image: {image_path}")

            # Process the image and get the text result
            text_result = process_image(image_path)

            if text_result:
                # Write the result to the output file
                output.write(f"--- {image_file} ---\n")
                output.write(text_result + "\n\n")
            else:
                output.write(
                    f"--- {image_file} ---\nError: Could not process this image.\n\n")

print(f"TrOCR results saved to {output_file}")
