import os
from doctr.models import detection, recognition
from PIL import Image

# Load the DocTR OCR model
model = detection.TDDetectionModel.from_pretrained(
    "dbnet_resnet50")  # Use TDDetectionModel
recognizer = recognition.TDRecognitionModel.from_pretrained(
    "crnn_vgg16")  # Use TDRecognitionModel

# Path to the result text file
result_file_path = './doctr_result.txt'

# Function to process a single image using DocTR and store the result


def process_image(image_file_name):
    try:
        # Open the image
        img = Image.open(image_file_name)

        # Perform OCR on the image using DocTR
        print(f"Processing image: {image_file_name}")
        result = model(img)  # Detects textboxes in the image

        # Perform recognition on the detected textboxes
        recognized_text = recognizer(result)

        # Extract and join the text from the OCR result
        text = "\n".join([res['text'] for res in recognized_text])

        # Store the result in the result text file
        with open(result_file_path, 'a', encoding='utf-8') as result_file:
            result_file.write(f"Extracted Text for {image_file_name}:\n")
            result_file.write(text)
            result_file.write("\n" + "="*50 + "\n")

        print(f"Result appended for {image_file_name} in {result_file_path}")
        print("\n" + "="*50 + "\n")

        return text

    except Exception as e:
        print(f"Error processing {image_file_name}: {e}")
        return None

# Function to process all images in the folder


def process_images_in_folder(folder_path):
    try:
        # Get a list of all image files in the directory
        image_files = [f for f in os.listdir(
            folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Process each image in the folder
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            process_image(image_path)
    except Exception as e:
        print(f"Error processing images in folder: {e}")


# Set the path to your image folder
image_folder = '../images/'  # Update the path to your folder containing images

# Process all images in the folder
process_images_in_folder(image_folder)
