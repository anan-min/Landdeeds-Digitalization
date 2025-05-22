from PIL import Image
import os
from typhoon_ocr import ocr_document

# Relative path to the 'images' folder in the same directory as the script
image_folder = 'images'  
typhoon_api_key = "sk-FsXJJmQBEw6YLmT1cM2TL5Tbo4bTRYmrZTaDZxqXNXMCrrtV"
os.environ['TYPHOON_OCR_API_KEY'] = typhoon_api_key
Image.MAX_IMAGE_PIXELS = None

def process_and_store_result(image_file_path):
    try:
        # Check if the file exists
        if not os.path.exists(image_file_path):
            print(f"Error: The file does not exist: {image_file_path}")
            return  # Skip this image if it doesn't exist

        # Try to open the image with PIL to verify if it's accessible and valid
        try:
            with Image.open(image_file_path) as img:
                print(f"Successfully opened image: {image_file_path}, Format: {img.format}")
        except Exception as e:
            print(f"Error opening image: {image_file_path}. Error: {e}")
            return

        print(f"Processing image: {image_file_path}")

        # Send the image to Typhoon OCR and get the result
        markdown = ocr_document(image_file_path, task_type="structure")

        with open("typhoon_ocr_structure_result.txt", "a") as result_file:
            result_file.write(f"Result for {image_file_path}:\n")
            result_file.write(markdown + "\n\n")

        print(f"Result for {image_file_path} stored in typhoon_ocr_result.txt")

    except Exception as e:
        print(f"Error processing image {image_file_path}: {e}")

def process_all_images_in_folder(folder_path):
    try:
        # Get a list of all image files in the directory
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in image_files:
            # Use os.path.join for the file path
            image_path = os.path.join(folder_path, image_file)

            # Print the full path before processing
            print(f"Full path to image: {image_path}")

            # Process image with Typhoon OCR and store the result
            process_and_store_result(image_path)

    except Exception as e:
        print(f"Error processing images in folder: {e}")

# Process all images in the specified folder
process_all_images_in_folder(image_folder)
