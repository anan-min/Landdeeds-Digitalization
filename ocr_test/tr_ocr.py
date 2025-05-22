import os
from PIL import Image
import pytesseract


Image.MAX_IMAGE_PIXELS = None

# Path to your image folder
image_folder = '../images'  # Folder with images (relative path)
# Path to the result text file
result_file_path = './tr_ocr_result.txt'

# Function to process a single image and extract text


def process_image(image_file_name):
    try:
        # Open the image
        with Image.open(image_file_name) as img:
            print(f"Processing image: {image_file_name}")

            # Perform OCR on the image
            text = pytesseract.image_to_string(img)

            # Append the OCR result to the result text file
            with open(result_file_path, 'a', encoding='utf-8') as result_file:
                result_file.write(f"Extracted Text for {image_file_name}:\n")
                result_file.write(text)
                result_file.write("\n" + "="*50 + "\n")

            print(
                f"Result appended for {image_file_name} in {result_file_path}")
            print("\n" + "="*50 + "\n")

        return image_file_name

    except Exception as e:
        print(f"Error processing {image_file_name}: {e}")
        return None

# Function to process all images in the folder


def process_all_images(image_folder):
    try:
        # Loop through all the images in the folder
        for filename in os.listdir(image_folder):
            # Check for image file extensions
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(image_folder, filename)
                print(f"Full path to image: {image_path}")

                # Process the image and extract text
                process_image(image_path)

    except Exception as e:
        print(f"Error processing images: {e}")


# Example usage
if __name__ == "__main__":
    process_all_images(image_folder)  # Process all images in the folder
