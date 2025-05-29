import os
import easyocr

# Initialize the EasyOCR reader for Thai
reader = easyocr.Reader(['th'])

# Path to the result text file
result_file_path = './easy_ocr_result.txt'

# Function to process a single image using EasyOCR and store the result


def process_image(image_file_name):
    try:
        # Perform OCR on the image using EasyOCR
        print(f"Processing image: {image_file_name}")
        result = reader.readtext(image_file_name)

        # Extract and join the text from the OCR result
        text = "\n".join([res[1] for res in result])

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
