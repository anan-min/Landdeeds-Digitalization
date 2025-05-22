import os
import kraken

# Path to your image folder
image_folder = '../images'  # Folder with images (relative path)
# Path to the result text file
result_file_path = './kraken_ocr_result.txt'

# Function to perform OCR with Kraken
def kraken_ocr(image_path):
    try:
        # Perform OCR with Kraken model directly
        text = kraken.model.recognize(image_path)
        return text
    except Exception as e:
        print(f"Error processing image {image_path} with Kraken: {e}")
        return None

# Function to process a single image and store the result
def process_image(image_file_name):
    try:
        print(f"Processing image: {image_file_name}")

        # Perform OCR with Kraken
        text = kraken_ocr(image_file_name)

        if text is None:
            print(f"Skipping {image_file_name} due to OCR failure.")
            return None

        # Append the OCR result to the result text file
        with open(result_file_path, 'a', encoding='utf-8') as result_file:
            result_file.write(f"Extracted Text for {image_file_name} (Kraken OCR):\n")
            result_file.write(text)
            result_file.write("\n" + "="*50 + "\n")

        print(f"Result appended for {image_file_name} in {result_file_path}")
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
