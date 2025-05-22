import os
from paddleocr import PaddleOCR

# Set the folder paths
input_folder = '../images'  # Folder where original images are stored
output_file = 'paddle_ocr_result.txt'  # File to store OCR results

# Initialize the PaddleOCR model for Thai language
ocr = PaddleOCR(use_angle_cls=True, lang='th')  # Use Thai language model

# Function to perform OCR and store results in the output file
def process_image(image_path, output_file):
    try:
        # Perform OCR on the image
        result = ocr.ocr(image_path, cls=True)

        # Open the output file to append OCR results
        with open(output_file, 'a', encoding='utf-8') as f:
            # Write the image name as the header
            f.write(f"Results for image: {os.path.basename(image_path)}\n")
            f.write("="*50 + "\n")
            
            # Loop through OCR results and write them to file
            for line in result[0]:
                text = line[1][0]
                f.write(text + '\n')
            
            f.write("\n\n")  # Add some space between each image's results

        print(f"Processed and saved results for: {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Open the output file in write mode to start fresh
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("PaddleOCR Results\n")
    f.write("="*50 + "\n\n")

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # Full path to the image
        image_path = os.path.join(input_folder, filename)
        
        # Process the image and store the result in the output file
        process_image(image_path, output_file)
