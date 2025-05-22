import os
import textract

# Set the folder paths
input_folder = '../images'  # Folder where original images are stored
output_file = 'textract_ocr_result.txt'  # File to store OCR results

# Function to perform OCR using Textract and store results


def process_image(image_path, output_file):
    try:
        # Perform OCR on the image
        text = textract.process(image_path)

        # Open the output file to append OCR results
        with open(output_file, 'a', encoding='utf-8') as f:
            # Write the image name as the header
            f.write(f"Results for image: {os.path.basename(image_path)}\n")
            f.write("="*50 + "\n")

            # Write the extracted text to the file
            f.write(text.decode('utf-8') + '\n')
            f.write("\n\n")  # Add some space between each image's results

        print(f"Processed and saved results for: {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


# Open the output file in write mode to start fresh
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("Textract OCR Results\n")
    f.write("="*50 + "\n\n")

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.pdf')):  # Process image and PDF files
        # Full path to the image or PDF
        image_path = os.path.join(input_folder, filename)

        # Process the image and store the result in the output file
        process_image(image_path, output_file)
