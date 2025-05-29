import os
import cv2
import pytesseract
from pytesseract import Output
import numpy as np

input_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', 'images')
output_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', 'research_results', 'cv_results', 'update_cd')

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to process each image


def process_image(image_path, output_folder):
    try:
        # Load image
        image_name = os.path.basename(image_path).split('.')[0]
        image = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 1: Increase brightness to fade text
        bright = cv2.add(gray, 150)  # Increase brightness
        blurred = cv2.GaussianBlur(bright, (5, 5), 0)

        # Step 2: OCR Text Detection
        d = pytesseract.image_to_data(blurred, output_type=Output.DICT)

        # Step 3: Mask text regions with stronger intensity
        mask = np.ones_like(gray) * 255  # Start with white canvas
        masking_factor = 2

        for i in range(len(d['text'])):
            if int(d['conf'][i]) > 30 and len(d['text'][i].strip()) > 1:
                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]

                # Calculate the mask size based on the masking factor
                # Multiply padding by masking_factor
                padding_x = int(masking_factor * 5)
                padding_y = int(masking_factor * 5)  # Same for y direction

                # Apply the mask with the adjusted size
                cv2.rectangle(blurred, (x - padding_x, y - padding_y),
                            (x + w + padding_x, y + h + padding_y), 255, -1)

        # Optional: further blur small text artifacts
        cleaned = cv2.medianBlur(blurred, 5)

        # Step 4: Edge detection
        edges = cv2.Canny(cleaned, 50, 100)

        # Step 5: Contour detection
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw only large contours (filter out small text noise)
        output = image.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Adjust this threshold as needed
                cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

        # Save the output image with contours
        output_name = os.path.join(output_folder, f"polygon_{image_name}.jpg")
        cv2.imwrite(output_name, output)

        print(f"Processed and saved result for: {image_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


# Loop through all files in the input folder and process the images
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Process image files only
        image_path = os.path.join(input_folder, filename)
        process_image(image_path, output_folder)
