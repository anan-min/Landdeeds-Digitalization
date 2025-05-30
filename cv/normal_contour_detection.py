import os
import cv2
import pytesseract
from pytesseract import Output
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(script_dir, '..', 'static', 'dol_compressed')
output_folder = os.path.join(
    script_dir, '..', 'research_results', 'cv_updated_results',  'normal_contour_detection')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)



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

        # Optional: further blur small text artifacts
        cleaned = cv2.medianBlur(blurred, 5)

        # Step 2: Edge detection
        edges = cv2.Canny(cleaned, 30, 100)

        # Step 3: Contour detection
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw only large contours (filter out small noise)
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
