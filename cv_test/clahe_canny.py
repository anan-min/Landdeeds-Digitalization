import os
import cv2
import pytesseract
from pytesseract import Output
import numpy as np

# Set the folder paths
input_folder = '../images'  # Folder where original images are stored
output_folder = '../contour2'  # Folder to save processed images with contours

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

        # Step 1: Increase brightness to fade text (use CLAHE for contrast enhancement)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Step 2: Apply adaptive thresholding to separate text from the background
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Step 3: Use morphological transformations (dilation) to remove small noise and fill text regions
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # Step 4: OCR Text Detection to remove text areas
        d = pytesseract.image_to_data(dilated, output_type=Output.DICT)

        for i in range(len(d['text'])):
            if int(d['conf'][i]) > 30 and len(d['text'][i].strip()) > 1:
                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                cv2.rectangle(dilated, (x - 5, y - 5), (x + w + 5,
                              y + h + 5), 255, -1)  # Mask text regions

        # Step 5: Edge detection (Canny)
        # Adjust these thresholds based on image quality
        edges = cv2.Canny(dilated, 50, 150)

        # Step 6: Contour detection
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw only large contours (filter out small text noise)
        output = image.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Filter out small contours (text or noise)
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
