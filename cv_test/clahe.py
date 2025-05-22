import os
import cv2
import pytesseract
from pytesseract import Output
import numpy as np

# Set the folder paths
input_folder = '../images'  # Folder where original images are stored
# Folder to save processed images with contours
output_folder = '../research_results/cv_results/clahe'

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

        # Step 1: CLAHE for contrast enhancement (tuned for better contrast)
        # Increased clipLimit for better enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Step 2: Apply adaptive thresholding to separate text from the background
        # Adaptive thresholding for better separation
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Step 3: Denoising using morphological transformations (dilation)
        # Use larger kernel size to help clean up regions
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=3)

        # Optional: Further denoise small noise using Median Blur
        denoised = cv2.medianBlur(dilated, 5)

        # Step 4: OCR Text Detection (mask out text regions)
        d = pytesseract.image_to_data(denoised, output_type=Output.DICT)

        # Mask text regions with stronger intensity (smaller masks for text)
        for i in range(len(d['text'])):
            if int(d['conf'][i]) > 30 and len(d['text'][i].strip()) > 1:  # Filter based on confidence
                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                # Use smaller mask to remove text without affecting shape
                cv2.rectangle(denoised, (x - 5, y - 5),
                              (x + w + 5, y + h + 5), 255, -1)  # Smaller mask

        # Step 5: Find contours (detection of potential land shapes)
        contours, _ = cv2.findContours(
            denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank image to draw detected shapes
        output = image.copy()

        # Filter out small contours and focus on larger shapes
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:  # Larger area threshold to ignore text and noise
                # Draw only large contours
                cv2.drawContours(output, [cnt], -1, (0, 255, 0), 3)

        # Step 6: Connected component analysis (to further clean up)
        # Create a binary mask for connected components analysis
        binary_mask = np.zeros_like(gray)
        cv2.drawContours(binary_mask, contours, -1, 255, thickness=cv2.FILLED)

        # Find connected components in the binary mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8)

        # Step 7: Filter out small components and keep only the land shape
        for i in range(1, num_labels):  # Skip label 0, which is the background
            x, y, w, h, area = stats[i]
            if area > 5000:  # Adjust this area threshold to focus on the land shape
                # Draw the bounding box for each connected component
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the output image with detected land shape
        output_name = os.path.join(
            output_folder, f"polygon_{image_name}.jpg")
        cv2.imwrite(output_name, output)

        print(f"Processed and saved result for: {image_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


# Loop through all files in the input folder and process the images
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Process image files only
        image_path = os.path.join(input_folder, filename)
        process_image(image_path, output_folder)
