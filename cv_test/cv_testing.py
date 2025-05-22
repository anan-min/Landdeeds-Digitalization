import os
import cv2
import pytesseract
from pytesseract import Output
import numpy as np

# Set the folder paths
input_folder = '../images'  # Folder where original images are stored
# Folder to save processed images with contours
output_folder = '../research_results/cv_results/testing'

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

        # Step 1: Increase brightness slightly to fade text, with a modest adjustment
        bright = cv2.add(gray, 100)  # Moderate brightness enhancement
        blurred = cv2.GaussianBlur(bright, (5, 5), 0)

        # Step 2: OCR Text Detection
        d = pytesseract.image_to_data(blurred, output_type=Output.DICT)

        # Step 3: Mask text regions with stronger intensity (slightly enlarged)
        mask = np.ones_like(gray) * 255  # Start with white canvas

        for i in range(len(d['text'])):
            if int(d['conf'][i]) > 30 and len(d['text'][i].strip()) > 1:
                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                # Slightly enlarged mask
                cv2.rectangle(blurred, (x - 5, y - 5),
                              (x + w + 5, y + h + 5), 255, -1)

        # Optional: further blur small text artifacts
        # Slightly stronger blur but not excessive
        cleaned = cv2.medianBlur(blurred, 5)

        # Step 4: Apply CLAHE for contrast enhancement (moderate clipLimit)
        # Slightly stronger CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(cleaned)

        # Step 5: Edge detection (with moderate sensitivity)
        edges = cv2.Canny(enhanced, 50, 150)  # Balanced edge detection

        # Step 6: Denoise edges using morphological operations (moderate dilation)
        # Small kernel to preserve fine lines
        kernel = np.ones((3, 3), np.uint8)
        # Moderate number of iterations
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Step 7: Contour detection
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image to draw contours
        output = image.copy()

        # Draw only large contours (filter out small text noise)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:  # Reduced threshold for better detail preservation
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
