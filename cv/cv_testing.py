import os
import cv2
import pytesseract
from pytesseract import Output
import numpy as np


def process_image(image_path, output_folder,
                  brightness_increase=150,
                  gaussian_blur_kernel=(5, 5),
                  median_blur_kernel=5,
                  canny_threshold1=30,
                  canny_threshold2=100,
                  min_contour_area=100,
                  approx_poly_epsilon_factor=0.02):
    try:
        # Load image
        image_name = os.path.basename(image_path).split('.')[0]
        image = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 1: Increase brightness to fade text
        bright = cv2.add(gray, brightness_increase)  # Increase brightness
        blurred = cv2.GaussianBlur(bright, gaussian_blur_kernel, 0)

        # Optional: further blur small text artifacts
        cleaned = cv2.medianBlur(blurred, median_blur_kernel)

        # Step 2: Edge detection
        edges = cv2.Canny(cleaned, canny_threshold1, canny_threshold2)

        # Step 3: Contour detection
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw only large contours (filter out small noise)
        output = image.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_contour_area:  # Adjust this threshold as needed
                cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

        # Save the output image with contours
        output_name = os.path.join(output_folder, f"polygon_{image_name}.jpg")
        cv2.imwrite(output_name, output)

        print(f"Processed and saved result for: {image_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(
    script_dir, '..', 'research_results', 'object_detection', 'roboflow_cropped')
# Define specific output folders for each test set
testing1_folder = os.path.join(
    script_dir, '..', 'research_results', 'cv_updated_results', 'testing', 'testing1')
testing2_folder = os.path.join(
    script_dir, '..', 'research_results', 'cv_updated_results', 'testing',  'testing2')
testing3_folder = os.path.join(
    script_dir, '..', 'research_results', 'cv_updated_results', 'testing',  'testing3')

# Ensure output directories exist
os.makedirs(testing1_folder, exist_ok=True)
os.makedirs(testing2_folder, exist_ok=True)
os.makedirs(testing3_folder, exist_ok=True)


default_params = {
    'brightness_increase': 150,
    'gaussian_blur_kernel': (5, 5),
    'median_blur_kernel': 5,
    'canny_threshold1': 30,
    'canny_threshold2': 100,
    'min_contour_area': 100,
    'approx_poly_epsilon_factor': 0.02
}

more_sensitive_1_params = {
    'brightness_increase': 210,  # Increased brightness to enhance even fainter lines
    # Smaller blur kernel to preserve even finer details
    'gaussian_blur_kernel': (3, 3),
    'median_blur_kernel': 3,  # Minimal blur to preserve small features
    'canny_threshold1': 15,  # Lower threshold to detect very faint edges
    'canny_threshold2': 70,  # Slightly adjusted upper threshold to refine edge detection
    'min_contour_area': 40,  # Lower the contour area to capture very fine contours
    # More precise approximation for ultra-thin lines
    'approx_poly_epsilon_factor': 0.01
}

more_sensitive_2_params = {
    'brightness_increase': 230,  # Maximum brightness to reveal very faint lines
    'gaussian_blur_kernel': (3, 3),  # Even smaller kernel to keep sharp edges
    'median_blur_kernel': 3,  # Minimal median blur to preserve the smallest details
    'canny_threshold1': 5,  # Extremely low threshold to catch very faint edges
    'canny_threshold2': 50,  # Further lowered threshold for better edge sensitivity
    'min_contour_area': 20,  # Very low contour area to capture fine contours
    # More precise approximation to preserve ultra-thin lines
    'approx_poly_epsilon_factor': 0.005
}

more_sensitive_3_params = {
    # Maximal brightness to ensure even the faintest lines are visible
    'brightness_increase': 240,
    # Minimal blur to preserve the finest details
    'gaussian_blur_kernel': (3, 3),
    'median_blur_kernel': 3,  # No blur to ensure no loss of very small lines
    'canny_threshold1': 3,  # Extremely low threshold to catch the faintest edges
    'canny_threshold2': 40,  # Very low upper threshold for extremely subtle edge detection
    'min_contour_area': 10,  # Minimal contour area to capture the smallest possible contours
    # Ultra-precise polygon approximation for capturing every detail of thin lines
    'approx_poly_epsilon_factor': 0.002
}



balanced_sensitivity_params = {
    'brightness_increase': 0,               # No brightness increase
    'gaussian_blur_kernel': (7, 7),          # Slightly larger Gaussian blur
    'median_blur_kernel': 5,                 # Slightly larger Median blur
    'canny_threshold1': 20,                  # Lower Canny threshold for detecting more edges
    'canny_threshold2': 80,                  # Adjusted Canny threshold (ratio of 1:4)
    'min_contour_area': 75,                  # Reasonable contour area to filter out noise
    'approx_poly_epsilon_factor': 0.02,      # Standard approximation factor for contours
}

sharper_edges_params = {
    'brightness_increase': 0,                # No brightness increase
    'gaussian_blur_kernel': (3, 3),          # Minimal Gaussian blur to retain sharp edges
    'median_blur_kernel': 3,                 # Minimal Median blur to preserve fine details
    'canny_threshold1': 15,                  # Lower Canny threshold for detecting faint edges
    'canny_threshold2': 45,                  # Closer Canny thresholds (ratio of 1:3)
    'min_contour_area': 60,                  # Lower contour area to capture more details
    'approx_poly_epsilon_factor': 0.015,     # More precise approximation for contours
}

contrast_enhancement_params = {
    'brightness_increase': 0,                # No brightness increase (as inversion is applied)
    'gaussian_blur_kernel': (5, 5),          # Moderate blur to smooth out noise
    'median_blur_kernel': 5,                 # Moderate median blur to reduce noise
    'canny_threshold1': 25,                  # Slightly higher lower Canny threshold for filtering some noise
    'canny_threshold2': 75,                  # Standard Canny ratio (1:3)
    'min_contour_area': 80,                  # Slightly higher contour area to filter small contours
    'approx_poly_epsilon_factor': 0.02,      # Standard approximation factor
}



for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        process_image(image_path, testing1_folder, **balanced_sensitivity_params)
        process_image(image_path, testing2_folder, **sharper_edges_params)
        process_image(image_path, testing3_folder, **contrast_enhancement_params)
