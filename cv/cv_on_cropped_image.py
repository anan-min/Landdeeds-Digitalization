import os
import cv2
import pytesseract
from pytesseract import Output
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..','research_results','object_detection', 'roboflow_cropped')
output_folder = os.path.join(
    script_dir, '..', 'research_results', 'cv_updated_results',  'cv_on_cropped_image')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)



def process_image(image_path: str, output_folder: str,
                           brightness_increase: int = 150,
                           gaussian_blur_kernel: tuple = (5, 5),
                           median_blur_kernel: int = 5,
                           canny_threshold1: int = 30,
                           canny_threshold2: int = 100,
                           min_contour_area: int = 100,
                           approx_poly_epsilon_factor: float = 0.02):
    """
    Processes an image to detect and highlight polygons by fading text,
    performing edge detection, and finding contours.

    Args:
        image_path: Path to the input image.
        output_folder: Path to the folder where the processed image will be saved.
        brightness_increase: Value to add to pixel intensity to increase brightness.
        gaussian_blur_kernel: Kernel size for Gaussian blur.
        median_blur_kernel: Kernel size for Median blur.
        canny_threshold1: Lower threshold for the Canny edge detector.
        canny_threshold2: Upper threshold for the Canny edge detector.
        min_contour_area: Minimum area of a contour to be considered a polygon.
        approx_poly_epsilon_factor: Factor multiplied by arc length for polygon approximation.
                                    Smaller values mean more vertices, larger mean fewer.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    if not os.path.isdir(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        except OSError as e:
            print(f"Error creating output folder {output_folder}: {e}")
            return

    try:
        # Load image
        image_name = os.path.basename(image_path).split('.')[0]
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not load image from {image_path}. Check file format/corruption.")
            return

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 1: Increase brightness and blur to fade text
        # Using np.clip to ensure values stay within 0-255 after addition
        bright = np.clip(gray + brightness_increase, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(bright, gaussian_blur_kernel, 0)
        cleaned = cv2.medianBlur(blurred, median_blur_kernel)

        # Optional: Apply adaptive thresholding to further fade text (experimental)
        # _, cleaned_thresh = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cleaned = cv2.bitwise_not(cleaned_thresh) # Invert if text is black on white

        # Step 2: Edge detection
        edges = cv2.Canny(cleaned, canny_threshold1, canny_threshold2)

        # Step 3: Contour detection
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL for outer contours

        # Draw only large and approximated contours
        output = image.copy()
        found_polygons_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_contour_area:
                perimeter = cv2.arcLength(cnt, True)
                # Approximate the contour to a polygon
                approx = cv2.approxPolyDP(cnt, approx_poly_epsilon_factor * perimeter, True)

                # Filter by number of vertices (e.g., at least 3 for a polygon)
                if len(approx) >= 3:
                    cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)
                    found_polygons_count += 1

        # Save the output image with contours
        output_name = os.path.join(output_folder, f"polygon_detected_{image_name}.jpg")
        cv2.imwrite(output_name, output)

        print(f"Processed image: {image_path}. Found {found_polygons_count} potential polygons. Result saved to: {output_name}")

    except cv2.error as e:
        print(f"OpenCV error processing {image_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred processing {image_path}: {e}")



testing1_folder = os.path.join(
    script_dir, '..', 'research_results', 'cv_updated_results',  'testing', 'testing1')
testing2_folder = os.path.join(
    script_dir, '..', 'research_results', 'cv_updated_results', 'testing',  'testing2')
testing3_folder = os.path.join(
    script_dir, '..', 'research_results', 'cv_updated_results', 'testing', 'testing3')

os.makedirs(testing1_folder, exist_ok=True)
os.makedirs(testing2_folder, exist_ok=True)
os.makedirs(testing3_folder, exist_ok=True)


# Loop through all files in the input folder and process the images
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Process image files only
        image_path = os.path.join(input_folder, filename)

        process_image(
            image_path=image_path,
            output_folder=testing1_folder,
            brightness_increase=80,  # High brightness increase
            gaussian_blur_kernel=(15, 15),  # Strong blur to remove noise
            median_blur_kernel=9,  # Strong median blur to reduce smaller noise
            canny_threshold1=50,  # Higher Canny threshold to reduce excessive edges
            canny_threshold2=150,  # Higher second threshold to suppress minor edges
            min_contour_area=200,  # Large minimum contour area to filter small noise
            approx_poly_epsilon_factor=0.03  # Slightly higher epsilon for smoother contours
        )

        process_image(
            image_path=image_path,
            output_folder=testing2_folder,
            brightness_increase=80,  # High brightness increase
            gaussian_blur_kernel=(9, 9),  # Moderate blur to soften edges
            median_blur_kernel=7,  # Moderate median blur
            canny_threshold1=20,  # Lower Canny threshold to detect edges more conservatively
            canny_threshold2=100,  # Adjusted second threshold for edge refinement
            min_contour_area=100,  # Moderate minimum contour area for noise filtering
            approx_poly_epsilon_factor=0.02  # Balanced epsilon for contour approximation
        )
        process_image(
            image_path=image_path,
            output_folder=testing3_folder,
            brightness_increase=80,  # High brightness increase
            gaussian_blur_kernel=(5, 5),  # Minimal blur to retain edges and details
            median_blur_kernel=5,  # Mild median blur
            canny_threshold1=10,  # Lower Canny threshold to detect even faint edges
            canny_threshold2=60,  # Lower second threshold for more edge sensitivity
            min_contour_area=50,  # Smaller contour area threshold to catch fine details
            approx_poly_epsilon_factor=0.01  # Lower epsilon to keep more detailed contours
        )