import cv2
import numpy as np
import os

# Load the EAST model for text detection
# Make sure you have the EAST model .pb file
net = cv2.dnn.readNet("frozen_east_text_detection.pb")
input_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', 'research_results', 'ocr_results', 'ms_azure_results')
output_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', 'research_results', 'object_detection', 'yolo_text_processed')

# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Decode predicted bounding boxes (detect text areas)


def decode_predictions(scores, geometry, score_thresh=0.5, nms_thresh=0.4):
    detections = []
    confidences = []
    for y in range(scores.shape[2]):
        for x in range(scores.shape[3]):
            score = scores[0, 0, y, x]
            if score < score_thresh:
                continue
            offset_x, offset_y = x * 4.0, y * 4.0
            angle = geometry[0, 4, y, x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = geometry[0, 0, y, x]
            w = geometry[0, 1, y, x]
            dx = geometry[0, 2, y, x]
            dy = geometry[0, 3, y, x]
            end_x = int(offset_x + cos * dx + sin * dy)
            end_y = int(offset_y - sin * dx + cos * dy)
            start_x = int(offset_x - cos * dx - sin * dy)
            start_y = int(offset_y + sin * dx - cos * dy)

            # Add bounding box
            detections.append([start_x, start_y, end_x, end_y])
            confidences.append(score)
    return detections, confidences

# Function to process and save each image


def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    orig_image = image.copy()

    # Resize image for better detection
    height, width = image.shape[:2]
    new_width = 320
    new_height = int((new_width / width) * height)
    image_resized = cv2.resize(image, (new_width, new_height))

    # Prepare the image for EAST detector
    blob = cv2.dnn.blobFromImage(
        image_resized, 1.0, (new_width, new_height), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    # Run forward pass to get bounding box predictions
    scores, geometry = net.forward(["score1", "geometry1"])

    # Get boxes and confidences
    boxes, confidences = decode_predictions(scores, geometry)

    # Apply Non-Maximum Suppression (NMS) to remove redundant boxes
    boxes = np.array(boxes)
    indices = cv2.dnn.NMSBoxesRotated(
        boxes, confidences, score_thresh=0.5, nms_thresh=0.4)

    # Mask out detected text areas
    for i in range(len(indices)):
        box = boxes[indices[i]]
        # Mask text with black color
        cv2.polylines(orig_image, [box], True, (0, 0, 0), thickness=2)

    # Now apply contour detection on the remaining image (without text)
    gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours (detected shapes)
    cv2.drawContours(orig_image, contours, -1, (0, 255, 0), 2)

    # Save the processed image to the output folder
    cv2.imwrite(output_path, orig_image)

# Process all images in the input folder and save the result


def process_all_images(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"processed_{filename}")
            process_image(image_path, output_path)
            print(f"Processed and saved: {output_path}")

# Main function to start processing


def main():
    process_all_images(input_folder, output_folder)


if __name__ == "__main__":
    main()
