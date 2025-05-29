import cv2
import numpy as np
import os


def remove_text_from_image(image_path, output_path):
    image = cv2.imread(image_path)
    original_size = image.shape[:2]
    image = cv2.resize(image, inputSize)
    inpaint_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    boxesDB50, _ = textDetectorEAST.detect(image)


    for box in boxesDB50:
        box = np.array(box, np.int32)
        x, y, w, h = cv2.boundingRect(box)
        shrink_factor = 0.5
        x, y, w, h = int(x + (1 - shrink_factor) * w), int(y + (1 - shrink_factor)
                                                        * h), int(w * shrink_factor), int(h * shrink_factor)
        smaller_box = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.int32)
        cv2.fillPoly(inpaint_mask, [smaller_box], 255)
        cv2.polylines(image, [smaller_box], isClosed=True,
                    color=(0, 0, 255), thickness=1)

    inpainted_image = cv2.inpaint(
        image, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_NS)

    inpainted_image_resize = cv2.resize(
        inpainted_image, (original_size[1], original_size[0]))

    cv2.imwrite(output_path, inpainted_image_resize)


if __name__ == "__main__":
    input_folder = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), '..', 'images')
    output_folder = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), '..', 'research_results', 'text_removal', 'east_results')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    textDetectorEAST = cv2.dnn_TextDetectionModel_EAST(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frozen_east_text_detection.pb'))

    inputSize = (320, 320)
    conf_thresh = 0.8
    nms_thresh = 0.4
    bin_thresh = 0.3
    poly_thresh = 0.5
    mean = (122.67891434, 116.66876762, 104.00698793)

    textDetectorEAST.setConfidenceThreshold(
        conf_thresh).setNMSThreshold(nms_thresh)
    textDetectorEAST.setInputParams(
        1.0, inputSize, (123.68, 116.78, 103.94), True)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)

        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_path = os.path.join(output_folder, filename)

            remove_text_from_image(image_path, output_path)
            print(f"Processed and saved: {output_path}")
