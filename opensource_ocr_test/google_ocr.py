import os
from google.cloud import vision


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './focal-lens-449316-r7-baaeecd8992c.json'

# Initialize the Google Vision API client
client = vision.ImageAnnotatorClient()


def detect_text_in_image(image_path):
    """Detects text in an image file."""
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Perform text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Check if any text was found
    if texts:
        # The first item in the texts list is the full detected text
        detected_text = texts[0].description
        print(f"Detected text: {detected_text}")
        return detected_text
    else:
        print("No text detected.")
        return ""


# Example usage: Detect text in an image
image_path = '../images/02.jpg'  # Replace with your image path
detected_text = detect_text_in_image(image_path)

# Optionally, save the result to a file
with open("google_ocr_result.txt", "w") as result_file:
    result_file.write(f"Detected text for {image_path}:\n")
    result_file.write(detected_text + "\n")
