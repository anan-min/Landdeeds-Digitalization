from inference_sdk import InferenceHTTPClient
import dotenv
import os
import base64
from PIL import Image, ImageDraw


dotenv.load_dotenv()
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY"),
)

input_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', 'static', 'dol_compressed')
output_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..','research_results','object_detection', 'roboflow')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)




def process_image(image_path):
    # Open the original image to get its size
    original_image = Image.open(image_path)
    original_width, original_height = original_image.size
    
    # Read and encode the image in base64 for API inference
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

    # Perform inference with the encoded image
    result = CLIENT.infer(encoded_image, model_id="mapdetection-pyotq/5")
    
    # Extract predictions from the result
    predictions = result.get("predictions", [])
    
    # Get the size of the image used for inference
    infer_width = result["image"]["width"]
    infer_height = result["image"]["height"]
    
    # Calculate the scaling factors for width and height
    scale_x = original_width / infer_width
    scale_y = original_height / infer_height

    # Open the image again to draw on it
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Iterate over each prediction and adjust the coordinates
    for prediction in predictions:
        # Get the bounding box coordinates and size from API result
        x_center = int(prediction['x'] * scale_x)  # Adjust for original size
        y_center = int(prediction['y'] * scale_y)  # Adjust for original size
        width = int(prediction['width'] * scale_x)  # Adjust for original size
        height = int(prediction['height'] * scale_y)  # Adjust for original size

        # Calculate the top-left and bottom-right corners based on center and size
        x_left = x_center - (width // 2)
        y_top = y_center - (height // 2)
        x_right = x_center + (width // 2)
        y_bottom = y_center + (height // 2)

        # Draw the bounding box on the image (outline in red)
        draw.rectangle([x_left, y_top, x_right, y_bottom], outline="red", width=3)

    # Define the output path for the image with highlighted objects
    output_image_path = os.path.join(output_folder, os.path.basename(image_path))

    # Save the image with bounding boxes
    image.save(output_image_path)
    print(f"Highlighted image saved to: {output_image_path}")


    

count = 0
for filename in os.listdir(input_folder):
    count += 1
    if count == 10:
        break
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        process_image(image_path)