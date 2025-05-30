from inference_sdk import InferenceHTTPClient
import dotenv
import os
import base64
from PIL import Image, ImageDraw

confidence_threshold = 0.6


dotenv.load_dotenv()
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_TD_API_KEY"),
)

input_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..','research_results','object_detection', 'roboflow_cropped')
output_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..','research_results','object_detection', 'roboflow_text_detection')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)



def process_image(image_path, confidence_threshold):
    original_image = Image.open(image_path)
    original_width, original_height = original_image.size
    
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

    result = CLIENT.infer(encoded_image, model_id="thaitextdetection/1")
    
    predictions = result.get("predictions", [])
    
    infer_width = result["image"]["width"]
    infer_height = result["image"]["height"]
    
    scale_x = original_width / infer_width
    scale_y = original_height / infer_height

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    

    prediction = sorted_predictions[0]
    if predictions:
        for prediction in predictions:
            x_center = int(prediction['x'] * scale_x)  
            y_center = int(prediction['y'] * scale_y)  
            width = int(prediction['width'] * scale_x)  
            height = int(prediction['height'] * scale_y)  

            
            x_left = x_center - (width // 2)
            y_top = y_center - (height // 2)
            x_right = x_center + (width // 2)
            y_bottom = y_center + (height // 2)


            # find region crop blur paste
            region = image.crop((x_left, y_top, x_right, y_bottom))
            blurred_region = region.filter(ImageFilter.GaussianBlur(radius=5))
            image.paste(blurred_region, (x_left, y_top))

    output_image_path = os.path.join(output_folder, os.path.basename(image_path))

    image.save(output_image_path)
    print(f"Filter image saved to: {output_image_path}")


    

count = 0
for filename in os.listdir(input_folder):
    count += 1
    if count == 10:
        break
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        process_image(image_path, confidence_threshold)