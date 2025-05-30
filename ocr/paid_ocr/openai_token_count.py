import openai
import dotenv
import os
import base64
from openai import OpenAI


# Load environment variables from .env file
dotenv.load_dotenv()
# Initialize OpenAI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
# Path to your image
# image_path = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), '..', 'images', '03-11.jpg')


input_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', '..', 'static', 'dol_compressed')
output_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', '..','research_results','ocr_results', 'openai_ocr_results')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)




def openai_ocr_image_to_text(image_path):
    # Getting the Base64 string
    base64_image = encode_image(image_path)


    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "This Image is landdeeds document in thai could you extract the text from this image" },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )

    tokens_used = response.usage.total_tokens
    print(f"Tokens used: {tokens_used}")


count = 0
for image_file in os.listdir(input_folder):
    # Check if the file is an image (you can add more extensions if needed)
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, image_file)
        
        # Process the image and get the extracted text
        extracted_text = openai_ocr_image_to_text(image_path)

        # # Save the extracted text to a text file
        # text_file_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
        
        # with open(text_file_path, 'w', encoding='utf-8') as f:
        #     f.write(extracted_text)

        # print(f"Text from {image_file} saved to {text_file_path}")

        count += 1
        if count == 10:
            break