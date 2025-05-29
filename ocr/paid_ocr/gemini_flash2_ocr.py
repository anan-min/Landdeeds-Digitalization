import os
from google import genai
from google.genai import types
import base64
from PIL import Image
import io



# don't accept prepaid card 


# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to authenticate
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\\Projects\\typhoon_ocr\\focal-lens-449316-r7-51e6999bcd63.json"

def extract_text_from_image(image_path):
    """
    Extracts text from an image using Google Vertex AI's gemini-2.0-flash-001 model.
    """

    # Open the image file and convert it to base64
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

    # Create a GenAI client
    client = genai.Client(
        vertexai=True,
        project="focal-lens-449316-r7",
        location="global",
    )

    # Define model and content configuration
    model = "gemini-2.0-flash-001"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text=image_base64  # Send the base64 image data as text
                )
            ]
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=1,
        max_output_tokens=8192,
        safety_settings=[types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
        )],
    )

    # Make the API request to extract content
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

# Example usage
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, '..', '..', 'static', 'dol_compressed', '03-1.jpg')
extract_text_from_image(image_path)
