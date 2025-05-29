from inference_sdk import InferenceHTTPClient
import dotenv
import os
dotenv.load_dotenv()

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY"),
)

test_image = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', '..', 'static', 'dol_compressed', '03-1.jpg')
result = CLIENT.infer(test_image, model_id="mapdetection-pyotq/5")
print(result)