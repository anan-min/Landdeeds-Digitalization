import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import torch

# === CONFIGURATION ===
image_path = "../images/02-68.jpg"
sam_checkpoint = "sam_vit_h.pth"  # Download from Meta and set path
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD IMAGE ===
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# === LOAD SAM MODEL ===
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# === SET PROMPT: POINT IN THE CENTER OF THE MAP ===
# You might adjust this based on your image layout
h, w = image_rgb.shape[:2]
input_point = np.array([[w // 2, h // 2]])  # middle of image
input_label = np.array([1])  # foreground point

# === RUN PREDICTION ===
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# === SELECT BEST MASK ===
best_mask = masks[np.argmax(scores)]

# === DRAW RESULT ===
segmented = image_rgb.copy()
segmented[~best_mask] = [255, 255, 255]  # make non-selected area white

# === SAVE RESULTS ===
cv2.imwrite("segmented_map.png", cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))

# Optional: Show it
plt.imshow(segmented)
plt.title("Segmented Land Area")
plt.axis("off")
plt.show()
