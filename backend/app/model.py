import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import os

IMG_HEIGHT, IMG_WIDTH = 224, 224
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "unet_model_final.h5")

print("Loading model from:", MODEL_PATH)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(img_bytes: bytes) -> np.ndarray:
    """
    Convert uploaded image bytes to normalized model input.
    """
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def predict_mask(img_array: np.ndarray) -> np.ndarray:
    """
    Predict mask and return it as a NumPy array (0 or 1 per pixel).
    """
    pred = model.predict(img_array)

    # debug information
    print("Pred shape:", pred.shape)
    print("Pred min/max:", pred.min(), pred.max())

    # multi-class segmentation (2 logit channels)
    mask = tf.math.argmax(pred, axis=-1)[0].numpy().astype(np.uint8)
    print("Mask unique values:", np.unique(mask))
    return mask