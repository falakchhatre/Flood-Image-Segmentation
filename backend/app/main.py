from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from model import preprocess_image, predict_mask
from PIL import Image
import numpy as np
import os

# Instantiate app
app = FastAPI(title="Flood Segmentation API")

# Enable CORS so React frontend can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Make sure static folder exists
os.makedirs("static", exist_ok=True)

# Mount static folder to serve images
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return {"message": "Flood Segmentation API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img_array = preprocess_image(img_bytes)
        mask = predict_mask(img_array)

        # Convert mask to RGB
        mask_rgb = np.stack([mask]*3, axis=-1)
        mask_img = Image.fromarray(mask_rgb.astype(np.uint8))

        # Convert preprocessed input to image
        input_img = (img_array[0] * 255).astype(np.uint8)
        input_img_pil = Image.fromarray(input_img)

        # Save both images
        input_path = "static/input.png"
        mask_path = "static/result.png"
        input_img_pil.save(input_path)
        mask_img.save(mask_path)

        return {"input": input_path, "mask": mask_path}

    except Exception as e:
        return {"error": str(e)}

