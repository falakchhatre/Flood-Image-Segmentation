from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from model import preprocess_image, predict_mask
from PIL import Image
import numpy as np
import os
from uuid import uuid4


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
        # read uploaded file
        img_bytes = await file.read()

        # preprocess for model
        img_array = preprocess_image(img_bytes)

        # run model to get mask (0/1 per pixel)
        mask = predict_mask(img_array)

        # convert predicted mask to grayscale (0 or 255)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))        

        # convert normalized input back to image
        input_img = (img_array[0] * 255).astype(np.uint8)
        input_img_pil = Image.fromarray(input_img)

        # generate unique filenames
        input_filename = f"input_{uuid4().hex}.png"
        mask_filename  = f"mask_{uuid4().hex}.png"

        # build paths for saving images
        input_path = os.path.join("static", input_filename)
        mask_path  = os.path.join("static", mask_filename)

        # save output images
        input_img_pil.save(input_path)
        mask_img.save(mask_path)

        # send file paths back to frontend
        return {
            "input": f"static/{input_filename}",
            "mask": f"static/{mask_filename}",
        }

    except Exception as e:
        # return error message if something goes wrong
        return {"error": str(e)}
