from model import preprocess_image, model  # import your functions and model

# Replace with the path to one of your training or test images
image_path = "../../data/images/60.jpg"  

# Read and preprocess
with open(image_path, "rb") as f:
    img_bytes = f.read()
img_array = preprocess_image(img_bytes)

# Run prediction
pred = model.predict(img_array)

print("Prediction stats:")
print("min:", pred.min())
print("max:", pred.max())
print("shape:", pred.shape)
