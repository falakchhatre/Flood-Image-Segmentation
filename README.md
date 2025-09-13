# Flood Image Segmentation  

Floods are among the most common natural disasters, causing widespread damage to infrastructure and human lives. Effective monitoring and early detection are crucial to minimize the risks associated with flooding. Traditional methods, such as satellite imagery or ground sensors, are often expensive, time-consuming, and limited in coverage.

This project explores the use of **deep learning for real-time flood detection** using **semantic segmentation**. The model can automatically identify water in images captured by drones, video surveillance, or social media, providing a pathway toward accessible and efficient flood monitoring.

---

## Project Overview

The main goal is to build an **efficient, accurate, and accessible flood segmentation model**. Key steps include:

- Combining two public datasets into a **sample dataset** of ~439 images.
- Splitting data into training and testing sets.
- Applying **image augmentation** (vertical flip, horizontal flip, zoom) to improve generalization due to limited dataset size.
- Evaluating different U-Net backbones (MobileNet vs. ResNet) for performance, accuracy, and computational efficiency.
- Using a **MobileNet encoder + Pix2Pix decoder** for lightweight, real-time segmentation.

**Note:** The current dataset is **not complete**, and results are indicative. The model is primarily for prototyping and experimentation.

---

## Key Findings

- MobileNet backbone achieved **better pixel accuracy** than ResNet on this sample dataset, despite having fewer parameters.
- MobileNet is suitable for resource-constrained platforms: faster training, lower memory usage, and reduced risk of overfitting.
- Data augmentation improves segmentation consistency, especially with limited data.
- Overall, **MobileNet-based U-Net shows promise for real-time flood detection and monitoring**.

---

## How to Run Locally

### Requirements
- Python 3.9+
- TensorFlow 2.14+
- OpenCV, NumPy, Matplotlib
- Optional: GPU for faster training

### Setup
```bash
git clone https://github.com/falakchhatre/Flood-Image-Segmentation.git
cd Flood-Image-Segmentation
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt
```
## Training

1. Place your images and masks in `data/images` and `data/masks`.
2. Run the training script:
```bash
python unet_training.py
```

### Outputs

- Trained models are saved in the `models/` directory.
- Sample predicted masks are saved in the `results/` directory.

> Note: The current dataset is small (sample subset provided). Accuracy and predictions are limited and meant for demonstration purposes. Full performance requires a complete dataset.

---

## Training Details

- **Model:** U-Net with MobileNet encoder and Pix2Pix decoder  
- **Loss function:** Sparse Categorical Crossentropy  
- **Metrics:** Accuracy  
- **Batch size:** 16  
- **Epochs:** 50 (training may stop earlier with early stopping)  
- **Augmentation:** Vertical flip, horizontal flip, zoom  

---

## Results & Visualizations
<img width="555" height="540" alt="Untitled design (14)" src="https://github.com/user-attachments/assets/854bbcba-dfee-4905-90f8-07fd6b30e4d8" />
<img width="1005" height="500" alt="Figure 2" src="https://github.com/user-attachments/assets/54c661b9-b308-4017-8766-4f61d2c2ac8f" />

---
## Future Work

- Expand the dataset for better generalization and real-world performance.  
- Explore additional backbone architectures and optimization strategies.  
- Develop a real-time web or mobile application for live flood monitoring.  
- Improve visualization and result exporting features.

---
