# unet_training.py
import os
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Paths
images_path = "data/images"    # raw images 
masks_path = "data/masks"      # corresponding masks
results_path = "results"       # folder to save prediction images
models_path = "models"         # folder to save trained models

os.makedirs(results_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# List images and split into train/test
images_dir = sorted(os.listdir(images_path))
masks_dir = sorted(os.listdir(masks_path))
assert len(images_dir) == len(masks_dir), "Number of images and masks do not match!"

train_images, test_images, train_masks, test_masks = train_test_split(
    images_dir, masks_dir, test_size=0.2, random_state=42
)

# Helper functions
IMG_HEIGHT = 224
IMG_WIDTH = 224

def read_image(path):
    x = cv2.imread(path)
    if x is None:
        raise ValueError(f"Cannot load image: {path}")
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (IMG_WIDTH, IMG_HEIGHT))
    x = x / 255.0
    return x.astype(np.float32)

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if x is None:
        raise ValueError(f"Cannot load mask: {path}")
    x = cv2.resize(x, (IMG_WIDTH, IMG_HEIGHT))
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)
    return x.astype(np.float32)

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        return read_image(os.path.join(images_path, x)), read_mask(os.path.join(masks_path, y))
    images, masks = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    images.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    masks.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])
    return images, masks

# Create datasets
train_data = tf.data.Dataset.from_tensor_slices((train_images, train_masks)).map(preprocess)
test_data = tf.data.Dataset.from_tensor_slices((test_images, test_masks)).map(preprocess)

# Data augmentation
class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=20):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed)
    def call(self, inputs, labels):
        return self.augment_inputs(inputs), self.augment_labels(labels)

# Build MobileNetV2 U-Net model
base_model = tf.keras.applications.MobileNetV2(input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], include_top=False)
layer_names = ['block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu',
               'block_13_expand_relu', 'block_16_project']
down_stack = tf.keras.Model(inputs=base_model.input, outputs=[base_model.get_layer(name).output for name in layer_names])
down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),
    pix2pix.upsample(256, 3),
    pix2pix.upsample(128, 3),
    pix2pix.upsample(64, 3),
]

def unet_model(output_channels, down_stack):
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

model = unet_model(output_channels=2, down_stack=down_stack)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Training parameters
epochs = 50
batch_size = 16
buffer_size = 500
steps_per_epoch = len(train_images) // batch_size
validation_steps = len(test_images) // batch_size

train_batches = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat().map(Augment()).prefetch(tf.data.AUTOTUNE)
test_batches = test_data.batch(batch_size)

# Callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(models_path, "unet_model_best.h5"), save_best_only=True)
earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='auto')

# Train model
history = model.fit(train_batches, epochs=epochs, steps_per_epoch=steps_per_epoch,
                    validation_data=test_batches, validation_steps=validation_steps,
                    callbacks=[checkpoint_cb, earlystop_cb])

# Save final model
model.save(os.path.join(models_path, "unet_model_final.h5"))

# Make predictions and save sample results
def create_mask(pred_mask):
    return tf.math.argmax(pred_mask, axis=-1)[..., tf.newaxis]

def save_sample_predictions(n=3):
    for i in range(n):
        idx = random.randint(0, len(images_dir)-1)
        img_path = os.path.join(images_path, images_dir[idx])
        mask_path = os.path.join(masks_path, masks_dir[idx])
        img = read_image(img_path)
        mask = read_mask(mask_path)
        pred_mask = create_mask(model.predict(img[tf.newaxis, ...]))
        plt.figure(figsize=(10,10))
        plt.subplot(1,3,1); plt.imshow(img); plt.title("Image"); plt.axis("off")
        plt.subplot(1,3,2); plt.imshow(mask[:,:,0], cmap='gray'); plt.title("Mask"); plt.axis("off")
        plt.subplot(1,3,3); plt.imshow(pred_mask[0,:,:,0], cmap='gray'); plt.title("Predicted"); plt.axis("off")
        plt.savefig(os.path.join(results_path, f"prediction_{i}.png"))
        plt.close()

save_sample_predictions()