import os, keras
import pandas as pd
import numpy as np
import tensorflow as tf
from ConditionalGANModel import ConditionalGAN
from ConditionalWGAN_GP import ConditionalWGAN_GP
from sampleCallback import SampleCallback
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras import ops
import matplotlib.pyplot as plt


IMG_SIZE = 128
BATCH_SIZE = 64
IMAGE_CHANNELS = 1
LATENT_DIM = 100
CONDITION_DIM = 6

# function to load and preprocess images so it is 128x128 size in -1 to 1 scale
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=1)
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    image = image / 127.5 - 1.0  # [-1,1]
    return image


# reading manifest file with runs conditions and image paths
manifest_path = "dataset/manifest.csv"
df = pd.read_csv(manifest_path)

image_paths = df["traj_img"].tolist()

# loading and preprocessing images
X = np.array([load_and_preprocess_image(image) for image in image_paths])

# choose condition columns from manifest
cond_cols = ["drift_angle", "drift_speed", "yaw_kp", "yaw_ki", "yaw_kd", "speed"]

# normalize condition values from 0 to 1
y_cond_raw = df[cond_cols].values.astype(np.float32)
scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
y_cond = scaler.fit_transform(y_cond_raw).astype(np.float32)

# create dataset with images and conditions for cGAN training
dataset = tf.data.Dataset.from_tensor_slices((X, y_cond))
dataset = dataset.batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# Pobierz kilka pierwszych próbek
for images, conds in dataset.take(1):  
    images = images[:5]
    conds = conds[:5]

    # Pokaż obrazki
    plt.figure(figsize=(15, 3))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        # Skala z [-1,1] → [0,1] do wyświetlenia
        img = (images[i].numpy() + 1.0) / 2.0  
        plt.imshow(img.squeeze(), cmap="gray")  # grayscale
        plt.axis("off")
        plt.title(f"cond: {conds[i].numpy()}")
    plt.show()

    # Dodatkowo print samego warunku, żeby było czytelniej
    print("Conditions (first batch):")
    print(y_cond_raw[:5])