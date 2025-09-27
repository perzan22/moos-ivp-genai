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


IMG_SIZE = 128
BATCH_SIZE = 64
IMAGE_CHANNELS = 1
LATENT_DIM = 100
CONDITION_DIM = 8

def parse_waypoints(wp_str):
    parts = wp_str.split(" : ")
    vals = []
    for p in parts:
        x, y = map(float, p.split(","))
        vals.extend([x, y])
    return vals

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

# # choose condition columns from manifest
# cond_cols = ["drift_angle", "drift_speed", "yaw_kp", "yaw_ki", "yaw_kd", "speed"]
starts = df[["start_x", "start_y"]].values.astype(np.float32)
waypoints = np.array([parse_waypoints(w) for w in df["waypoints"]], dtype=np.float32)

# normalize condition values from -1 to 1
# y_cond_raw = df[cond_cols].values.astype(np.float32)
y_cond_raw = np.concatenate([starts, waypoints], axis=1)  # shape (N, 8)
scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
y_cond = scaler.fit_transform(y_cond_raw).astype(np.float32)

# create dataset with images and conditions for cGAN training
dataset = tf.data.Dataset.from_tensor_slices((X, y_cond))
dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# create condition embeddings
# img_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS), name="img_in")
# cond_in = keras.Input(shape=(CONDITION_DIM,), name="cond_in")

# cond_emb = layers.Dense(128, activation="relu")(cond_in)

# # create discriminator

# x = layers.Conv2D(64, 4, strides=2, padding="same")(img_in)
# x = layers.LeakyReLU(0.2)(x)

# x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
# x = layers.BatchNormalization()(x)
# x = layers.LeakyReLU(0.2)(x)

# x = layers.Conv2D(256, 4, strides=2, padding="same")(x)
# x = layers.BatchNormalization()(x)
# x = layers.LeakyReLU(0.2)(x)
# x = layers.Dropout(0.4)(x)

# x = layers.Conv2D(512, 4, strides=2, padding="same")(x)
# x = layers.BatchNormalization()(x)
# x = layers.LeakyReLU(0.2)(x)
# x = layers.Dropout(0.4)(x)

# x = layers.Flatten()(x)

# # concatenate conditions and image
# x = layers.Concatenate()([x, cond_emb])

# out = layers.Dense(1, activation="sigmoid")(x)

# discriminator = keras.Model([img_in, cond_in], out, name="discriminator")


# # generator latent input

# z_in = keras.Input(shape=(LATENT_DIM,), name="z_in")

# gen_in = layers.Concatenate()([z_in, cond_emb])

# # create generator core
# generator_core = keras.Sequential(
#     [
#         layers.Dense(8 * 8 * 512, use_bias=False, kernel_initializer="he_normal"),
#         layers.BatchNormalization(),
#         layers.ReLU(),
#         layers.Reshape((8, 8, 512)),

#         layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding="same", use_bias=False),
#         layers.BatchNormalization(),
#         layers.ReLU(),

#         layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", use_bias=False),
#         layers.BatchNormalization(),
#         layers.ReLU(),

#         layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", use_bias=False),
#         layers.BatchNormalization(),
#         layers.ReLU(),

#         layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding="same", use_bias=False),
#         layers.BatchNormalization(),
#         layers.ReLU(),

#         layers.Conv2D(1, (7, 7), padding="same", activation="tanh", use_bias=False),
#     ],
#     name="generator_core"
# )

# gen_out = generator_core(gen_in)
# generator = keras.Model([z_in, cond_in], gen_out, name="generator")

generator = keras.models.load_model("samples/grayscale_waypoint_conds/model/generator_epoch_250.keras")
discriminator = keras.models.load_model("samples/grayscale_waypoint_conds/model/discriminator_epoch_250.keras")

generator.summary()
discriminator.summary()

cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM
)

cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=False),
)

# prepare fixed seed (for visualization)
NUM_SAMPLE = 5
rng = tf.random.Generator.from_seed(42)
seed_noise = rng.normal((NUM_SAMPLE, LATENT_DIM), dtype=tf.float32)
# pick some random condition samples from dataset (or create a grid)
seed_conditions = tf.convert_to_tensor(y_cond[:NUM_SAMPLE], dtype=tf.float32)
sample_cb = SampleCallback(seed_noise, seed_conditions)

# train cgan
cond_gan.fit(dataset, initial_epoch=250, epochs=700, callbacks=[sample_cb])

