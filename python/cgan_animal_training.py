import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from animal_sample_Callback import SampleCallback

from ConditionalGANAnimalModel import ConditionalGANAnimal

# --- hyperparams ---
BATCH_SIZE = 64
EPOCHS = 500
LATENT_DIM = 100
NUM_CLASSES = 10
IMG_SIZE = 32
CHANNELS = 3
TAGS = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# loading dataset 
(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data() 
x_train = (x_train - 127.5) / 127.5 
y_train = y_train.astype("int32")

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) 
dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# === Lossy ===
def discriminator_loss(real, fake):
    bce = keras.losses.BinaryCrossentropy(from_logits=False)
    real_loss = bce(tf.ones_like(real), real)
    fake_loss = bce(tf.zeros_like(fake), fake)
    return real_loss + fake_loss


def generator_loss(fake_preds):
    bce = keras.losses.BinaryCrossentropy(from_logits=False)
    return bce(tf.ones_like(fake_preds), fake_preds)


# === Budowa generatora ===
# def build_generator(latent_dim, num_classes):
#     noise_in = keras.Input(shape=(latent_dim,), name="noise_in")
#     label_in = keras.Input(shape=(1,), dtype="int32", name="label_in")

#     li = layers.Embedding(num_classes, 50)(label_in)
#     li = layers.Dense(8 * 8)(li)
#     li = layers.Reshape((8, 8, 1))(li)

#     n_nodes = 128 * 8 * 8
#     x = layers.Dense(n_nodes)(noise_in)
#     x = layers.LeakyReLU(0.2)(x)
#     x = layers.Reshape((8, 8, 128))(x)

#     x = layers.Concatenate()([x, li])

#     x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
#     x = layers.LeakyReLU(0.2)(x)

#     x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
#     x = layers.LeakyReLU(0.2)(x)

#     out = layers.Conv2D(CHANNELS, (7, 7), activation="tanh", padding="same")(x)

#     return keras.Model([noise_in, label_in], out, name="generator")


# # === Budowa dyskryminatora ===
# def build_discriminator(num_classes):
#     img_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS), name="img_in")
#     label_in = keras.Input(shape=(1,), dtype="int32", name="label_in")

#     li = layers.Embedding(num_classes, 50)(label_in)
#     li = layers.Dense(IMG_SIZE * IMG_SIZE)(li)
#     li = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(li)

#     x = layers.Concatenate()([img_in, li])

#     x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
#     x = layers.LeakyReLU(0.2)(x)

#     x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
#     x = layers.LeakyReLU(0.2)(x)

#     x = layers.Flatten()(x)
#     x = layers.Dropout(0.4)(x)
#     out = layers.Dense(1, activation="sigmoid")(x)

#     return keras.Model([img_in, label_in], out, name="discriminator")

# generator = build_generator(LATENT_DIM, NUM_CLASSES)
# discriminator = build_discriminator(NUM_CLASSES)

generator = keras.models.load_model("samples/example_cgan_animals/models/generator_epoch_200.keras")
discriminator = keras.models.load_model("samples/example_cgan_animals/models/discriminator_epoch_200.keras")

generator.summary()
discriminator.summary()

cgan = ConditionalGANAnimal(
    discriminator=discriminator,
    generator=generator,
    latent_dim=LATENT_DIM,
    num_classes=NUM_CLASSES,
)

cgan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=False),
)

sample_cb = SampleCallback(num_classes=NUM_CLASSES, latent_dim=LATENT_DIM)

cgan.fit(dataset, initial_epoch=200, epochs=EPOCHS, callbacks=[sample_cb])