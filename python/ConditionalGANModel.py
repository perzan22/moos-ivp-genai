import os, keras
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import ops

IMG_SIZE = 128
IMAGE_CHANNELS = 3
LATENT_DIM = 100
CONDITION_DIM = 8


class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.rng = tf.random.Generator.from_seed(1337)
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.lambda_l1 = 10.0

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]
    
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, cond = data
        batch_size = tf.shape(real_images)[0]

        # === Train discriminator 1 time ===
        # produce fake images
        z = self.rng.normal((batch_size, self.latent_dim))
        generated_images = self.generator([z, cond], training=True)

        # labels: fake=0, real=1
        fake_labels = tf.zeros((batch_size, 1)) + 0.1
        real_labels = tf.ones((batch_size, 1)) * 0.9

        with tf.GradientTape() as tape:
            preds_real = self.discriminator([real_images, cond], training=True)
            preds_fake = self.discriminator([generated_images, cond], training=True)
            d_loss_real = self.loss_fn(real_labels, preds_real)
            d_loss_fake = self.loss_fn(fake_labels, preds_fake)
            d_loss = d_loss_real + d_loss_fake

        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        d_real_mean = tf.reduce_mean(preds_real)
        d_fake_mean = tf.reduce_mean(preds_fake)

        # === Train generator 2 times ===
        g_losses = []
        for _ in range(2):
            z = self.rng.normal((batch_size, LATENT_DIM))
            misleading_labels = tf.ones((batch_size, 1))  # generator wants D to output "real"

            with tf.GradientTape() as tape:
                fake_images = self.generator([z, cond], training=True)
                preds = self.discriminator([fake_images, cond], training=True)
                adv_loss = self.loss_fn(misleading_labels, preds)

                # L1 (pixel-wise) loss
                l1_loss = tf.reduce_mean(tf.abs(real_images - fake_images))

                g_loss = adv_loss + self.lambda_l1 * l1_loss

            grads = tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
            g_losses.append(g_loss)

        g_fake_mean = tf.reduce_mean(preds)
        g_loss = tf.reduce_mean(g_losses)

        # update metrics
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
            "D(real)": d_real_mean,
            "D(fake)": d_fake_mean,
            "G(fake)": g_fake_mean,
            "L1": l1_loss
        }

