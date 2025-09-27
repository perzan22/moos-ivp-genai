import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


class SampleCallback(keras.callbacks.Callback):
    def __init__(self, num_classes, latent_dim, outdir="samples\example_cgan_animals"):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.outdir = outdir

        # stały seed na całą sesję (1 próbka, powielimy dla każdej klasy)
        self.seed_noise = tf.random.normal((num_classes, latent_dim))
        self.fixed_labels = tf.range(0, num_classes, dtype=tf.int32)
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(os.path.join(outdir, "models"), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        fig, axes = plt.subplots(1, self.num_classes, figsize=(2*self.num_classes, 2))
        preds = self.model.generator([self.seed_noise, self.fixed_labels], training=False)
        preds = (preds + 1) / 2.0

        for i in range(self.num_classes):
            # rescale z [-1,1] -> [0,1] dla matplotlib
            ax = axes[i]
            ax.imshow(preds[i])
            ax.axis("off")
            ax.set_title(f"class {i}")

        plt.tight_layout()
        save_path = os.path.join(self.outdir, f"epoch_{epoch+1:04d}.png")
        plt.savefig(save_path)
        plt.close(fig)

         # co 50 epok zapis modeli
        if (epoch + 1) % 50 == 0:
            gen_path = os.path.join(self.outdir, "models", f"generator_epoch_{epoch+1}.keras")
            disc_path = os.path.join(self.outdir, "models", f"discriminator_epoch_{epoch+1}.keras")
            self.model.generator.save(gen_path)
            self.model.discriminator.save(disc_path)
            print(f"[Callback] Saved generator and discriminator at epoch {epoch+1}")