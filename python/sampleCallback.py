# Callback: generate samples on epoch end for monitoring
import matplotlib.pyplot as plt
import os, keras
import tensorflow as tf
import numpy as np


class SampleCallback(keras.callbacks.Callback):
    def __init__(self, seed_noise, seed_conditions, out_dir="samples\grayscale_waypoint_conds"):
        self.seed_noise = seed_noise
        self.seed_conditions = seed_conditions
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

    def on_epoch_end(self, epoch, logs=None):
        z = self.seed_noise
        cond = self.seed_conditions
        imgs = self.model.generator([z, cond], training=False)  # in [-1,1]
        imgs = (imgs + 1.0) * 127.5
        imgs = tf.clip_by_value(imgs, 0, 255).numpy().astype(np.uint8)
        # grid save
        n = imgs.shape[0]
        fig, axs = plt.subplots(1, n, figsize=(n*2,2))
        for i in range(n):
            axs[i].imshow(imgs[i].squeeze(), cmap="gray", vmin=0, vmax=255)
            axs[i].axis("off")
        fig.suptitle(f"epoch {epoch+1}")
        plt.savefig(os.path.join(self.out_dir, f"epoch_{epoch+1:03d}.png"))
        plt.close(fig)

        if (epoch + 1) % 50 == 0:
            self.model.generator.save(os.path.join(self.out_dir, "model", f"generator_epoch_{epoch+1}.keras"))
            self.model.discriminator.save(os.path.join(self.out_dir, "model", f"discriminator_epoch_{epoch+1}.keras"))
            print(f"Saved models at epoch {epoch+1}")