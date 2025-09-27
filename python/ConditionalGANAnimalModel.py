import tensorflow as tf
from tensorflow import keras

class ConditionalGANAnimal(keras.Model):
    def __init__(self, generator, discriminator, latent_dim, num_classes):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # rng for noise
        self.rng = tf.random.Generator.from_seed(1337)

        # metrics
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")


    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn


    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]


    def train_step(self, data):
        real_images, real_labels = data
        batch_size = tf.shape(real_images)[0]


        # === Train discriminator ===
        z = self.rng.normal((batch_size, self.latent_dim))
        fake_images = self.generator([z, real_labels], training=True)

        fake_labels = tf.zeros((batch_size, 1))
        real_labels_target = tf.ones((batch_size, 1))


        with tf.GradientTape() as tape:
            pred_fake = self.discriminator([fake_images, real_labels], training=True)
            pred_real = self.discriminator([real_images, real_labels], training=True)
            d_loss_real = self.loss_fn(real_labels_target, pred_real)
            d_loss_fake = self.loss_fn(fake_labels, pred_fake)
            d_loss = d_loss_real + d_loss_fake


        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))


        # === Trening generatora ===
        z = self.rng.normal((batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            fake_images = self.generator([z, real_labels], training=True)
            pred_fake = self.discriminator([fake_images, real_labels], training=True)
            g_loss = self.loss_fn(misleading_labels, pred_fake)


        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))


        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)


        return {"g_loss": self.gen_loss_tracker.result(), "d_loss": self.disc_loss_tracker.result()}