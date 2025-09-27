import tensorflow as tf
from tensorflow import keras

class ConditionalWGAN_GP(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, gp_weight=10.0, n_critic=5):
        super().__init__()
        self.discriminator = discriminator  # critic, outputs scalar (no activation)
        self.generator = generator
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.n_critic = n_critic
        self.lambda_l1 = 0.5

        self.rng = tf.random.Generator.from_non_deterministic_state()
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def gradient_penalty(self, real_images, fake_images, cond):
        # real_images, fake_images: (B,H,W,C), cond: (B,COND)
        batch_size = tf.shape(real_images)[0]
        # sample epsilon
        eps = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        # interpolate
        interp = eps * real_images + (1.0 - eps) * fake_images
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interp)
            # critic score for interpolates
            pred = self.discriminator([interp, cond], training=True)
        grads = gp_tape.gradient(pred, interp)  # shape (B,H,W,C)
        grads = tf.reshape(grads, [batch_size, -1])
        grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
        gp = tf.reduce_mean((grads_norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_images, cond = data
        batch_size = tf.shape(real_images)[0]

        # === Train discriminator n_critic times ===
        d_loss_total = 0.0
        for i in range(self.n_critic):
            z = self.rng.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_images = self.generator([z, cond], training=True)
                real_score = self.discriminator([real_images, cond], training=True)
                fake_score = self.discriminator([fake_images, cond], training=True)

                # Wasserstein critic loss: E[fake] - E[real]
                d_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)

                # gradient penalty
                gp = self.gradient_penalty(real_images, fake_images, cond)
                d_loss += self.gp_weight * gp

            grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
            d_loss_total += d_loss

        d_loss_total /= tf.cast(self.n_critic, tf.float32)

        # === Train generator (1 step) ===
        z = self.rng.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            fake_images = self.generator([z, cond], training=True)
            fake_score_for_g = self.discriminator([fake_images, cond], training=True)
            # generator wants to maximize E[critic(fake)] -> minimize -E[critic(fake)] add l1 loss [pixel-wise] function with small lambda
            l1_loss = tf.reduce_mean(tf.abs(real_images - fake_images))
            g_loss = -tf.reduce_mean(fake_score_for_g) + self.lambda_l1 * l1_loss

        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        # update metrics
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss_total)

        # for monitoring, compute scalars
        d_real_mean = tf.reduce_mean(self.discriminator([real_images, cond], training=False))
        d_fake_mean = tf.reduce_mean(self.discriminator([fake_images, cond], training=False))
        g_fake_mean = d_fake_mean

        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
            "D(real)": d_real_mean,
            "D(fake)": d_fake_mean,
            "G(fake)": g_fake_mean,
        }