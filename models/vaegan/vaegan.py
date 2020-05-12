import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from databases import OmniglotDatabase

import matplotlib.pyplot as plt


class CheckPointFreq(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, epochs, freq=1, *args, **kwargs):
        super(CheckPointFreq, self).__init__(*args, **kwargs)
        self.freq = freq
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch != 0 and (epoch + 1) % self.freq == 0:
            super(CheckPointFreq, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        self.epochs_since_last_save = np.inf
        self._save_model(self.epochs, logs)

        super(CheckPointFreq, self).on_train_end(logs)


class VisualizationCallback(tf.keras.callbacks.TensorBoard):
    def __init__(self, visualization_freq=1, *args, **kwargs):
        super(VisualizationCallback, self).__init__(*args, **kwargs)
        self.visualization_freq = visualization_freq

    def on_epoch_end(self, epoch, logs=None):
        super(VisualizationCallback, self).on_epoch_end(epoch, logs)
        if epoch != 0 and epoch % self.visualization_freq == 0:
            vae = self.model
            for item in vae.get_train_dataset().take(1):
                z_mean, z_log_var, z = vae.encode(item)
                new_item = vae.decode(z)

                writer = self._get_writer(self._train_run_name)
                with writer.as_default():
                    tf.summary.image(name='x', data=item, step=epoch, max_outputs=5)
                    tf.summary.image(name='x^', data=new_item, step=epoch, max_outputs=5)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, latent_dim, database, visualization_freq, learning_rate=0.001, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.database = database
        self.visualization_freq = visualization_freq
        self.image_shape = None
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.encoder, self.sampler = self.get_encoder()
        self.decoder = self.get_decoder()

        self.loss_metric = tf.keras.metrics.Mean()
        self.reconstruction_loss_metric = tf.keras.metrics.Mean()
        self.kl_loss_metric = tf.keras.metrics.Mean()

    def get_encoder(self):
        self.image_shape = (28, 28, 1)
        encoder_inputs = keras.Input(shape=(28, 28, 1))
        x = layers.Conv2D(64, 3, activation=None, strides=2, padding="same", use_bias=False)(encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(64, 3, activation=None, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(64, 3, activation=None, strides=1, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(64, 3, activation=None, strides=1, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Flatten()(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        sampler = Sampling()
        z = sampler([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        return encoder, sampler

    def get_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation=None, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2DTranspose(64, 3, activation=None, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2DTranspose(64, 3, activation=None, strides=1, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2DTranspose(64, 3, activation=None, strides=1, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        return decoder

    def encode(self, item):
        return self.encoder.predict(item)

    def decode(self, item):
        return self.decoder.predict(item)

    def call(self, inputs, training=None, mask=None):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(inputs, reconstruction)
        )
        reconstruction_loss *= np.prod(self.image_shape)
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        outputs = self.call(data)
        self.loss_metric.update_state(outputs['loss'])
        self.reconstruction_loss_metric.update_state(outputs['reconstruction_loss'])
        self.kl_loss_metric.update_state(outputs['kl_loss'])
        return outputs

    def train_step(self, data):
        with tf.GradientTape() as tape:
            outputs = self.call(data)

        grads = tape.gradient(outputs['loss'], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_metric.update_state(outputs['loss'])
        self.reconstruction_loss_metric.update_state(outputs['reconstruction_loss'])
        self.kl_loss_metric.update_state(outputs['kl_loss'])

        return outputs

    def get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_png(tf.io.read_file(example_address))
            image = tf.reshape(tf.image.resize(image, (28, 28)), (28, 28, 1))
            image = tf.cast(image, tf.float32)

            return image / 255.
        return parse_function

    def get_dataset(self, partition='train'):
        instances = self.database.get_all_instances(partition_name=partition)
        train_dataset = tf.data.Dataset.from_tensor_slices(instances).shuffle(len(self.database.train_folders))
        train_dataset = train_dataset.map(self.get_parse_function())
        train_dataset = train_dataset.batch(128)
        return train_dataset

    def get_train_dataset(self):
        return self.get_dataset(partition='train')

    def get_val_dataset(self):
        return self.get_dataset(partition='val')

    def load_latest_checkpoint(self, epoch_to_load_from=None):
        latest_checkpoint = tf.train.latest_checkpoint('./vaegan_checkpoints/')

        if latest_checkpoint is not None:
            self.load_weights(latest_checkpoint)
            epoch = int(latest_checkpoint[latest_checkpoint.rfind('_') + 1:])
            return epoch

        return -1

    def perform_training(self, epochs, checkpoint_freq=100):
        initial_epoch = self.load_latest_checkpoint()
        if initial_epoch != -1:
            print(f'Continue training from epoch {initial_epoch}.')

        train_dataset = self.get_train_dataset()
        val_dataset = self.get_val_dataset()

        checkpoint_callback = CheckPointFreq(
            freq=checkpoint_freq,
            filepath='./vaegan_checkpoints/vaegan_{epoch:02d}',
            save_freq='epoch',
            save_weights_only=True,
            epochs=epochs - 1
        )
        tensorboard_callback = VisualizationCallback(
            log_dir='./vaegan_logs/',
            visualization_freq=self.visualization_freq
        )

        callbacks = [tensorboard_callback, checkpoint_callback]

        self.compile(optimizer=self.optimizer)
        self.fit(
            train_dataset,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_dataset,
            initial_epoch=initial_epoch
        )

    def visualize_meta_learning_task2(self):
        tf.random.set_seed(10)
        for item in self.get_train_dataset().take(1):
            z_mean, z_log_var, z = self.encode(item)
            fig, axes = plt.subplots(1, 6)
            fig.set_figwidth(6)
            fig.set_figheight(1)

            axes[0].imshow(item[0, ..., 0], cmap='gray')
            for i in range(1, 6):
                axes[i].imshow(self.decode(z + tf.random.normal(shape=z.shape, stddev=0.2 * i))[0, ..., 0], cmap='gray')
                axes[i].set_xlabel(f'noise stddev: {0.2 * i:0.2f}', size='xx-small')

            plt.show()

    def visualize_meta_learning_task(self):
        tf.random.set_seed(10)
        for item in self.get_train_dataset().take(1):
            z_mean, z_log_var, z = self.encode(item)
            new_item = self.decode(z)

            std = tf.exp(0.5 * z_log_var)
            std = 1 / tf.nn.softmax(std) * std

            new_zs = list()
            length = 15
            for i in range(length):
                new_z = z_mean + i / 5 * std
                new_z = new_z[0, ...][tf.newaxis, ...]
                new_zs.append(new_z)

            for i in range(length):
                new_z = z_mean - i / 5 * std
                new_z = new_z[0, ...][tf.newaxis, ...]
                new_zs.append(new_z)

            fig, axes = plt.subplots(length + 1, 2)
            fig.set_figwidth(2)
            fig.set_figheight(length + 1)

            axes[0, 0].imshow(item[0, ..., 0], cmap='gray')
            axes[0, 0].set_xlabel('Real image', size='xx-small')
            axes[0, 1].imshow(new_item[0, ..., 0], cmap='gray')
            axes[0, 1].set_xlabel('Reconstruction', size='xx-small')
            for i in range(1, length + 1):
                new_item = self.decode(new_zs[i - 1][tf.newaxis, ...])
                axes[i, 0].imshow(new_item[0, ..., 0], cmap='gray')
                axes[i, 0].set_xlabel(f'mean + {i / 5} * std', size='xx-small')

                new_item = self.decode(new_zs[length + i - 1][tf.newaxis, ...])
                axes[i, 1].imshow(new_item[0, ..., 0], cmap='gray')
                axes[i, 1].set_xlabel(f'mean - {i / 5} * std', size='xx-small')

            plt.show()


def run_vae_gan():
    database = OmniglotDatabase(random_seed=47, num_train_classes=1200, num_val_classes=100)
    latent_dim = 20

    vae = VAE(
        latent_dim=latent_dim,
        database=database,
        visualization_freq=5,
        learning_rate=0.001,
    )
    vae.perform_training(epochs=1000, checkpoint_freq=100)

    vae.load_latest_checkpoint()
    vae.visualize_meta_learning_task()


if __name__ == '__main__':
    run_vae_gan()
