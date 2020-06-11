import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import settings
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


class AudioCallback(tf.keras.callbacks.TensorBoard):
    def __init__(self, visualization_freq=1, *args, **kwargs):
        super(AudioCallback, self).__init__(*args, **kwargs)
        self.visualization_freq = visualization_freq

    def on_epoch_end(self, epoch, logs=None):
        super(AudioCallback, self).on_epoch_end(epoch, logs)
        if epoch != 0 and epoch % self.visualization_freq == 0:
            vae = self.model
            for item in vae.get_train_dataset().take(1):
                z_mean, z_log_var, z = vae.encode(item)
                new_item = vae.decode(z)

                writer = self._get_writer(self._train_run_name)
                with writer.as_default():
                    tf.summary.audio(name='x', data=item, sample_rate=16000, step=epoch, max_outputs=5)
                    tf.summary.audio(name='x^', data=new_item, step=epoch, sample_rate=16000, max_outputs=5)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(
        self,
        vae_name,
        image_shape,
        latent_dim,
        database,
        parser,
        encoder,
        decoder,
        visualization_freq,
        learning_rate,
        **kwargs
    ):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.database = database
        self.parser = parser
        self.visualization_freq = visualization_freq
        self.image_shape = image_shape
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.sampler = Sampling()
        self.vae_name = vae_name
        self.encoder = encoder
        self.decoder = decoder

        self.loss_metric = tf.keras.metrics.Mean()
        self.reconstruction_loss_metric = tf.keras.metrics.Mean()
        self.kl_loss_metric = tf.keras.metrics.Mean()

    def get_vae_name(self):
        return self.vae_name

    def sample(self, z_mean, z_log_var):
        return self.sampler((z_mean, z_log_var))

    def encode(self, item):
        z_mean, z_log_var = self.encoder(item)
        z = self.sample(z_mean, z_log_var)
        return z_mean, z_log_var, z

    def decode(self, item):
        return self.decoder(item)

    def call(self, inputs, training=None, mask=None):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampler([z_mean, z_log_var])
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

    def get_dataset(self, partition='train'):
        instances = self.database.get_all_instances(partition_name=partition)
        train_dataset = tf.data.Dataset.from_tensor_slices(instances).shuffle(len(instances))
        train_dataset = train_dataset.map(self.parser.get_parse_fn())
        train_dataset = train_dataset.batch(128)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset

    def get_train_dataset(self):
        return self.get_dataset(partition='train')

    def get_val_dataset(self):
        return self.get_dataset(partition='val')

    def load_latest_checkpoint(self, epoch_to_load_from=None):
        latest_checkpoint = tf.train.latest_checkpoint(
            os.path.join(
                os.path.dirname(sys.argv[0]),
                'vae',
                self.get_vae_name(),
                'vae_checkpoints'
            )
        )

        if latest_checkpoint is not None:
            self.load_weights(latest_checkpoint)
            epoch = int(latest_checkpoint[latest_checkpoint.rfind('_') + 1:])
            return epoch
        return -1

    def perform_training(self, epochs, checkpoint_freq=100, vis_callback_cls=None):
        initial_epoch = self.load_latest_checkpoint()
        if initial_epoch != -1:
            print(f'Continue training from epoch {initial_epoch}.')

        train_dataset = self.get_train_dataset()
        val_dataset = self.get_val_dataset()

        checkpoint_callback = CheckPointFreq(
            freq=checkpoint_freq,
            filepath=os.path.join(
                os.path.dirname(sys.argv[0]),
                'vae',
                self.get_vae_name(),
                'vae_checkpoints',
                'vae_{epoch:02d}'
            ),
            save_freq='epoch',
            save_weights_only=True,
            epochs=epochs - 1
        )
        if vis_callback_cls is None:
            vis_callback_cls = VisualizationCallback

        tensorboard_callback = vis_callback_cls(
            log_dir=os.path.join(
                os.path.dirname(sys.argv[0]),
                'vae',
                self.get_vae_name(),
                'vae_logs'
            ),
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
#                 new_z = new_z[0, ...][tf.newaxis, ...]
                new_z = new_z[0, ...]
                new_zs.append(new_z)

            for i in range(length):
                new_z = z_mean - i / 5 * std
#                 new_z = new_z[0, ...][tf.newaxis, ...]
                new_z = new_z[0, ...]
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

        tf.random.set_seed(None)