import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from databases import VoxCelebDatabase

from models.lasiummamlvae.database_parsers import VoxCelebParser
from models.lasiummamlvae.maml_vae import MAML_VAE
from models.lasiummamlvae.vae import VAE, AudioCallback
from networks.maml_umtra_networks import SimpleModel


class Conv1DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid', use_bias=True):
        super().__init__()
        self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(
            filters, (kernel_size, 1), (strides, 1), padding, use_bias=use_bias
        )

    def call(self, x, *args, **kwargs):
        x = tf.expand_dims(x, axis=2)
        x = self.conv2dtranspose(x)
        x = tf.squeeze(x, axis=2)
        return x


def get_encoder(latent_dim):
    encoder_inputs = keras.Input(shape=(16000, 1))
    x = layers.Conv1D(64, 3, activation=None, strides=16, padding="same", use_bias=False)(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(64, 3, activation=None, strides=16, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(64, 3, activation=None, strides=8, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(64, 3, activation=None, strides=8, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

    encoder.summary()
    return encoder


def get_decoder(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(125 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((125, 64))(x)
    x = Conv1DTranspose(64, 3, strides=4, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = Conv1DTranspose(64, 3, strides=4, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = Conv1DTranspose(64, 3, strides=4, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = Conv1DTranspose(64, 3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = Conv1DTranspose(1, 3, padding="same")(x)
    decoder_outputs = layers.Activation('tanh')(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    return decoder


if __name__ == '__main__':
    # import tensorflow as tf
    # tf.config.experimental_run_functions_eagerly(True)

    voxceleb_database = VoxCelebDatabase()
    shape = (16000, 1)
    latent_dim = 20
    voxceleb_encoder = get_encoder(latent_dim)
    voxceleb_decoder = get_decoder(latent_dim)
    voxceleb_parser = VoxCelebParser(shape=shape)

    vae = VAE(
        'voxceleb',
        image_shape=shape,
        latent_dim=latent_dim,
        database=voxceleb_database,
        parser=voxceleb_parser,
        encoder=voxceleb_encoder,
        decoder=voxceleb_decoder,
        visualization_freq=1,
        learning_rate=0.001,
    )
    vae.perform_training(epochs=1000, checkpoint_freq=100, vis_callback_cls=AudioCallback)
    vae.load_latest_checkpoint()
    # vae.visualize_meta_learning_task()

    maml_vae = MAML_VAE(
        vae=vae,
        database=voxceleb_database,
        network_cls=SimpleModel,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=5,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.4,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=False,
        experiment_name='voxceleb_std_1.0',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    maml_vae.visualize_meta_learning_task(shape, num_tasks_to_visualize=2)

    maml_vae.train(iterations=5000)
    maml_vae.evaluate(50, seed=42)
