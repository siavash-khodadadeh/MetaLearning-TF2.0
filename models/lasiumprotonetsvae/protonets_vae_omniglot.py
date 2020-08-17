from tensorflow import keras
from tensorflow.keras import layers

from databases import OmniglotDatabase

from models.lasiumprotonetsvae.database_parsers import OmniglotParser
from models.lasiumprotonetsvae.protonets_vae import ProtoNetsVAE
from models.lasiumprotonetsvae.vae import VAE
from networks.proto_networks import SimpleModelProto, VGGSmallModel

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 4.5)])
    except RuntimeError as e:
        print(e)

def get_encoder(latent_dim):
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
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

    return encoder


def get_decoder(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
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


if __name__ == '__main__':
    # import tensorflow as tf
    # tf.config.experimental_run_functions_eagerly(True)

    omniglot_database = OmniglotDatabase(random_seed=47, num_train_classes=1200, num_val_classes=100)
    shape = (28, 28, 1)
    latent_dim = 20
    omniglot_encoder = get_encoder(latent_dim)
    omniglot_decoder = get_decoder(latent_dim)
    omniglot_parser = OmniglotParser(shape=shape)

    vae = VAE(
        'omniglot',
        image_shape=shape,
        latent_dim=latent_dim,
        database=omniglot_database,
        parser=omniglot_parser,
        encoder=omniglot_encoder,
        decoder=omniglot_decoder,
        visualization_freq=5,
        learning_rate=0.001,
    )
    vae.perform_training(epochs=1000, checkpoint_freq=100)
    vae.load_latest_checkpoint()
    vae.visualize_meta_learning_task()

    proto_vae = ProtoNetsVAE(
        vae=vae,
        latent_algorithm='p2',
        database=omniglot_database,
        network_cls=SimpleModelProto,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_train=None,
        k_val_val=15,
        k_val_test=15,
        k_test=1,
        meta_batch_size=4,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        experiment_name='proto_vae_omniglot_shift_0.4',
        val_seed=42
    )

    proto_vae.visualize_meta_learning_task(shape, num_tasks_to_visualize=2)

    proto_vae.train(iterations=8000)
    proto_vae.evaluate(-1, seed=42)