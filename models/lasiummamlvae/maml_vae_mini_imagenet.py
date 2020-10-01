import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from databases import CelebADatabase, MiniImagenetDatabase
from models.lasiummamlvae.database_parsers import MiniImagenetParser

from models.lasiummamlvae.database_parsers import CelebAParser
from models.lasiummamlvae.maml_vae import MAML_VAE
from models.lasiummamlvae.vae import VAE, AudioCallback
from networks.maml_umtra_networks import SimpleModel, MiniImagenetModel


def get_encoder(latent_dim):
    encoder_inputs = keras.Input(shape=(84, 84, 3))
    x = layers.Conv2D(64, 4, activation=None, strides=2, padding="same", use_bias=False)(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, 4, activation=None, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, 4, activation=None, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, 4, activation=None, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(512, 4, activation=None, strides=2, padding="same", use_bias=False)(x)
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
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(512, 4, activation=None, strides=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(256, 4, activation=None, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(256, 4, activation=None, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, 4, activation=None, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 4, activation=None, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    decoder_outputs = layers.Conv2DTranspose(3, 4, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    return decoder


if __name__ == '__main__':
    # import tensorflow as tf
    # tf.config.experimental_run_functions_eagerly(True)

    mini_imagenet_database = MiniImagenetDatabase()
    shape = (84, 84, 3)
    latent_dim = 512
    mini_imagenet_encoder = get_encoder(latent_dim)
    mini_imagenet_decoder = get_decoder(latent_dim)
    mini_imagenet_parser = MiniImagenetParser(shape=shape)

    vae = VAE(
        'mini-imagenet',
        image_shape=shape,
        latent_dim=latent_dim,
        database=mini_imagenet_database,
        parser=mini_imagenet_parser,
        encoder=mini_imagenet_encoder,
        decoder=mini_imagenet_decoder,
        visualization_freq=1,
        learning_rate=0.001,
    )
    # vae.perform_training(epochs=20, checkpoint_freq=100)
    vae.load_latest_checkpoint()
    # vae.visualize_meta_learning_task()

    maml_vae = MAML_VAE(
        vae=vae,
        latent_algorithm='p1',
        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val=1,
        k_val_val=15,
        k_test=1,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='mini_imagenet_crop_random_uniform',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    maml_vae.visualize_meta_learning_task(shape, num_tasks_to_visualize=2)

    maml_vae.train(iterations=8000)
    maml_vae.evaluate(50, seed=42, num_tasks=1000)
