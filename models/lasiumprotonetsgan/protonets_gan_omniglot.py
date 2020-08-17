from tensorflow import keras
from tensorflow.keras import layers

from databases import OmniglotDatabase
from models.lasiumprotonetsgan.database_parsers import OmniglotParser
from models.lasiumprotonetsgan.gan import GAN
from models.lasiumprotonetsgan.protonets_gan import ProtoNetsGAN
from networks.proto_networks import SimpleModelProto

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 4.5)])
    except RuntimeError as e:
        print(e)

def get_generator(latent_dim):
    generator = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            # We want to generate 128 coefficients to reshape into a 7x7x128 map
            layers.Dense(7 * 7 * 128),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((7, 7, 128)),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
        ],
        name="generator",
    )

    generator.summary()
    return generator


def get_discriminator():
    discriminator = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),

            layers.GlobalMaxPooling2D(),
            layers.Dense(1),
        ],
        name="discriminator",
    )

    discriminator.summary()
    return discriminator


if __name__ == '__main__':
    omniglot_database = OmniglotDatabase(random_seed=47, num_train_classes=1200, num_val_classes=100)
    shape = (28, 28, 1)
    latent_dim = 128
    omniglot_generator = get_generator(latent_dim)
    omniglot_discriminator = get_discriminator()
    omniglot_parser = OmniglotParser(shape=shape)

    gan = GAN(
        'omniglot',
        image_shape=shape,
        latent_dim=latent_dim,
        database=omniglot_database,
        parser=omniglot_parser,
        generator=omniglot_generator,
        discriminator=omniglot_discriminator,
        visualization_freq=50,
        d_learning_rate=0.0003,
        g_learning_rate=0.0003,
    )
    gan.perform_training(epochs=500, checkpoint_freq=50)
    gan.load_latest_checkpoint()

    proto_gan = ProtoNetsGAN(
        gan=gan,
        latent_dim=latent_dim,
        generated_image_shape=shape,
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
        experiment_name='omniglot_p3_shift_0.2',
        val_seed=42,
    )

    proto_gan.visualize_meta_learning_task(shape, num_tasks_to_visualize=2)

    proto_gan.train(iterations=8000)
    proto_gan.evaluate(-1, seed=42)
