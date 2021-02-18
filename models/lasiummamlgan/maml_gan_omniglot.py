from tensorflow import keras
from tensorflow.keras import layers

from databases import OmniglotDatabase
from models.lasiummamlgan.database_parsers import OmniglotParser
from models.lasiummamlgan.gan import GAN
from models.lasiummamlgan.maml_gan import MAMLGAN
from networks.maml_umtra_networks import SimpleModel


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
    # gan.perform_training(epochs=49, checkpoint_freq=1)
    gan.load_latest_checkpoint(epoch_to_load_from='500')

    maml_gan = MAMLGAN(
        gan=gan,
        latent_dim=latent_dim,
        generated_image_shape=shape,
        database=omniglot_database,
        network_cls=SimpleModel,
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val=1,
        k_val_val=15,
        k_val_test=15,
        k_test=1,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.4,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
        num_tasks_val=100,
        clip_gradients=False,
        epsilon=246.09375,
        experiment_name='omniglot_p1_0.5_epsilon_246.09375',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    # for checkpoint in ('00', '10', '30', '50', '100', '200', '300', '400', '500'):
    #     gan.load_latest_checkpoint(epoch_to_load_from=checkpoint)
    #     import tensorflow as tf
    #     tf.random.set_seed(None)
    #      maml_gan.visualize_meta_learning_task(shape, num_tasks_to_visualize=1, checkpoint=checkpoint)
    #  exit()

    print(maml_gan.epsilon)
    maml_gan.visualize_meta_learning_task(shape, num_tasks_to_visualize=1)

    maml_gan.train(iterations=1000)
    maml_gan.evaluate(50, num_tasks=1000, seed=42)
    print(maml_gan.epsilon)
    print(maml_gan.num_epsilon_ignore)
