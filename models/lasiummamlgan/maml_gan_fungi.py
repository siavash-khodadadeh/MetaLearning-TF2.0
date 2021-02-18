import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from databases import OmniglotDatabase, MiniImagenetDatabase, CelebADatabase, FungiDatabase
from models.lasiummamlgan.database_parsers import OmniglotParser, MiniImagenetParser, CelebAGANParser
from models.lasiummamlgan.gan import GAN
from models.lasiummamlgan.maml_gan import MAMLGAN
from networks.maml_umtra_networks import MiniImagenetModel
import tensorflow_hub as hub


def get_generator(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64)(latent_inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((7, 7, 64))(x)

    x = layers.Conv2DTranspose(64, 4, activation=None, strides=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(64, 4, activation=None, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(64, 4, activation=None, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(64, 4, activation=None, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(64, 4, activation=None, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # x = layers.Conv2DTranspose(64, 4, activation=None, strides=1, padding="same", use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU(alpha=0.2)(x)

    generator_outputs = layers.Conv2D(3, 4, activation="sigmoid", padding="same")(x)
    generator = keras.Model(latent_inputs, generator_outputs, name="generator")
    generator.summary()

    return generator


def get_discriminator():
    discriminator_inputs = keras.Input(shape=(84, 84, 3))
    x = layers.Conv2D(64, 4, activation=None, strides=2, padding="same", use_bias=False)(discriminator_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(64, 4, activation=None, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(64, 4, activation=None, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(64, 4, activation=None, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(64, 4, activation=None, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.GlobalMaxPooling2D()(x)

    discriminator_outputs = layers.Dense(1)(x)

    discriminator = keras.Model(discriminator_inputs, discriminator_outputs, name="discriminator")

    discriminator.summary()
    return discriminator


class MAMLGANFungi(MAMLGAN):
    def generate_all_vectors(self):
        # vector = tf.random.normal((1, latent_dim))
        # vector2 = -vector
        # class_vectors = tf.concat((vector, vector2), axis=0)

        class_vectors = tf.random.normal((self.n, latent_dim))
        # class_vectors = class_vectors / tf.reshape(tf.norm(class_vectors, axis=1), (class_vectors.shape[0], 1))
        vectors = list()

        vectors.append(class_vectors)
        for i in range(self.k_ml + self.k_val_ml - 1):
            new_vectors = class_vectors
            noise = tf.random.normal(shape=class_vectors.shape, mean=0, stddev=1.5)
            new_vectors += noise
            # new_vectors = new_vectors / tf.reshape(tf.norm(new_vectors, axis=1), (new_vectors.shape[0], 1))
            vectors.append(new_vectors)

        return vectors

    def generate_all_vectors_p2(self):
        class_vectors = tf.random.normal((self.n, latent_dim))
        class_vectors = class_vectors / tf.reshape(tf.norm(class_vectors, axis=1), (class_vectors.shape[0], 1))
        vectors = list()

        vectors.append(class_vectors)
        for i in range(self.k_ml + self.k_val_ml - 1):
            new_vectors = class_vectors
            noise = tf.random.normal(shape=class_vectors.shape, mean=0, stddev=1.0)
            noise = noise / tf.reshape(tf.norm(noise, axis=1), (noise.shape[0], 1))

            new_vectors = new_vectors + (noise - new_vectors) * 0.5

            vectors.append(new_vectors)

        return vectors

    def generate_all_vectors_p3(self):
        z = tf.random.normal((self.n, self.latent_dim))

        vectors = list()
        vectors.append(z)

        for i in range(self.k_ml + self.k_val_ml - 1):
            # if (i + 1) % self.n == 0:
            #     new_z = z + tf.random.normal(shape=z.shape, mean=0, stddev=0)
            #     vectors.append(new_z)
            # else:
            new_z = tf.stack(
                [
                    z[0, ...] + (z[1, ...] - z[0, ...]) * (0.25 + 0.05 * i),
                    z[1, ...] + (z[0, ...] - z[1, ...]) * (0.25 + 0.05 * i),
                    # z[2, ...] + (z[(i + 3) % self.n, ...] - z[2, ...]) * 0.5,
                    # z[3, ...] + (z[(i + 4) % self.n, ...] - z[3, ...]) * 0.5,
                    # z[4, ...] + (z[(i + 5) % self.n, ...] - z[4, ...]) * 0.5
                ],
                axis=0
            )
            vectors.append(new_z)

        return vectors


if __name__ == '__main__':
    fungi_database = FungiDatabase()
    shape = (84, 84, 3)
    latent_dim = 512
    mini_imagenet_generator = get_generator(latent_dim)
    mini_imagenet_discriminator = get_discriminator()
    mini_imagenet_parser = MiniImagenetParser(shape=shape)

    gan = GAN(
        'fungi',
        image_shape=shape,
        latent_dim=latent_dim,
        database=fungi_database,
        parser=mini_imagenet_parser,
        generator=mini_imagenet_generator,
        discriminator=mini_imagenet_discriminator,
        visualization_freq=1,
        d_learning_rate=0.0003,
        g_learning_rate=0.0003,
    )
    # gan.perform_training(epochs=1000, checkpoint_freq=5)
    gan.load_latest_checkpoint()

    maml_gan = MAMLGANFungi(
        gan=gan,
        latent_dim=latent_dim,
        generated_image_shape=shape,
        database=fungi_database,
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
        clip_gradients=False,
        experiment_name='fungi_p1_1.5_v2',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    maml_gan.visualize_meta_learning_task(shape, num_tasks_to_visualize=2)

    maml_gan.train(iterations=60000)
    maml_gan.evaluate(50, num_tasks=1000, seed=42)

# 54 -> 1.5
# 53 -> 1.0
# 52 -> 0.25