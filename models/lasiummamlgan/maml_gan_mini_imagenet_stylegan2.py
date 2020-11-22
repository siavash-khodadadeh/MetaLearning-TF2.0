from math import floor, log2

import tensorflow as tf
import numpy as np

from databases import MiniImagenetDatabase
from models.lasiummamlgan.database_parsers import MiniImagenetParser
from models.lasiummamlgan.maml_gan import MAMLGAN
from networks.maml_umtra_networks import MiniImagenetModel
from stylegan.stylegan_two import StyleGAN


class MiniImageNetMAMLStyleGan2(MAMLGAN):
    @tf.function
    def get_images_from_vectors(self, vectors):

        n_layers = int(log2(64) - 1)
        n1 = [vectors] * n_layers
        n2 = np.zeros(shape=[vectors.shape[0], 64, 64, 1]).astype('float32')
        # trunc = 1.0
        # trunc = np.ones([64, 1]) * trunc
        im = self.gan.GAN.GM(n1 + [n2], training=False)
        im = tf.clip_by_value(im, 0, 1)

        return im
        # return (self.gan(vectors)['default'] + 1) / 2.

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
            noise = tf.random.normal(shape=class_vectors.shape, mean=0, stddev=1.0)
            new_vectors += noise
            # new_vectors = new_vectors / tf.reshape(tf.norm(new_vectors, axis=1), (new_vectors.shape[0], 1))
            vectors.append(new_vectors)

        return vectors

    def generate_all_vectors_p2(self):
        class_vectors = tf.random.truncated_normal((self.n, latent_dim))
        # class_vectors = tf.random.normal((self.n, latent_dim))
        # class_vectors = class_vectors / tf.reshape(tf.norm(class_vectors, axis=1), (class_vectors.shape[0], 1))
        vectors = list()

        vectors.append(class_vectors)
        for i in range(self.k_ml + self.k_val_ml - 1):
            new_vectors = class_vectors
            noise = tf.random.normal(shape=class_vectors.shape, mean=0, stddev=1)
            # noise = noise / tf.reshape(tf.norm(noise, axis=1), (noise.shape[0], 1))

            new_vectors = new_vectors + (noise - new_vectors) * 0.3

            vectors.append(new_vectors)

        return vectors

    def generate_all_vectors_p3(self):
        # z = tf.random.truncated_normal((self.n, self.latent_dim))
        z = tf.random.normal((self.n, self.latent_dim))

        vectors = list()
        vectors.append(z)

        for i in range(self.k_ml + self.k_val_ml - 1):
            if (i + 1) % self.n == 0:
                new_z = z + tf.random.normal(shape=z.shape, mean=0, stddev=0.5)
                vectors.append(new_z)
            else:
                new_z = tf.stack(
                    [
                        z[0, ...] + (z[(i + 1) % self.n, ...] - z[0, ...]) * 0.3,
                        z[1, ...] + (z[(i + 2) % self.n, ...] - z[1, ...]) * 0.3,
                        z[2, ...] + (z[(i + 3) % self.n, ...] - z[2, ...]) * 0.3,
                        z[3, ...] + (z[(i + 4) % self.n, ...] - z[3, ...]) * 0.3,
                        z[4, ...] + (z[(i + 0) % self.n, ...] - z[4, ...]) * 0.3,
                    ],
                    axis=0
                )
                vectors.append(new_z)

        return vectors


if __name__ == '__main__':
    # tf.config.experimental_run_functions_eagerly(True)

    mini_imagenet_database = MiniImagenetDatabase()
    shape = (84, 84, 3)
    latent_dim = 512
    gan = StyleGAN(lr=0.00001, silent=False)
    gan.load(20)
    gan.trainable = False
    setattr(gan, 'parser', MiniImagenetParser(shape=shape))

    maml_gan = MiniImageNetMAMLStyleGan2(
        gan=gan,
        latent_dim=latent_dim,
        generated_image_shape=shape,
        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val=1,
        k_val_val=15,
        k_test=1,
        k_val_test=15,
        meta_batch_size=1,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='mini_imagenet_stylegan2_1',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    maml_gan.visualize_meta_learning_task(shape, num_tasks_to_visualize=5)

    maml_gan.train(iterations=60000)
    maml_gan.evaluate(50, seed=42, num_tasks=1000)
