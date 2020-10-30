import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers

from databases import OmniglotDatabase, MiniImagenetDatabase
from models.lasiummamlgan.database_parsers import OmniglotParser, MiniImagenetParser
from models.lasiummamlgan.gan import GAN
from models.lasiummamlgan.maml_gan import MAMLGAN
from networks.maml_umtra_networks import SimpleModel, MiniImagenetModel, VGG19Model, FiveLayerResNet


# Hub module info
# Signature: discriminate
# Inputs: {'x': <hub.ParsedTensorInfo shape=(?, 128, 128, 3) dtype=float32 is_sparse=False>,
#  'z': <hub.ParsedTensorInfo shape=(?, 120) dtype=float32 is_sparse=False>}
# Outputs: {'score_x': <hub.ParsedTensorInfo shape=(?,) dtype=float32 is_sparse=False>,
#  'score_xz': <hub.ParsedTensorInfo shape=(?,) dtype=float32 is_sparse=False>,
#  'score_z': <hub.ParsedTensorInfo shape=(?,) dtype=float32 is_sparse=False>}
#
# Signature: default
# Inputs: {'x': <hub.ParsedTensorInfo shape=(?, 256, 256, 3) dtype=float32 is_sparse=False>}
# Outputs: {'default': <hub.ParsedTensorInfo shape=(?, 120) dtype=float32 is_sparse=False>}
#
# Signature: encode
# Inputs: {'x': <hub.ParsedTensorInfo shape=(?, 256, 256, 3) dtype=float32 is_sparse=False>}
# Outputs: {'avepool_feat': <hub.ParsedTensorInfo shape=(?, 2048) dtype=float32 is_sparse=False>,
#  'bn_crelu_feat': <hub.ParsedTensorInfo shape=(?, 4096) dtype=float32 is_sparse=False>,
#  'default': <hub.ParsedTensorInfo shape=(?, 120) dtype=float32 is_sparse=False>,
#  'z_mean': <hub.ParsedTensorInfo shape=(?, 120) dtype=float32 is_sparse=False>,
#  'z_sample': <hub.ParsedTensorInfo shape=(?, 120) dtype=float32 is_sparse=False>,
#  'z_stdev': <hub.ParsedTensorInfo shape=(?, 120) dtype=float32 is_sparse=False>}
#
# Signature: generate
# Inputs: {'z': <hub.ParsedTensorInfo shape=(?, 120) dtype=float32 is_sparse=False>}
# Outputs: {'default': <hub.ParsedTensorInfo shape=(?, 128, 128, 3) dtype=float32 is_sparse=False>,
#  'upsampled': <hub.ParsedTensorInfo shape=(?, 256, 256, 3) dtype=float32 is_sparse=False>}


class MiniImageNetMAMLBigGan(MAMLGAN):
    @tf.function
    def get_images_from_vectors(self, vectors):
        # return self.gan(vectors)['generate']
        return (self.gan(vectors)['default'] + 1) / 2.

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
            noise = tf.random.normal(shape=class_vectors.shape, mean=0, stddev=0.5)
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
    mini_imagenet_database = MiniImagenetDatabase(input_shape=(224, 224, 3))
    shape = (224, 224, 3)
    latent_dim = 120
    import os
    os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser('~/tf_hub')

    gan = hub.load("https://tfhub.dev/deepmind/bigbigan-resnet50/1", tags=[]).signatures['generate']
    setattr(gan, 'parser', MiniImagenetParser(shape=shape))

    maml_gan = MiniImageNetMAMLBigGan(
        gan=gan,
        latent_dim=latent_dim,
        generated_image_shape=shape,
        database=mini_imagenet_database,
        network_cls=FiveLayerResNet,
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val=1,
        k_val_val=15,
        k_test=1,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.001,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.0001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='mini_imagenet_p1_resnet',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    maml_gan.visualize_meta_learning_task(shape, num_tasks_to_visualize=5)

    maml_gan.train(iterations=60000)
    maml_gan.evaluate(50, seed=42, num_tasks=1000)
