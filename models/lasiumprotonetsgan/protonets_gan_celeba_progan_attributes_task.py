import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub

from databases import OmniglotDatabase, MiniImagenetDatabase, CelebADatabase
from models.lasiumprotonetsgan.database_parsers import OmniglotParser, MiniImagenetParser, CelebAAttributeParser
from models.lasiumprotonetsgan.gan import GAN
from models.lasiumprotonetsgan.protonets_gan import ProtoNetsGAN
from utils import combine_first_two_axes
from tqdm import tqdm


class MiniImagenetModel(tf.keras.Model):
    name = 'MiniImagenetModel'
    def __init__(self, *args, **kwargs):

        super(MiniImagenetModel, self).__init__(*args, **kwargs)
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.conv1 = tf.keras.layers.Conv2D(32, 3, name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn1')
        # self.bn1 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, name='conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn2')
        # self.bn2 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(32, 3, name='conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn3')
        # self.bn3 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(32, 3, name='conv4')
        self.bn4 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn4')
        # self.bn4 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn4')
        self.flatten = tf.keras.layers.Flatten(name='flatten')

    def conv_block(self, features, conv, bn=None, training=False):
        conv_out = conv(features)
        batch_normalized_out = bn(conv_out, training=training)
        batch_normalized_out = self.max_pool(batch_normalized_out)
        return tf.keras.activations.relu(batch_normalized_out)

    def get_features(self, inputs, training=False):
        import numpy as np
        image = inputs
        c1 = self.conv_block(image, self.conv1, self.bn1, training=training)
        c2 = self.conv_block(c1, self.conv2, self.bn2, training=training)
        c3 = self.conv_block(c2, self.conv3, self.bn3, training=training)
        c4 = self.conv_block(c3, self.conv4, self.bn4, training=training)
        c4 = tf.reshape(c4, [-1, np.prod([int(dim) for dim in c4.get_shape()[1:]])])
        f = self.flatten(c4)
        return f

    def call(self, inputs, training=False):
        out = self.get_features(inputs, training=training)

        return out


class ProtoGANProGAN(ProtoNetsGAN):
    @tf.function
    def get_images_from_vectors(self, vectors):
        return self.gan(vectors)['default']

    def generate_all_vectors_p1(self):
        # vector = tf.random.normal((1, latent_dim))
        # vector2 = -vector
        # class_vectors = tf.concat((vector, vector2), axis=0)

        class_vectors = tf.random.normal((self.n, latent_dim))
        class_vectors = class_vectors / tf.reshape(tf.norm(class_vectors, axis=1), (class_vectors.shape[0], 1))
        vectors = list()

        vectors.append(class_vectors)
        for i in range(self.k_ml + self.k_val_ml - 1):
            new_vectors = class_vectors
            noise = tf.random.normal(shape=class_vectors.shape, mean=0, stddev=0.08)
            new_vectors += noise
            new_vectors = new_vectors / tf.reshape(tf.norm(new_vectors, axis=1), (new_vectors.shape[0], 1))
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

    def generate_all_vectors(self):
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
                    z[0, ...] + (z[1, ...] - z[0, ...]) * (0.35 + 0.05 * i),
                    z[1, ...] + (z[0, ...] - z[1, ...]) * (0.35 + 0.05 * i),
                    # z[2, ...] + (z[(i + 3) % self.n, ...] - z[2, ...]) * 0.5,
                    # z[3, ...] + (z[(i + 4) % self.n, ...] - z[3, ...]) * 0.5,
                    # z[4, ...] + (z[(i + 5) % self.n, ...] - z[4, ...]) * 0.5
                ],
                axis=0
            )
            vectors.append(new_z)

        return vectors

    def get_val_dataset(self):
        val_dataset = self.database.get_attributes_task_dataset(
            partition='val',
            k=self.k_val,
            k_val=self.k_val_val,
            meta_batch_size=1,
            parse_fn=self.gan.parser.get_parse_fn(),
            seed=self.val_seed
        )
        val_dataset = val_dataset.repeat(-1)
        val_dataset = val_dataset.take(self.num_tasks_val)
        setattr(val_dataset, 'steps_per_epoch', self.num_tasks_val)
        return val_dataset

    def get_test_dataset(self, num_tasks=1000, seed=-1):
        test_dataset = self.database.get_attributes_task_dataset(
            partition='test',
            k=self.k_test,
            k_val=self.k_val_test,
            meta_batch_size=1,
            parse_fn=self.gan.parser.get_parse_fn(),
            seed=seed
        )
        test_dataset = test_dataset.repeat(-1)
        test_dataset = test_dataset.take(num_tasks)

        setattr(test_dataset, 'steps_per_epoch', num_tasks)
        return test_dataset


if __name__ == '__main__':
    celeba_database = CelebADatabase()
    shape = (84, 84, 3)
    latent_dim = 512
    gan = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']
    # you can download the module manually and load it with code below:
    # gan = tf.saved_model.load('gan/celeba').signatures['default']
    setattr(gan, 'parser', CelebAAttributeParser(shape=(84, 84, 3)))

    proto_gan = ProtoGANProGAN(
        gan=gan,
        latent_dim=latent_dim,
        generated_image_shape=shape,
        database=celeba_database,
        network_cls=MiniImagenetModel,
        n=2,  # n=2
        k_ml=1,
        k_val_ml=5,
        k_val=5,
        k_val_val=5,
        k_val_test=5,  # k_val_test=5
        k_test=5,  # k_test=5
        meta_batch_size=4,
        save_after_iterations=1000,
        meta_learning_rate=1e-4,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
        num_tasks_val=100,
        experiment_name='celeba_attributes_p3_0.35',
        val_seed=42,
    )

    proto_gan.visualize_meta_learning_task(shape, num_tasks_to_visualize=2)

    proto_gan.train(iterations=120000)
    proto_gan.evaluate(-1, num_tasks=1000, seed=42, iterations_to_load_from=1000)
