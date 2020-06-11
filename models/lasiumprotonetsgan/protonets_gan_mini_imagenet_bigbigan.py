import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub

from databases import MiniImagenetDatabase
from models.lasiumprotonetsgan.database_parsers import MiniImagenetParser
from models.lasiumprotonetsgan.gan import GAN
from models.lasiumprotonetsgan.protonets_gan import ProtoNetsGAN

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

class ProtoNetsProGAN(ProtoNetsGAN):
    @tf.function
    def get_images_from_vectors(self, vectors):
        return self.gan(vectors)['default']

if __name__ == '__main__':
    mini_imagenet_database = MiniImagenetDatabase()
    shape = (84, 84, 3)
    latent_dim = 120

    gan = hub.load("https://tfhub.dev/deepmind/bigbigan-resnet50/1", tags=[]).signatures['generate']
    setattr(gan, 'parser', MiniImagenetParser(shape=shape))
    # gan = tf.saved_model.load('../mamlgan/gan/mini_imagenet/bigbigan', tags=[]).signatures['generate']
    # setattr(gan, 'parser', MiniImagenetParser(shape=(84, 84, 3)))

    proto_gan = ProtoNetsProGAN(
        gan=gan,
        latent_dim=latent_dim,
        generated_image_shape=shape,
        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_train=1,
        k_val_val=15,
        k_val_test=15,
        k_test=1,
        meta_batch_size=2,
        save_after_iterations=1000,
        meta_learning_rate=0.0001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        experiment_name='mini_imagenet_p1',
        val_seed=42,
    )

    proto_gan.visualize_meta_learning_task(shape, num_tasks_to_visualize=2)

    proto_gan.train(iterations=40000)
    proto_gan.evaluate(-1, seed=42, use_val_batch_statistics=False, iterations_to_load_from=30000)

    proto_gan.k_test = 5
    proto_gan.evaluate(-1, seed=42, use_val_batch_statistics=False, iterations_to_load_from=30000)

    proto_gan.k_test = 20
    proto_gan.evaluate(-1, seed=42, use_val_batch_statistics=False, iterations_to_load_from=30000)

    proto_gan.k_test = 50
    proto_gan.evaluate(-1, seed=42, use_val_batch_statistics=False, iterations_to_load_from=30000)