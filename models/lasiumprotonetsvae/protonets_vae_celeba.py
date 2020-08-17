import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from databases import CelebADatabase

from models.lasiumprotonetsvae.database_parsers import CelebAParser
from models.lasiumprotonetsvae.protonets_vae import ProtoNetsVAE
from models.lasiumprotonetsvae.vae import VAE, AudioCallback

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
#     import tensorflow as tf
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(gpus[0], True)
#     tf.config.experimental_run_functions_eagerly(True)

    celebalot_database = CelebADatabase()
    shape = (84, 84, 3)
    latent_dim = 500
    celebalot_encoder = get_encoder(latent_dim)
    celebalot_decoder = get_decoder(latent_dim)
    celebalot_parser = CelebAParser(shape=shape)

    vae = VAE(
        'celeba2',
        image_shape=shape,
        latent_dim=latent_dim,
        database=celebalot_database,
        parser=celebalot_parser,
        encoder=celebalot_encoder,
        decoder=celebalot_decoder,
        visualization_freq=1,
        learning_rate=0.001,
    )
    vae.perform_training(epochs=100, checkpoint_freq=5)
    vae.load_latest_checkpoint()
    vae.visualize_meta_learning_task()

    proto_vae = ProtoNetsVAE(
        vae=vae,
        latent_algorithm='p2',
        database=celebalot_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_train=None,
        k_val_val=15,
        k_val_test=15,
        k_test=5,
        meta_batch_size=4,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        experiment_name='proto_vae_celeba',
        val_seed=42
    )

    proto_vae.visualize_meta_learning_task(shape, num_tasks_to_visualize=2)

    proto_vae.train(iterations=20000)
    proto_vae.evaluate(-1, seed=42)
