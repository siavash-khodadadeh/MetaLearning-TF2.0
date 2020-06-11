import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from databases import CelebADatabase

from models.lasiummamlvae.database_parsers import CelebAParser
from models.lasiummamlvae.maml_vae import MAML_VAE
from models.lasiummamlvae.vae import VAE, AudioCallback
from networks.maml_umtra_networks import SimpleModel, MiniImagenetModel


class MAMLVAECelebA(MAML_VAE):
    def generate_with_p3(self, z, z_mean, z_log_var, rotation_index):
        if (rotation_index + 1) % self.n == 0:
            return z + tf.random.normal(shape=z.shape, mean=0, stddev=0.5)

        z = self.vae.sample(z_mean, z_log_var)
        new_z = tf.stack(
            [
                z[0, ...] + (z[(rotation_index + 1) % self.n, ...] - z[0, ...]) / 2.5,
                z[1, ...] + (z[(rotation_index + 2) % self.n, ...] - z[1, ...]) / 2.5,
                # z[2, ...] + (z[(rotation_index + 3) % 5, ...] - z[2, ...]) / 2.5,
                # z[3, ...] + (z[(rotation_index + 4) % 5, ...] - z[3, ...]) / 2.5,
                # z[4, ...] + (z[(rotation_index + 0) % 5, ...] - z[4, ...]) / 2.5,
            ],
            axis=0
        )

        return new_z

    # def get_val_dataset(self):
    #     val_dataset = self.database.get_attributes_task_dataset(
    #         partition='val',
    #         k=self.k_val_train,
    #         k_val=self.k_val_val,
    #         meta_batch_size=1,
    #         seed=self.val_seed
    #     )
    #     val_dataset = val_dataset.repeat(-1)
    #     val_dataset = val_dataset.take(self.number_of_tasks_val)
    #     setattr(val_dataset, 'steps_per_epoch', self.number_of_tasks_val)
    #     return val_dataset

    # def get_test_dataset(self, seed=-1):
    #     # dataset = super(MAMLGANProGAN, self).get_test_dataset(seed=seed)
    #     test_dataset = self.database.get_attributes_task_dataset(
    #         partition='test',
    #         k=self.k_test,
    #         k_val=self.k_val_test,
    #         meta_batch_size=1,
    #         seed=seed
    #     )
    #     test_dataset = test_dataset.repeat(-1)
    #     test_dataset = test_dataset.take(self.number_of_tasks_test)
    #     setattr(test_dataset, 'steps_per_epoch', self.number_of_tasks_test)
    #     return test_dataset


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

    celebalot_database = CelebADatabase()
    shape = (84, 84, 3)
    latent_dim = 500
    celebalot_encoder = get_encoder(latent_dim)
    celebalot_decoder = get_decoder(latent_dim)
    celebalot_parser = CelebAParser(shape=shape)

    vae = VAE(
        'celeba',
        image_shape=shape,
        latent_dim=latent_dim,
        database=celebalot_database,
        parser=celebalot_parser,
        encoder=celebalot_encoder,
        decoder=celebalot_decoder,
        visualization_freq=1,
        learning_rate=0.001,
    )
    # vae.perform_training(epochs=20, checkpoint_freq=100)
    vae.load_latest_checkpoint()
    # vae.visualize_meta_learning_task()

    maml_vae = MAMLVAECelebA(
        vae=vae,
        latent_algorithm='p3',
        database=celebalot_database,
        network_cls=MiniImagenetModel,
        n=2,
        k=1,
        k_val_ml=5,
        k_val_train=5,
        k_val_val=5,
        k_test=5,
        k_val_test=5,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=True,
        experiment_name='celeba_better_vae2',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    # final
    # acc: 78.78 + - 0.96

    maml_vae.visualize_meta_learning_task(shape, num_tasks_to_visualize=2)

    maml_vae.train(iterations=8000)
    maml_vae.evaluate(50, seed=42)
