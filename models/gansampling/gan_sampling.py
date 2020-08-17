import os

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt

import settings
from databases import OmniglotDatabase
from models.maml.maml import ModelAgnosticMetaLearningModel
from utils import combine_first_two_axes


class GANSampling(ModelAgnosticMetaLearningModel):

    def __init__(self, *args, **kwargs):
        super(GANSampling, self).__init__(*args, **kwargs)
        self.gan_vectors_max_stddev = 1.2
        self.gan_batch_size = 256
        self.gan_epochs = 1001
        self.gan_noise_dim = 100
        self.gan_num_examples_to_generate = 16
        self.gan_seed = tf.random.normal([self.gan_num_examples_to_generate, self.gan_noise_dim])
        self.gan_generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.gan_discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.gan_generator = self.make_generator_model()
        self.gan_discriminator = self.make_discriminator_model()
        self.gan_checkpoint_dir = os.path.join(
            settings.PROJECT_ROOT_ADDRESS,
            'models',
            'gansampling',
            'training_checkpoints'
        )
        self.gan_checkpoint_prefix = os.path.join(self.gan_checkpoint_dir, "ckpt")

    def get_network_name(self):
        return self.model.name

    # def get_parse_function(self):
    #     return self.get_gan_parse_function()

    def generate_by_gan(self, class_vectors, method='noise'):
        if method == 'noise':
            return self.generate_by_gan_noise(class_vectors)
        elif method == 'noise_random_stddev':
            return self.generate_by_gan_noise_random_stddev(class_vectors)

    def generate_by_gan_noise(self, class_vectors):
        return class_vectors + tf.random.normal(
            shape=class_vectors.shape, mean=0, stddev=self.gan_vectors_max_stddev
        )

    def generate_by_gan_noise_random_stddev(self, class_vectors):
        return class_vectors + tf.random.normal(
            shape=class_vectors.shape, mean=0, stddev=tf.random.uniform(0, self.gan_vectors_max_stddev)
        )

    def generate_all_vectors_by_class_vectors(self, latent_dim):
        class_vectors = tf.random.normal((self.n, latent_dim))
        vectors = list()

        vectors.append(class_vectors)
        for i in range(self.k + self.k_val_ml - 1):
            vectors.append(
                self.generate_by_gan(class_vectors=class_vectors)
            )

        return vectors

    def generate_all_vectors(self, latent_dim, method='interpolation'):
        if method == 'by_class':
            return self.generate_all_vectors_by_class_vectors(latent_dim)
        elif method == 'interpolation':
            return self.generate_all_vectors_by_interpolation(latent_dim)

    def generate_all_vectors_by_interpolation(self, latent_dim):
        class_vectors = tf.random.normal((self.n, latent_dim))
        class_vectors = class_vectors / tf.reshape(tf.norm(class_vectors, axis=1), (class_vectors.shape[0], 1))
        vectors = list()

        vectors.append(class_vectors)
        for i in range(self.k + self.k_val_ml - 1):
            # if (i + 1) % self.n == 0:
            #     new_vectors = class_vectors
            # else:
            #     shifted_vectors = tf.roll(class_vectors, shift=i + 1, axis=0)
            #     new_vectors = (
            #         class_vectors +
            #         (shifted_vectors - class_vectors) * tf.random.uniform(
            #             shape=(class_vectors.shape[0], 1),
            #             minval=0.6,
            #             maxval=0.6
            #         )
            #     )

            new_vectors = class_vectors
            noise = tf.random.normal(shape=class_vectors.shape, mean=0, stddev=1)
            noise = noise / tf.reshape(tf.norm(noise, axis=1), (noise.shape[0], 1))

            new_vectors = new_vectors + (noise - new_vectors) * 0.6

            vectors.append(new_vectors)

        return vectors

    def get_train_dataset(self):
        instances = list()
        for folder_full_path in self.database.train_folders:
            instances.extend(
                [os.path.join(folder_full_path, instance_file) for instance_file in os.listdir(folder_full_path)]
            )

        self.gan_generator = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']

        # self.train_gan(instances)
        # self.load_gan()

        def tf_image_translate(images, tx, ty, interpolation='NEAREST'):
            transforms = [1, 0, -tx, 0, 1, -ty, 0, 0]
            return tfa.image.transform(images, transforms, interpolation)

        @tf.function
        def get_images_from_vectors(vectors):
            return self.gan_generator(vectors)['default']

        train_labels = np.repeat(np.arange(self.n), self.k)
        train_labels = tf.one_hot(train_labels, depth=self.n)
        train_labels = np.stack([train_labels] * self.meta_batch_size)
        val_labels = np.repeat(np.arange(self.n), self.k_val_ml)
        val_labels = tf.one_hot(val_labels, depth=self.n)
        val_labels = np.stack([val_labels] * self.meta_batch_size)

        latent_dim = 512  # 100 for omniglot
        generated_image_shape = (84, 84, 3)  # (28, 28, 1) for omniglot

        def get_task():
            meta_batch_vectors = list()

            for meta_batch in range(self.meta_batch_size):
                vectors = self.generate_all_vectors(latent_dim)
                vectors = tf.reshape(tf.stack(vectors, axis=0), (-1, latent_dim))
                meta_batch_vectors.append(vectors)

            meta_batch_vectors = tf.stack(meta_batch_vectors)
            meta_batch_vectors = combine_first_two_axes(meta_batch_vectors)
            images = get_images_from_vectors(meta_batch_vectors)
            images = tf.image.resize(images, (84, 84))
            images = tf.reshape(images, (self.meta_batch_size, self.n * (self.k + self.k_val_ml), *generated_image_shape))

            train_ds = images[:, :self.n * self.k, ...]
            train_indices = [i // self.k + i % self.k * self.n for i in range(self.n * self.k)]
            train_ds = tf.gather(train_ds, train_indices, axis=1)
            train_ds = tf.reshape(train_ds, (self.meta_batch_size, self.n, self.k, *generated_image_shape))

            val_ds = images[:, self.n * self.k:, ...]
            val_ds = combine_first_two_axes(val_ds)

            # random_num = tf.random.uniform(shape=(), minval=0, maxval=1)
            # if random_num < 4:
            #     angles = tf.random.uniform(
            #         shape=(self.meta_batch_size * self.n * self.k_val_ml, ),
            #         minval=tf.constant(-np.pi),
            #         maxval=tf.constant(np.pi)
            #     )
            #     val_ds = tfa.image.rotate(val_ds, angles)
            # else:
            #     val_ds = tf_image_translate(
            #         val_ds,
            #         tf.random.uniform((), -5, 5, dtype=tf.int32),
            #         tf.random.uniform((), -5, 5, dtype=tf.int32)
            #     )

            val_ds = tf.reshape(val_ds, (self.meta_batch_size, self.n * self.k_val_ml, *generated_image_shape))

            val_indices = [i // self.k_val_ml + i % self.k_val_ml * self.n for i in range(self.n * self.k_val_ml)]
            val_ds = tf.gather(val_ds, val_indices, axis=1)
            val_ds = tf.reshape(val_ds, (self.meta_batch_size, self.n, self.k_val_ml, *generated_image_shape))

            yield (train_ds, val_ds), (train_labels, val_labels)

        rs = 2135
        tf.random.set_seed(rs)
        for item in get_task():
            (generated_image, val_generated_image), (_, _) = item
            generated_image = generated_image[0, ...]
            val_generated_image = val_generated_image[0, ...]
            # input_vector = tf.random.normal([self.n, 100], mean=0, stddev=1)
            # generated_image = self.gan_generator.predict(input_vector)
            generated_image = generated_image[:, 0, :, :, :]
            generated_image = generated_image
            fig, axes = plt.subplots(6, self.n)
            fig.set_figwidth(self.n)
            fig.set_figheight(6)
            for i in range(self.n):
                image = generated_image[i, :, :, :]
                axes[0, i].imshow(image)

            # val_input_vector = self.generate_by_gan(class_vectors=input_vector)
            # val_generated_image = self.gan_generator.predict(val_input_vector)
            for j in range(5):
                cur_image = val_generated_image[:, j, :, :, :]

                for i in range(self.n):
                    val_image = cur_image[i, :, :, :]
                    axes[j + 1, i].imshow(val_image)  # cmap='gray' for omniglot
            plt.show()
            fig.savefig(f'/home/siavash/Desktop/{rs}.jpg')
        exit()

        dataset = tf.data.Dataset.from_generator(
            get_task,
            output_types=((tf.float32, tf.float32), (tf.float32, tf.float32))
        )
        # dataset = dataset.repeat(self.meta_batch_size)
        # dataset = dataset.repeat(-1)
        # dataset = dataset.batch(batch_size=self.meta_batch_size)
        # steps_per_epoch = 50
        # dataset = dataset.take(50)
        # dataset = dataset.prefetch(5)
        steps_per_epoch = 1

        # for i in range(1):
        #     for item in dataset:
        #         (train_ds, val_ds), (train_labels, val_labels) = item
        #         print(train_ds.shape)
        #         print(val_ds.shape)
        #         print(train_labels.shape)
        #         print(val_labels.shape)
        #         print(tf.unique(tf.cast(tf.reshape(train_ds, (-1, )) * 127.5, tf.int32)))
        #         print(tf.reduce_min(train_ds))
        #         print(tf.reduce_max(train_ds))
        #         # exit()
        #         plt.imshow(train_ds[0, 0, 0, :, :, 0], cmap='gray')
        #         plt.show()
        #         plt.imshow(val_ds[0, 0, 0, :, :, 0], cmap='gray')
        #         plt.show()
        #
        # exit()

        setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset

    def get_gan_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_png(tf.io.read_file(example_address))
            image = tf.image.resize(image, (28, 28))
            image = tf.cast(image, tf.float32)

            return (image - 127.5) / 127.5
        return parse_function

    def load_gan(self):
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.gan_generator_optimizer,
                                         discriminator_optimizer=self.gan_discriminator_optimizer,
                                         generator=self.gan_generator,
                                         discriminator=self.gan_discriminator)

        latest_checkpoint = tf.train.latest_checkpoint(self.gan_checkpoint_dir)
        if latest_checkpoint is not None:
            checkpoint.restore(latest_checkpoint)
            current_epoch = int(latest_checkpoint[latest_checkpoint.find('-') + 1:latest_checkpoint.rfind('-')])
        else:
            current_epoch = 0

        return checkpoint, current_epoch

    def train_gan(self, instances):
        buffer_size = len(instances)
        train_dataset = tf.data.Dataset.from_tensor_slices(instances).shuffle(buffer_size)
        train_dataset = train_dataset.map(self.get_gan_parse_function())
        train_dataset = train_dataset.batch(self.gan_batch_size)

        checkpoint, current_epoch = self.load_gan()

        gen_loss_metric = tf.metrics.Mean()
        disc_loss_metric = tf.metrics.Mean()
        reg_loss_metric = tf.metrics.Mean()

        for epoch in range(current_epoch, self.gan_epochs):
            if epoch % 25 == 0:
                self.generate_and_save_images(self.gan_generator, epoch, self.gan_seed)
                print(f'Epoch: {epoch}')
                print(f'Gen Loss: {gen_loss_metric.result()}')
                print(f'Disc Loss: {disc_loss_metric.result()}')
                print(f'Reg Loss: {reg_loss_metric.result()}')

                # Save the model every 50 epochs
                if epoch != 0 and epoch % 50 == 0:
                    checkpoint.save(file_prefix=self.gan_checkpoint_prefix + f'-{epoch}')

            gen_loss_metric.reset_states()
            disc_loss_metric.reset_states()
            reg_loss_metric.reset_states()

            for image_batch in train_dataset:
                gl, dl, rl = self.gan_train_step(image_batch)
                gen_loss_metric.update_state(gl)
                disc_loss_metric.update_state(dl)
                reg_loss_metric.update_state(rl)

        # Generate after the final epoch
        # self.generate_and_save_images(self.gan_generator, self.gan_epochs, self.gan_seed)

    @tf.function
    def gan_train_step(self, images):
        noise = tf.random.normal([images.shape[0], self.gan_noise_dim])
        noise2 = tf.random.normal([images.shape[0], self.gan_noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.gan_generator(noise, training=True)
            generated_images_2 = self.gan_generator(noise2, training=True)

            real_output = self.gan_discriminator(images, training=True)
            fake_output = self.gan_discriminator(generated_images, training=True)
            fake_output2 = self.gan_discriminator(generated_images_2, training=True)

            gen_loss = self.generator_loss(fake_output) + self.generator_loss(fake_output2)
            disc_loss = self.discriminator_loss(real_output, fake_output) + \
                self.discriminator_loss(real_output, fake_output2)
            regularization_loss = self.gan_regularization_loss(noise, noise2, generated_images, generated_images_2)
            final_gen_loss = gen_loss + regularization_loss

        gradients_of_generator = gen_tape.gradient(
            final_gen_loss, self.gan_generator.trainable_variables + self.gan_discriminator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.gan_discriminator.trainable_variables)

        self.gan_generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.gan_generator.trainable_variables))
        self.gan_discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.gan_discriminator.trainable_variables))

        return gen_loss, disc_loss, regularization_loss

    def gan_regularization_loss(self, z1, z2, i1, i2):
        images_norm = tf.reduce_mean(tf.abs(i1 - i2), axis=[1, 2, 3])
        zs_norm = tf.reduce_mean(tf.abs(z1 - z2), axis=1)

        lz = tf.reduce_mean(images_norm / zs_norm)
        eps = 1e-5
        lz = 1 / (lz + eps)
        return lz

    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # None is the batch size

        model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(
            tf.keras.layers.Conv2DTranspose(
                1,
                (5, 5),
                strides=(2, 2),
                padding='same',
                use_bias=False,
                activation='tanh'
            )
        )
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        # tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
        real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output, from_logits=True)
        fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output, from_logits=True)
        total_loss = real_loss + fake_loss
        return tf.reduce_mean(total_loss)

    def generator_loss(self, fake_output):
        fake_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output, from_logits=True)
        return tf.reduce_mean(fake_loss)

    def generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        if not os.path.exists('gan_training_images'):
            os.mkdir('gan_training_images')
        plt.savefig(f'gan_training_images/image_at_epoch_{epoch:04d}.png')
        plt.show()
