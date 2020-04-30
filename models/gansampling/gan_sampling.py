import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models.maml.maml import ModelAgnosticMetaLearningModel


class GANSampling(ModelAgnosticMetaLearningModel):

    def __init__(self, *args, **kwargs):
        super(GANSampling, self).__init__(*args, **kwargs)
        self.gan_vectors_max_stddev = 0.5
        self.gan_batch_size = 256
        self.gan_epochs = 300
        self.gan_noise_dim = 100
        self.gan_num_examples_to_generate = 16
        self.gan_seed = tf.random.normal([self.gan_num_examples_to_generate, self.gan_noise_dim])
        self.gan_generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.gan_discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.gan_generator = self.make_generator_model()
        self.gan_discriminator = self.make_discriminator_model()

    def get_parse_function(self):
        return self.get_gan_parse_function()

    def get_train_dataset(self):
        instances = list()
        for folder_full_path in self.database.train_folders:
            instances.extend(
                [os.path.join(folder_full_path, instance_file) for instance_file in os.listdir(folder_full_path)]
            )

        self.train_gan(instances)

        # input_vector = tf.random.normal([self.n, 100])
        # generated_image = self.gan_generator.predict(input_vector)
        # generated_image = generated_image * 127.5 + 127.5
        # fig, axes = plt.subplots(2, self.n)
        # for i in range(self.n):
        #     image = generated_image[i, :, :, 0]
        #     axes[0, i].imshow(image, cmap='gray')
        #
        # val_input_vector = input_vector + tf.random.normal(shape=(self.n, 100), mean=0, stddev=0.5)
        # val_generated_image = self.gan_generator.predict(val_input_vector)
        # val_generated_image = val_generated_image * 127.5 + 127.5
        # for i in range(self.n):
        #     val_image = val_generated_image[i, :, :, 0]
        #     axes[1, i].imshow(val_image, cmap='gray')
        # plt.show()

        def get_task():
            # return (train_ds, val_ds), (train_labels, val_labels)
            class_vectors = tf.random.normal((self.n, 100))
            vectors = list()

            vectors.append(class_vectors)
            for i in range(self.k + self.k_val_ml - 1):
                vectors.append(
                    class_vectors + tf.random.normal(
                        shape=(self.n, 100), mean=0, stddev=np.random.uniform(0, self.gan_vectors_max_stddev)
                    )
                )

            vectors = tf.reshape(tf.stack(vectors, axis=0), (-1, 100))
            images = self.gan_generator.predict(vectors)
            train_ds = images[:self.n * self.k]
            train_indices = [i // self.k + i % self.k * self.n for i in range(self.n * self.k)]
            train_ds = tf.gather(train_ds, train_indices, axis=0)
            train_ds = tf.reshape(train_ds, (self.n, self.k, 28, 28, 1))

            val_ds = images[self.n * self.k:]
            val_indices = [i // self.k_val_ml + i % self.k_val_ml * self.n for i in range(self.n * self.k_val_ml)]
            val_ds = tf.gather(val_ds, val_indices, axis=0)
            val_ds = tf.reshape(val_ds, (self.n, self.k_val_ml, 28, 28, 1))

            train_labels = np.repeat(np.arange(self.n), self.k)
            val_labels = np.repeat(np.arange(self.n), self.k_val_ml)
            train_labels = tf.one_hot(train_labels, depth=self.n)
            val_labels = tf.one_hot(val_labels, depth=self.n)

            yield (train_ds, val_ds), (train_labels, val_labels)

        dataset = tf.data.Dataset.from_generator(
            get_task,
            output_types=((tf.float32, tf.float32), (tf.float32, tf.float32))
        )
        dataset = dataset.repeat(self.meta_batch_size)
        dataset = dataset.batch(batch_size=self.meta_batch_size)
        steps_per_epoch = 1
        dataset = dataset.take(1)

        # for i in range(10):
        #     for item in dataset:
        #         (train_ds, val_ds), (train_labels, val_labels) = item
        #         print(train_ds.shape)
        #         print(val_ds.shape)
        #         print(train_labels.shape)
        #         print(val_labels.shape)
        #         exit()
        #         plt.imshow(train_ds[0, 0, 0, :, :, 0])
        #         plt.show()
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

    def train_gan(self, instances):
        buffer_size = len(instances)
        train_dataset = tf.data.Dataset.from_tensor_slices(instances).shuffle(buffer_size)
        train_dataset = train_dataset.map(self.get_gan_parse_function())
        train_dataset = train_dataset.batch(self.gan_batch_size)

        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.gan_generator_optimizer,
                                         discriminator_optimizer=self.gan_discriminator_optimizer,
                                         generator=self.gan_generator,
                                         discriminator=self.gan_discriminator)

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is not None:
            checkpoint.restore(latest_checkpoint)
            current_epoch = int(latest_checkpoint[latest_checkpoint.find('-') + 1:latest_checkpoint.rfind('-')])
        else:
            current_epoch = 0

        gen_loss_metric = tf.metrics.Mean()
        disc_loss_metric = tf.metrics.Mean()
        gen_loss_metric.reset_states()
        disc_loss_metric.reset_states()

        for epoch in range(current_epoch, self.gan_epochs):
            if epoch % 25 == 0:
                self.generate_and_save_images(self.gan_generator, epoch, self.gan_seed)
                print(f'Epoch: {epoch}')
                print(f'Gen Loss: {gen_loss_metric.result()}')
                print(f'Disc Loss: {disc_loss_metric.result()}')

                # Save the model every 50 epochs
                if epoch != 0 and epoch % 50 == 0:
                    checkpoint.save(file_prefix=checkpoint_prefix + f'-{epoch}')

            for image_batch in train_dataset:
                gl, dl = self.gan_train_step(image_batch)
                gen_loss_metric.update_state(gl)
                disc_loss_metric.update_state(dl)

        # Generate after the final epoch
        # self.generate_and_save_images(self.gan_generator, self.gan_epochs, self.gan_seed)

    @tf.function
    def gan_train_step(self, images):
        noise = tf.random.normal([images.shape[0], self.gan_noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.gan_generator(noise, training=True)

            real_output = self.gan_discriminator(images, training=True)
            fake_output = self.gan_discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.gan_generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.gan_discriminator.trainable_variables)

        self.gan_generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.gan_generator.trainable_variables))
        self.gan_discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.gan_discriminator.trainable_variables))

        return gen_loss, disc_loss

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
        return total_loss

    def generator_loss(self, fake_output):
        fake_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output, from_logits=True)
        return fake_loss

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
