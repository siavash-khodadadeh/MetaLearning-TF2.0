import tensorflow as tf
import tensorflow_addons as tfa
from models.protonets.proto_nets import PrototypicalNetworks
from utils import combine_first_two_axes


class ProtoNetsGAN(PrototypicalNetworks):
    def __init__(self, gan, latent_dim, generated_image_shape, *args, **kwargs):
        super(ProtoNetsGAN, self).__init__(*args, **kwargs)
        self.gan = gan
        self.latent_dim = latent_dim
        self.generated_image_shape = generated_image_shape

    def get_network_name(self):
        return self.model.name

    def get_parse_function(self):
        return self.gan.parser.get_parse_fn()

    def visualize_meta_learning_task(self, shape, num_tasks_to_visualize=1):
        import matplotlib.pyplot as plt

        dataset = self.get_train_dataset()
        for item in dataset.repeat(-1).take(num_tasks_to_visualize):
            fig, axes = plt.subplots(self.k + self.k_val_ml, self.n)
            fig.set_figwidth(self.n)
            fig.set_figheight(self.k + self.k_val_ml)

            (train_ds, val_ds), (_, _) = item
            # Get the first meta batch
            train_ds = train_ds[0, ...]
            val_ds = val_ds[0, ...]
            if shape[2] == 1:
                train_ds = train_ds[..., 0]
                val_ds = val_ds[..., 0]

            for n in range(self.n):
                for k in range(self.k):
                    axes[k, n].imshow(train_ds[n, k, ...])

                for k in range(self.k_val_ml):
                    axes[k + self.k, n].imshow(val_ds[n, k, ...])

            plt.show()

    def generate_all_vectors_p1(self):
        class_vectors = tf.random.normal((self.n, self.latent_dim))
#         class_vectors = class_vectors / tf.reshape(tf.norm(class_vectors, axis=1), (class_vectors.shape[0], 1))
        vectors = list()

        vectors.append(class_vectors)
        for i in range(self.k + self.k_val_ml - 1):
            new_vectors = class_vectors
            noise = tf.random.normal(shape=class_vectors.shape, mean=0, stddev=1)
            new_vectors += noise
#             new_vectors = new_vectors / tf.reshape(tf.norm(new_vectors, axis=1), (new_vectors.shape[0], 1))
            vectors.append(new_vectors)

        return vectors

    def generate_all_vectors_p2(self):
        class_vectors = tf.random.normal((self.n, self.latent_dim))
#         class_vectors = class_vectors / tf.reshape(tf.norm(class_vectors, axis=1), (class_vectors.shape[0], 1))
        vectors = list()
        vectors.append(class_vectors)
        for i in range(self.k + self.k_val_ml - 1):
            new_vectors = class_vectors
            noise = tf.random.normal(shape=class_vectors.shape, mean=0, stddev=1)
#             noise = noise / tf.reshape(tf.norm(noise, axis=1), (noise.shape[0], 1))
            new_vectors = new_vectors + (noise - new_vectors) * 0.8

            vectors.append(new_vectors)
        return vectors

    def generate_all_vectors_p3(self):
        coef = 0.2
        z = tf.random.normal((self.n, self.latent_dim))

        vectors = list()
        vectors.append(z)

        for i in range(self.k + self.k_val_ml - 1):
            if (i + 1) % 5 == 0:
                new_z = z + tf.random.normal(shape=z.shape, mean=0, stddev=1.0)
                vectors.append(new_z)
            else:
                new_z = tf.stack(
                    [
                        z[0, ...] + (z[(i + 1) % self.n, ...] - z[0, ...]) * coef,
                        z[1, ...] + (z[(i + 2) % self.n, ...] - z[1, ...]) * coef,
                        z[2, ...] + (z[(i + 3) % self.n, ...] - z[2, ...]) * coef,
                        z[3, ...] + (z[(i + 4) % self.n, ...] - z[3, ...]) * coef,
                        z[4, ...] + (z[(i + 0) % self.n, ...] - z[4, ...]) * coef,
                    ],
                    axis=0
                )
                vectors.append(new_z)

        return vectors

    def generate_all_vectors(self):
        return self.generate_all_vectors_p3()

    @tf.function
    def get_images_from_vectors(self, vectors):
        return self.gan.generator(vectors)

    def get_train_dataset(self):
        train_labels = tf.repeat(tf.range(self.n), self.k)
        train_labels = tf.one_hot(train_labels, depth=self.n)
        train_labels = tf.stack([train_labels] * self.meta_batch_size)
        val_labels = tf.repeat(tf.range(self.n), self.k_val_ml)
        val_labels = tf.one_hot(val_labels, depth=self.n)
        val_labels = tf.stack([val_labels] * self.meta_batch_size)

        # print('debug\n\n\n')
        # print(train_labels)
        # print(val_labels)
        # print('debug\n\n\n')

        train_indices = [i // self.k + i % self.k * self.n for i in range(self.n * self.k)]
        val_indices = [i // self.k_val_ml + i % self.k_val_ml * self.n for i in range(self.n * self.k_val_ml)]

        def get_task():
            meta_batch_vectors = list()

            for meta_batch in range(self.meta_batch_size):
                vectors = self.generate_all_vectors()
                vectors = tf.reshape(tf.stack(vectors, axis=0), (-1, self.latent_dim))
                meta_batch_vectors.append(vectors)

            meta_batch_vectors = tf.stack(meta_batch_vectors)
            meta_batch_vectors = combine_first_two_axes(meta_batch_vectors)
            images = self.get_images_from_vectors(meta_batch_vectors)
            images = tf.image.resize(images, self.generated_image_shape[:2])
            images = tf.reshape(
                images,
                (self.meta_batch_size, self.n * (self.k + self.k_val_ml), *self.generated_image_shape)
            )

            train_ds = images[:, :self.n * self.k, ...]
            train_ds = tf.gather(train_ds, train_indices, axis=1)
            train_ds = tf.reshape(train_ds, (self.meta_batch_size, self.n, self.k, *self.generated_image_shape))

            val_ds = images[:, self.n * self.k:, ...]
            val_ds = combine_first_two_axes(val_ds)

            # ================
            val_imgs = list()
            for i in range(val_ds.shape[0]):
                val_image = val_ds[i, ...]
                tx = tf.random.uniform((), -5, 5, dtype=tf.int32)
                ty = tf.random.uniform((), -5, 5, dtype=tf.int32)
                transforms = [1, 0, -tx, 0, 1, -ty, 0, 0]
                val_image = tfa.image.transform(val_image, transforms, 'NEAREST')
                val_imgs.append(val_image)

            val_ds = tf.stack(val_imgs, axis=0)
            # ================
            val_ds = tf.reshape(val_ds, (self.meta_batch_size, self.n * self.k_val_ml, *self.generated_image_shape))
            val_ds = tf.gather(val_ds, val_indices, axis=1)
            val_ds = tf.reshape(val_ds, (self.meta_batch_size, self.n, self.k_val_ml, *self.generated_image_shape))

            yield (train_ds, val_ds), (train_labels, val_labels)

        dataset = tf.data.Dataset.from_generator(
            get_task,
            output_types=((tf.float32, tf.float32), (tf.float32, tf.float32))
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        setattr(dataset, 'steps_per_epoch', tf.data.experimental.cardinality(dataset))
        return dataset
