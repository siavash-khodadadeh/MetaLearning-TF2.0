import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from databases import OmniglotDatabase, MiniImagenetDatabase, CelebADatabase
from models.lasiummamlgan.database_parsers import OmniglotParser, MiniImagenetParser, CelebAGANParser
from models.lasiummamlgan.gan import GAN
from models.lasiummamlgan.maml_gan import MAMLGAN
from networks.maml_umtra_networks import MiniImagenetModel
import tensorflow_hub as hub


class MAMLGANProGAN(MAMLGAN):
    @tf.function
    def get_images_from_vectors(self, vectors):
        return self.gan(vectors)['default']

    def generate_all_vectors(self):
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
            noise = tf.random.normal(shape=class_vectors.shape, mean=0, stddev=1)
            noise = noise / tf.reshape(tf.norm(noise, axis=1), (noise.shape[0], 1))

            new_vectors = new_vectors + (noise - new_vectors) * 0.6

            vectors.append(new_vectors)

        return vectors

    def generate_all_vectors_p3(self):
        z = tf.random.normal((self.n, self.latent_dim))

        vectors = list()
        vectors.append(z)

        for i in range(self.k_ml + self.k_val_ml - 1):
            if (i + 1) % self.n == 0:
                new_z = z + tf.random.normal(shape=z.shape, mean=0, stddev=0.03)
                vectors.append(new_z)
            else:
                new_z = tf.stack(
                    [
                        z[0, ...] + (z[(i + 1) % self.n, ...] - z[0, ...]) * 0.37,
                        z[1, ...] + (z[(i + 2) % self.n, ...] - z[1, ...]) * 0.37,
                        z[2, ...] + (z[(i + 3) % self.n, ...] - z[2, ...]) * 0.37,
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

    def get_test_dataset(self,num_tasks, seed=-1):
        # dataset = super(MAMLGANProGAN, self).get_test_dataset(seed=seed)
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
    setattr(gan, 'parser', CelebAGANParser(shape=(84, 84, 3)))

    maml_gan = MAMLGANProGAN(
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
        k_test=5,  # k_test=5
        k_val_test=5,  # k_val_test=5
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='celeba_attributes_p1_std_0.5',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    maml_gan.visualize_meta_learning_task(shape, num_tasks_to_visualize=2)

    # For p2 0.6
    # 20k with 1e-3
    # 20k with 5e-4
    # 20k with 1e-4
    # For p1 0.1
    # 20k with 1e-3
    # 25k with 5e-3

    # p1 with 0.08
    #  20k with 1e-3
    #  70k with 5e-4
    # from 90K go with 1e-4
    # 105K works the best based on validation accuracy
    maml_gan.train(iterations=120000)
    maml_gan.evaluate(50, num_tasks=1000, seed=42, iterations_to_load_from=105000)
