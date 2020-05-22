import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from models.maml.maml import ModelAgnosticMetaLearningModel


class MAML_VAE(ModelAgnosticMetaLearningModel):
    def __init__(self, vae, *args, **kwargs):
        super(MAML_VAE, self).__init__(*args, **kwargs)
        self.vae = vae

    def get_network_name(self):
        return self.model.name

    def get_parse_function(self):
        return self.vae.parser.get_parse_fn()

    def visualize_meta_learning_task(self, shape, num_tasks_to_visualize=1):
        import matplotlib.pyplot as plt

        dataset = self.get_train_dataset()
        for item in dataset.take(num_tasks_to_visualize):
            fig, axes = plt.subplots(self.k + self.k_val_ml, self.n)
            fig.set_figwidth(self.k + self.k_val_ml)
            fig.set_figheight(self.n)

            (train_ds, val_ds), (_, _) = item
            # Get the first meta batch
            train_ds = train_ds[0, ...]
            val_ds = val_ds[0, ...]
            if shape[2] == 1:
                train_ds = train_ds[..., 0]
                val_ds = val_ds[..., 0]

            for n in range(self.n):
                for k in range(self.k):
                    axes[k, n].imshow(train_ds[n, k, ...], cmap='gray')

                for k in range(self.k_val_ml):
                    axes[k + self.k, n].imshow(val_ds[n, k, ...], cmap='gray')

            plt.show()

    def generate_new_z_from_z_data(self, z, z_mean, z_log_var):
        # return self.vae.sample(z_mean, z_log_var)
        # return z + tf.random.normal(shape=z.shape, mean=0, stddev=0.5)

        return z + tf.random.normal(shape=z.shape, mean=0, stddev=1.0)

    def get_train_dataset(self):
        def generate_new_samples_with_vae(instances):
            # from datetime import datetime
            train_indices = [i // self.k + i % self.k * self.n for i in range(self.n * self.k)]
            val_indices = [
                self.n * self.k + i // self.k_val_ml + i % self.k_val_ml * self.n
                for i in range(self.n * self.k_val_ml)
            ]

            # TODO test speed change with this tf.function and without it.
            # @tf.function
            def f(instances):
                # current_time = datetime.now()
                z_mean, z_log_var, z = self.vae.encode(instances)
                # print(f'encode time spent: {datetime.now() - current_time}')

                # current_time = datetime.now()
                new_zs = list()
                for i in range(self.k + self.k_val_ml - 1):
                    new_z = self.generate_new_z_from_z_data(z, z_mean, z_log_var)
                    new_zs.append(new_z)
                new_zs = tf.concat(new_zs, axis=0)
                # print(f'generate z time spent: {datetime.now() - current_time}')

                # current_time = datetime.now()
                new_instances = self.vae.decode(new_zs)
                # print(f'decode time spent: {datetime.now() - current_time}')

                # current_time = datetime.now()
                new_instances = tf.concat((instances, new_instances), axis=0)

                train_instances = tf.gather(new_instances, train_indices, axis=0)
                val_instances = tf.gather(new_instances, val_indices, axis=0)

                # Translation
                # tx = tf.random.uniform((), -5, 5, dtype=tf.int32)
                # ty = tf.random.uniform((), -5, 5, dtype=tf.int32)
                # transforms = [1, 0, -tx, 0, 1, -ty, 0, 0]
                # val_instances = tfa.image.transform(val_instances, transforms, 'NEAREST')

                # Rotation
                # angles = tf.random.uniform(
                #     shape=(self.n * self.k_val_ml,),
                #     minval=tf.constant(-np.pi),
                #     maxval=tf.constant(np.pi)
                # )
                # val_instances = tfa.image.rotate(val_instances, angles)
                # print(f' gather time spent: {datetime.now() - current_time}')

                # print()
                return (
                    tf.reshape(train_instances, (self.n, self.k, *train_instances.shape[1:])),
                    tf.reshape(val_instances, (self.n, self.k_val_ml, *val_instances.shape[1:])),
                )

            return tf.py_function(f, inp=[instances], Tout=[tf.float32, tf.float32])

        instances = self.database.get_all_instances(partition_name='train')
        dataset = tf.data.Dataset.from_tensor_slices(instances)
        dataset = dataset.map(self.get_parse_function())
        dataset = dataset.shuffle(buffer_size=len(instances))
        dataset = dataset.batch(self.n, drop_remainder=True)

        dataset = dataset.map(generate_new_samples_with_vae)
        labels_dataset = self.make_labels_dataset(self.n, self.k, self.k_val_ml, one_hot_labels=True)

        dataset = tf.data.Dataset.zip((dataset, labels_dataset))
        dataset = dataset.batch(self.meta_batch_size)

        setattr(dataset, 'steps_per_epoch', tf.data.experimental.cardinality(dataset))
        return dataset




