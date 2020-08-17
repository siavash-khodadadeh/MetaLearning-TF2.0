import random

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
import numpy as np

from models.maml.maml import ModelAgnosticMetaLearningModel


class MAMLUMTRA(ModelAgnosticMetaLearningModel):
    def __init__(self, *args, **kwargs):
        super(MAMLUMTRA, self).__init__(*args, **kwargs)
        handle = 'https://tfhub.dev/google/image_augmentation/nas_imagenet/1'
        self.hub_model = hub.load(handle).signatures['from_decoded_images']

    def get_network_name(self):
        return self.model.name

    def visualize_umtra_task(self, shape, num_tasks_to_visualize=1):
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

    def augment(self, images, rotation_index):
        new_images = images

        # Translation
        # CELEBA
        tx = tf.random.uniform((), -35, 35, dtype=tf.int32)
        ty = tf.random.uniform((), -35, 35, dtype=tf.int32)

        # Omniglot
        # tx = tf.random.uniform((), -5, 5, dtype=tf.int32)
        # ty = tf.random.uniform((), -5, 5, dtype=tf.int32)
        transforms = [1, 0, -tx, 0, 1, -ty, 0, 0]
        new_images = tfa.image.transform(new_images, transforms, 'NEAREST')

        # Rotation
        # angles = tf.random.uniform(
        #     shape=(self.n * self.k,),
        #     minval=tf.constant(-np.pi / 6),
        #     maxval=tf.constant(np.pi / 6)
        # )
        # new_images = tfa.image.rotate(new_images, angles)

        # Zeroing

        # auto augment


        #
        # new_images = augmentation_module({
        #     'encoded_images': new_images,
        #     'image_size': (84, 84),
        #     'augmentation': False,
        # })

        return new_images

    def get_train_dataset(self):
        train_indices = [i // self.k + i % self.k * self.n for i in range(self.n * self.k)]
        val_indices = [
            self.n * self.k + i // self.k_val_ml + i % self.k_val_ml * self.n
            for i in range(self.n * self.k_val_ml)
        ]

        def generate_new_samples_with_auto_augment(instances):
            new_instances = list()
            for i in range(self.k + self.k_val_ml - 1):
                new_instance = self.hub_model(
                    images=instances,
                    image_size=tf.constant([84, 84]),
                    augmentation=tf.constant(True)
                )['default']
                new_instances.append(new_instance)

            new_instances = tf.concat(new_instances, axis=0)
            all_instances = tf.concat((instances, new_instances), axis=0)
            train_instances = tf.gather(all_instances, train_indices, axis=0)
            val_instances = tf.gather(all_instances, val_indices, axis=0)

            return (
                tf.reshape(train_instances, (self.n, self.k, *train_instances.shape[1:])),
                tf.reshape(val_instances, (self.n, self.k_val_ml, *val_instances.shape[1:])),
            )

        instances = self.database.get_all_instances(partition_name='train')
        random.shuffle(instances)

        dataset = tf.data.Dataset.from_tensor_slices(instances)
        dataset = dataset.map(self.get_parse_function())
        # dataset = dataset.shuffle(buffer_size=len(instances))
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.n, drop_remainder=True)

        dataset = dataset.map(generate_new_samples_with_auto_augment)
        # dataset = dataset.map(generate_new_samples_with_augmentation)

        labels_dataset = self.make_labels_dataset(self.n, self.k, self.k_val_ml, one_hot_labels=True)

        dataset = tf.data.Dataset.zip((dataset, labels_dataset))
        dataset = dataset.batch(self.meta_batch_size)

        setattr(dataset, 'steps_per_epoch', tf.data.experimental.cardinality(dataset))
        return dataset




