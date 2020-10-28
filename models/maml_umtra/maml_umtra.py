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
            fig, axes = plt.subplots(self.k_ml + self.k_val_ml, self.n)
            fig.set_figwidth(self.k_ml + self.k_val_ml)
            fig.set_figheight(self.n)

            (train_ds, val_ds), (_, _) = item
            # Get the first meta batch
            train_ds = train_ds[0, ...]
            val_ds = val_ds[0, ...]
            if shape[2] == 1:
                train_ds = train_ds[..., 0]
                val_ds = val_ds[..., 0]

            for n in range(self.n):
                for k in range(self.k_ml):
                    axes[k, n].imshow(train_ds[n, k, ...], cmap='gray')

                for k in range(self.k_val_ml):
                    axes[k + self.k_ml, n].imshow(val_ds[n, k, ...], cmap='gray')

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
        return self.data_loader.get_umtra_dataset(
            folders=self.database.train_folders,
            n=self.n,
            k=self.k_ml,
            k_validation=self.k_val_ml,
            meta_batch_size=self.meta_batch_size,
            one_hot_labels=True,
            reshuffle_each_iteration=True,
        )
