import os

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks import SimpleModel, MiniImagenetModel
from tf_datasets import OmniglotDatabase, MiniImagenetDatabase


class UMTRA(ModelAgnosticMetaLearningModel):
    def __init__(
            self,
            database,
            network_cls,
            n,
            meta_batch_size,
            num_steps_ml,
            lr_inner_ml,
            num_steps_validation,
            save_after_epochs,
            meta_learning_rate,
            report_validation_frequency,
            log_train_images_after_iteration,  # Set to -1 if you do not want to log train images.
            least_number_of_tasks_val_test=-1,  # Make sure the val and test dataset pick at least this many tasks.
            clip_gradients=False,
            augmentation_function=None
    ):
        self.augmentation_function = augmentation_function
        super(UMTRA, self).__init__(
            database=database,
            network_cls=network_cls,
            n=n,
            k=1,
            meta_batch_size=meta_batch_size,
            num_steps_ml=num_steps_ml,
            lr_inner_ml=lr_inner_ml,
            num_steps_validation=num_steps_validation,
            save_after_epochs=save_after_epochs,
            meta_learning_rate=meta_learning_rate,
            report_validation_frequency=report_validation_frequency,
            log_train_images_after_iteration=log_train_images_after_iteration,
            least_number_of_tasks_val_test=least_number_of_tasks_val_test,
            clip_gradients=clip_gradients
        )

    def get_root(self):
        return os.path.dirname(__file__)

    def get_train_dataset(self):
        dataset = self.database.get_umtra_dataset(
            self.database.train_folders,
            n=self.n,
            meta_batch_size=self.meta_batch_size,
            augmentation_function=self.augmentation_function
        )

        return dataset

    def get_config_info(self):
        return f'umtra_' \
               f'model-{self.network_cls.name}_' \
               f'mbs-{self.meta_batch_size}_' \
               f'n-{self.n}_' \
               f'k-{self.k}_' \
               f'stp-{self.num_steps_ml}'

def run_omniglot():
    @tf.function
    def augment(images):
        result = list()
        num_imgs = 1
        for i in range(images.shape[0]):
            image = tf.reshape(images[i], (num_imgs, 28, 28, 1))
            random_map = tf.random.uniform(shape=tf.shape(image), minval=0, maxval=2, dtype=tf.int32)
            random_map = tf.cast(random_map, tf.float32)
            image = tf.minimum(image, random_map)

            base_ = tf.convert_to_tensor(np.tile([1, 0, 0, 0, 1, 0, 0, 0], [num_imgs, 1]), dtype=tf.float32)
            mask_ = tf.convert_to_tensor(np.tile([0, 0, 1, 0, 0, 1, 0, 0], [num_imgs, 1]), dtype=tf.float32)
            random_shift_ = tf.random.uniform([num_imgs, 8], minval=-6., maxval=6., dtype=tf.float32)
            transforms_ = base_ + random_shift_ * mask_
            augmented_data = tfa.image.transform(images=image, transforms=transforms_)
            result.append(augmented_data)

        return tf.stack(result)

    omniglot_database = OmniglotDatabase(
        random_seed=-1,
        num_train_classes=1200,
        num_val_classes=100,
    )

    umtra = UMTRA(
        database=omniglot_database,
        network_cls=SimpleModel,
        n=5,
        meta_batch_size=32,
        num_steps_ml=5,
        lr_inner_ml=0.01,
        num_steps_validation=5,
        save_after_epochs=5,
        meta_learning_rate=0.001,
        log_train_images_after_iteration=10,
        report_validation_frequency=1,
        augmentation_function=augment
    )

    umtra.train(epochs=10)
    # umtra.evaluate(iterations=50)


def run_mini_imagenet():
    mini_imagenet_database = MiniImagenetDatabase(random_seed=-1)

    @tf.function
    def augment(images):
        result = list()
        for i in range(images.shape[0]):
            image = tf.image.flip_left_right(images[i, ...])
            result.append(image)

        return tf.stack(result)

    umtra = UMTRA(
        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_epochs=500,
        meta_learning_rate=0.001,
        report_validation_frequency=50,
        log_train_images_after_iteration=1000,
        least_number_of_tasks_val_test=50,
        clip_gradients=True,
        augmentation_function=augment
    )

    umtra.train(epochs=100)
    # umtra.evaluate(iterations=50)


if __name__ == '__main__':
    run_omniglot()
    # run_mini_imagenet()
