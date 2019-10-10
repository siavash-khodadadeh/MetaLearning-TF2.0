import os

import tensorflow as tf

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks import SimpleModel
from tf_datasets import OmniglotDatabase


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
        )

    def get_root(self):
        return os.path.dirname(__file__)

    def get_train_dataset(self):
        return self.database.get_umtra_dataset(
            self.database.train_folders,
            n=self.n,
            meta_batch_size=self.meta_batch_size,
            augmentation_function=self.augmentation_function
        )

    def get_config_info(self):
        return f'umtra_' \
               f'model-{self.network_cls.name}_' \
               f'mbs-{self.meta_batch_size}_' \
               f'n-{self.n}_' \
               f'k-{self.k}_' \
               f'stp-{self.num_steps_ml}'


if __name__ == '__main__':
    omniglot_database = OmniglotDatabase(
        random_seed=-1,
        num_train_classes=1200,
        num_val_classes=100,
    )

    def augment(images):
        result = list()
        for image in images:
            image = tf.image.flip_left_right(image)
            result.append(image)

        return tf.stack(images)

    umtra = UMTRA(
        database=omniglot_database,
        network_cls=SimpleModel,
        n=5,
        meta_batch_size=32,
        num_steps_ml=5,
        lr_inner_ml=0.01,
        num_steps_validation=5,
        save_after_epochs=3,
        augmentation_function=augment
    )

    umtra.train(epochs=5)
