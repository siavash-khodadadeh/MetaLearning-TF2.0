import os

import tensorflow as tf

from models.maml.maml import ModelAgnosticMetaLearningModel
from tf_datasets import OmniglotDatabase, MiniImagenetDatabase
from networks import SimpleModel, MiniImagenetModel


class MAMLRandomSelection(ModelAgnosticMetaLearningModel):
    def get_root(self):
        return os.path.dirname(__file__)

    def get_train_dataset(self):
        dataset = self.database.get_random_dataset(
            self.database.train_folders,
            n=self.n,
            meta_batch_size=self.meta_batch_size
        )
        # steps_per_epoch = dataset.steps_per_epoch
        # dataset = dataset.prefetch(1)
        # setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset


def run_omniglot():
    omniglot_database = OmniglotDatabase(
        random_seed=47,
        num_train_classes=1200,
        num_val_classes=100,
    )

    maml = MAMLRandomSelection(
        database=omniglot_database,
        network_cls=SimpleModel,
        n=5,
        k=1,
        meta_batch_size=32,
        num_steps_ml=1,
        lr_inner_ml=0.4,
        num_steps_validation=10,
        save_after_epochs=500,
        meta_learning_rate=0.001,
        report_validation_frequency=10,
        log_train_images_after_iteration=-1,
    )

    maml.train(epochs=4000)
    maml.evaluate(iterations=50)


def run_mini_imagenet():
    mini_imagenet_database = MiniImagenetDatabase(random_seed=-1)

    maml = MAMLRandomSelection(
        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.4,
        num_steps_validation=5,
        save_after_epochs=500,
        meta_learning_rate=0.001,
        report_validation_frequency=50,
        log_train_images_after_iteration=1000,
        least_number_of_tasks_val_test=50,
        clip_gradients=True
    )

    maml.train(epochs=24000)
    maml.evaluate(50)


if __name__ == '__main__':
    run_omniglot()
    # run_mini_imagenet()
