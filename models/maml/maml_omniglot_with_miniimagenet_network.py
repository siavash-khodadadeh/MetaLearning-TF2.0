from typing import Callable

import tensorflow as tf

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import MiniImagenetModel
from databases import Omniglot84x84Database, AirplaneDatabase, FungiDatabase, CUBDatabase, DTDDatabase, \
    VGGFlowerDatabase, MiniImagenetDatabase


def run_omniglot():
    omniglot_database = Omniglot84x84Database(
        random_seed=47,
        num_train_classes=1200,
        num_val_classes=100,
    )

    maml = ModelAgnosticMetaLearningModel(
        database=omniglot_database,
        test_database=MiniImagenetDatabase(),
        network_cls=MiniImagenetModel,
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val=1,
        k_val_val=15,
        k_test=5,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=15000,
        meta_learning_rate=0.001,
        report_validation_frequency=1000,
        log_train_images_after_iteration=1000,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='omniglot_84x84',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    maml.train(iterations=60000)
    maml.evaluate(iterations=50, num_tasks=1000, use_val_batch_statistics=True, seed=42)
    maml.evaluate(iterations=50, num_tasks=1000, use_val_batch_statistics=False, seed=42)


if __name__ == '__main__':
    run_omniglot()
