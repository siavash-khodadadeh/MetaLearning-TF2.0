from models.reptile.reptile import Reptile

from networks.maml_umtra_networks import SimpleModel
from databases import OmniglotDatabase


def run_omniglot():
    omniglot_database = OmniglotDatabase(
        random_seed=47,
        num_train_classes=1200,
        num_val_classes=100,
    )

    reptile = Reptile(
        database=omniglot_database,
        network_cls=SimpleModel,
        n=5,
        k_ml=10,
        k_val_ml=10,
        k_val=5,
        k_val_val=15,
        k_test=5,
        k_val_test=15,
        meta_batch_size=5,
        num_steps_ml=5,
        lr_inner_ml=0.001,
        num_steps_validation=5,
        save_after_iterations=10000,
        meta_learning_rate=1.0,
        report_validation_frequency=1000,
        log_train_images_after_iteration=1000,
        num_tasks_val=100,
        clip_gradients=False,
        experiment_name='omniglot1',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    reptile.train(iterations=100000)
    reptile.evaluate(iterations=50, num_tasks=1000, use_val_batch_statistics=True, seed=42)


if __name__ == '__main__':
    # import tensorflow as tf
    # tf.config.experimental_run_functions_eagerly(True)
    run_omniglot()
