from databases import OmniglotDatabase, Omniglot84x84Database
from models.maml_umtra.maml_umtra import MAMLUMTRA
from networks.maml_umtra_networks import SimpleModel

if __name__ == '__main__':
    # import tensorflow as tf
    # tf.config.experimental_run_functions_eagerly(True)

    omniglot_database = Omniglot84x84Database(random_seed=47, num_train_classes=1200, num_val_classes=100)

    maml_umtra = MAMLUMTRA(
        database=omniglot_database,
        network_cls=SimpleModel,
        n=5,
        k_ml=1,
        k_val_ml=1,
        k_val=1,
        k_val_val=15,
        k_test=1,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=5000,
        meta_learning_rate=0.001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='omniglot',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    shape = (28, 28, 1)
    # maml_umtra.visualize_umtra_task(shape, num_tasks_to_visualize=2)

    maml_umtra.train(iterations=60000)
    maml_umtra.evaluate(50, seed=42, num_tasks=1000)
