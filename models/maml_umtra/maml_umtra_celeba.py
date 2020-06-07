from databases import CelebADatabase
from models.maml_umtra.maml_umtra import MAMLUMTRA
from networks.maml_umtra_networks import MiniImagenetModel

if __name__ == '__main__':
    # import tensorflow as tf
    # tf.config.experimental_run_functions_eagerly(True)

    celeba_database = CelebADatabase()

    maml_umtra = MAMLUMTRA(
        database=celeba_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=5000,
        meta_learning_rate=0.001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=True,
        experiment_name='celeba',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    shape = (84, 84, 3)
    maml_umtra.visualize_umtra_task(shape, num_tasks_to_visualize=2)

    maml_umtra.train(iterations=5000)
    maml_umtra.evaluate(50, seed=42)
