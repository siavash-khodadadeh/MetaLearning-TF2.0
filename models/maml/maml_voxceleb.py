import tensorflow as tf

from databases import VoxCelebDatabase
from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import VoxCelebModel


def run_voxceleb():
    voxceleb_database = VoxCelebDatabase()
    maml = ModelAgnosticMetaLearningModel(
        database=voxceleb_database,
        network_cls=VoxCelebModel,
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val=1,
        k_val_val=15,
        k_test=1,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=15000,
        meta_learning_rate=0.001,
        report_validation_frequency=1000,
        log_train_images_after_iteration=-1,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='voxceleb3',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
    )

    # maml.train(iterations=60040)
    maml.evaluate(50, num_tasks=1000, seed=42, use_val_batch_statistics=True)


if __name__ == '__main__':
    # tf.config.experimental_run_functions_eagerly(True)
    run_voxceleb()
