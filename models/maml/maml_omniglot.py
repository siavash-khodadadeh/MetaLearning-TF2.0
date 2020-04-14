from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import SimpleModel
from tf_datasets import OmniglotDatabase


def run_omniglot():
    omniglot_database = OmniglotDatabase(
        random_seed=47,
        num_train_classes=1200,
        num_val_classes=100,
    )

    maml = ModelAgnosticMetaLearningModel(
        database=omniglot_database,
        network_cls=SimpleModel,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=1,
        meta_batch_size=32,
        num_steps_ml=5,
        lr_inner_ml=0.4,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=50,
        log_train_images_after_iteration=200,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=False,
        experiment_name='omniglot',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    maml.train(iterations=5000)
    maml.evaluate(iterations=50, use_val_batch_statistics=True, seed=42)


if __name__ == '__main__':
    run_omniglot()
