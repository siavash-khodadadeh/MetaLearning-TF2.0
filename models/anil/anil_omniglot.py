from models.anil.anil import ANIL
from networks.maml_umtra_networks import SimpleModel
from databases import OmniglotDatabase


def run_omniglot():
    omniglot_database = OmniglotDatabase(
        random_seed=47,
        num_train_classes=1200,
        num_val_classes=100,
    )

    anil = ANIL(
        database=omniglot_database,
        network_cls=SimpleModel,
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val=1,
        k_val_val=15,
        k_test=1,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.4,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=50,
        log_train_images_after_iteration=200,
        num_tasks_val=100,
        clip_gradients=False,
        experiment_name='omniglot3',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
        set_of_frozen_layers={'conv1', 'conv2', 'conv3', 'conv4', 'bn1', 'bn2', 'bn3', 'bn4'}
    )

    anil.train(iterations=5000)
    anil.evaluate(iterations=50, num_tasks=1000, use_val_batch_statistics=True, seed=42)


if __name__ == '__main__':
    run_omniglot()
