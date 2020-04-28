from models.crossdomain.transfermetalearning.transfer_meta_learning_vgg16 import TransferMetaLearningVGG16
from databases import MiniImagenetDatabase, EuroSatDatabase


def run_transfer_meta_learning():
    mini_imagenet_database = MiniImagenetDatabase()
    euro_sat_database = EuroSatDatabase()
    tml = TransferMetaLearningVGG16(
        database=mini_imagenet_database,
        val_database=euro_sat_database,
        target_database=euro_sat_database,
        network_cls=None,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=1,
        meta_batch_size=4,
        num_steps_ml=1,
        lr_inner_ml=0.001,
        num_steps_validation=5,
        save_after_iterations=1500,
        meta_learning_rate=0.0001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=100,
        clip_gradients=True,
        experiment_name='transfer_meta_learning_mini_imagenet_euro_sat',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
        random_layer_initialization_seed=42,
        num_trainable_layers=3,
    )

    tml.train(iterations=6000)
    tml.evaluate(50, seed=42, use_val_batch_statistics=True)
    tml.evaluate(50, seed=42, use_val_batch_statistics=False)


if __name__ == '__main__':
    run_transfer_meta_learning()
