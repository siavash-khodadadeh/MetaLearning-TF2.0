from databases import AirplaneDatabase, FungiDatabase, CUBDatabase, DTDDatabase, VGGFlowerDatabase, \
    MiniImagenetDatabase, Omniglot84x84Database
from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import MiniImagenetModel


def run_airplane():
    airplane_database = AirplaneDatabase()

    maml = ModelAgnosticMetaLearningModel(
        database=airplane_database,
        # test_database=Omniglot84x84Database(random_seed=47, num_train_classes=1200, num_val_classes=100),
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
        experiment_name='airplane',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
    )

    # maml.train(iterations=60000)
    maml.evaluate(50, num_tasks=1000, seed=42, use_val_batch_statistics=True)
    maml.evaluate(50, num_tasks=1000, seed=42, use_val_batch_statistics=False)


if __name__ == '__main__':
    run_airplane()
