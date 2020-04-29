from databases.meta_dataset import AirplaneDatabase
from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import MiniImagenetModel


def run_airplane():
    airplane_database = AirplaneDatabase()

    maml = ModelAgnosticMetaLearningModel(
        database=airplane_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=1,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=15000,
        meta_learning_rate=0.001,
        report_validation_frequency=1000,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=True,
        experiment_name='airplane',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
    )

    maml.train(iterations=60040)
    maml.evaluate(50, seed=42, use_val_batch_statistics=True)
    maml.evaluate(50, seed=42, use_val_batch_statistics=False)


if __name__ == '__main__':
    run_airplane()
