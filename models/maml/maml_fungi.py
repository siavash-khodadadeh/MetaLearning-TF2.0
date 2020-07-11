from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import MiniImagenetModel
from databases import FungiDatabase


def run_fungi():
    fungi_database = FungiDatabase()

    maml = ModelAgnosticMetaLearningModel(
        database=fungi_database,
        network_cls=MiniImagenetModel,
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
        log_train_images_after_iteration=1000,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='fungi',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
    )

    maml.train(iterations=60000)
    maml.evaluate(50, seed=42, use_val_batch_statistics=True)
    maml.evaluate(50, seed=42, use_val_batch_statistics=False)


if __name__ == '__main__':
    run_fungi()
