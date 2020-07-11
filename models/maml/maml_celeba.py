from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import MiniImagenetModel
from databases import CelebADatabase


def run_celeba():
    celeba_database = CelebADatabase()
    maml = ModelAgnosticMetaLearningModel(
        database=celeba_database,
        network_cls=MiniImagenetModel,
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val=1,
        k_val_val=15,
        k_test=1,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=1,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=5000,
        meta_learning_rate=0.0001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='celeba'
    )

    maml.train(iterations=60000)
    maml.evaluate(50, num_tasks=1000, seed=42)


if __name__ == '__main__':
    run_celeba()
