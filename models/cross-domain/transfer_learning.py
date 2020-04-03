from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import get_transfer_net
from tf_datasets import MiniImagenetDatabase


class TransferLearning(ModelAgnosticMetaLearningModel):
    pass


def run_mini_imagenet():
    mini_imagenet_database = MiniImagenetDatabase()
    transfer_learning = TransferLearning(
        database=mini_imagenet_database,
        network_cls=get_transfer_net,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=50,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=15000,
        meta_learning_rate=0.001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=True,
        experiment_name='mini_imagenet'
    )

    # transfer_learning.train(iterations=60000)
    transfer_learning.evaluate(50, seed=14)


if __name__ == '__main__':
    run_mini_imagenet()
