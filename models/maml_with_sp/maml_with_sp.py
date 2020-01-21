import os

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import MiniImagenetModel
from tf_datasets import CelebADatabase, MiniImagenetDatabase


class MAMLWithSP(ModelAgnosticMetaLearningModel):
    def get_train_dataset(self):
        return self.database.get_sp_meta_learning_dataset(
            self.database.train_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=self.meta_batch_size,
            features_name='vgg19_last_hidden_layer_train'
        )

    def get_root(self):
        return os.path.dirname(__file__)


def run_mini_imagenet():
    mini_imagenet_database = MiniImagenetDatabase(random_seed=30)

    maml = MAMLWithSP(
        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.01,
        num_steps_validation=5,
        save_after_epochs=500,
        meta_learning_rate=0.001,
        report_validation_frequency=50,
        log_train_images_after_iteration=1000,
        least_number_of_tasks_val_test=60,
        clip_gradients=True,
        experiment_name='mini_imagenet_sp_seed30',
    )

    maml.train(epochs=7501)
    maml.evaluate(5)


def run_celeba():
    # test accuracy: 0.85

    celeba_dataset = CelebADatabase(random_seed=30)
    maml_with_sp = MAMLWithSP(
        database=celeba_dataset,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_epochs=20,
        meta_learning_rate=0.001,
        log_train_images_after_iteration=1,
        least_number_of_tasks_val_test=50,
        report_validation_frequency=400,
        experiment_name='furthest_point_seed30',
    )

    maml_with_sp.train(epochs=81)
    maml_with_sp.evaluate(iterations=50)


if __name__ == '__main__':
    run_mini_imagenet()
    # run_celeba()
