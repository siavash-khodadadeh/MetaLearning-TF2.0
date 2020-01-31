import os

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import MiniImagenetModel
from tf_datasets import CelebADatabase, MiniImagenetDatabase


class MAMLWithSP(ModelAgnosticMetaLearningModel):
    def get_train_dataset(self):
        dataset = self.database.get_sp_meta_learning_dataset(
            self.database.train_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=self.meta_batch_size,
            features_name='vgg19_last_hidden_layer_train'
        )
        return dataset

    # def get_test_dataset(self):
    #     test_dataset = self.database.get_supervised_meta_learning_dataset(
    #         self.database.test_folders,
    #         n=self.n,
    #         k=self.k,
    #         k_validation=15,
    #         meta_batch_size=1,
    #     )
    #     steps_per_epoch = max(test_dataset.steps_per_epoch, self.least_number_of_tasks_val_test)
    #     test_dataset = test_dataset.repeat(-1)
    #     test_dataset = test_dataset.take(steps_per_epoch)
    #     setattr(test_dataset, 'steps_per_epoch', steps_per_epoch)
    #     return test_dataset

    # def get_test_dataset(self):
    #     test_dataset = self.database.get_sp_meta_learning_dataset(
    #         self.database.train_folders,
    #         n=self.n,
    #         k=self.k,
    #         meta_batch_size=1,
    #         features_name='vgg19_last_hidden_layer_train'
    #     )
    #
    #     steps_per_epoch = max(test_dataset.steps_per_epoch, self.least_number_of_tasks_val_test)
    #     test_dataset = test_dataset.repeat(-1)
    #     test_dataset = test_dataset.take(steps_per_epoch)
    #     setattr(test_dataset, 'steps_per_epoch', steps_per_epoch)
    #     return test_dataset

    def get_root(self):
        return os.path.dirname(__file__)


def run_mini_imagenet():
    # for epoch_number in (2500, 5000, 7500, 10000, 12500, 15000, 17500, 21000):
    # for epoch_number in (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)[::-1]:
    # for epoch_number in (1000, 1500, 2000):

    mini_imagenet_database = MiniImagenetDatabase(random_seed=-1)

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
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        least_number_of_tasks_val_test=100,
        clip_gradients=True,
        experiment_name='mini_imagenet_no_confusion_sp_and_random_random_seed_-1',
        # experiment_name='mini_imagenet_difficult_tasks',
        # experiment_name='mini_imagenet_sp_random_seed_-1',
        # experiment_name='mini_imagenet_exact_maml_random_seed_-1',
    )

    # maml.train(epochs=21501)
    maml.train(epochs=4001)
    # print(f'epoch number: {epoch_number}')
    # maml.evaluate(50, epochs_to_load_from=epoch_number)


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
