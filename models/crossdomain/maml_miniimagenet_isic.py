import tensorflow as tf

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import MiniImagenetModel
from databases import MiniImagenetDatabase, ISICDatabase


class MiniImageNetIsicMAML(ModelAgnosticMetaLearningModel):
    def get_train_dataset(self):
        database = MiniImagenetDatabase()
        dataset = self.get_supervised_meta_learning_dataset(
            database.train_folders,
            n=self.n,
            k=self.k,
            k_validation=self.k_val_ml,
            meta_batch_size=self.meta_batch_size
        )
        return dataset

    def get_val_dataset(self):
        database = ISICDatabase()
        val_dataset = self.get_supervised_meta_learning_dataset(
            database.val_folders,
            n=self.n,
            k=self.k,
            k_validation=self.k_val_val,
            meta_batch_size=1,
            seed=self.val_seed,
        )
        val_dataset = val_dataset.repeat(-1)
        val_dataset = val_dataset.take(self.number_of_tasks_val)
        setattr(val_dataset, 'steps_per_epoch', self.number_of_tasks_val)
        return val_dataset

    def get_test_dataset(self, seed=-1):
        database = ISICDatabase()
        test_dataset = self.get_supervised_meta_learning_dataset(
            database.test_folders,
            n=self.n,
            k=self.k_test,
            k_validation=self.k_val_test,
            meta_batch_size=1,
            seed=seed
        )
        test_dataset = test_dataset.repeat(-1)
        test_dataset = test_dataset.take(self.number_of_tasks_test)
        setattr(test_dataset, 'steps_per_epoch', self.number_of_tasks_test)
        return test_dataset

    def initialize_network(self):
        model = self.network_cls(num_classes=self.n)
        model(tf.zeros(shape=(1, 84, 84, 3)))
        return model

    def get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, (84, 84))
            image = tf.cast(image, tf.float32)

            return image / 255.

        return parse_function


def run_isic():
    maml = MiniImageNetIsicMAML(
        database=None,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=5,
        meta_batch_size=1,
        num_steps_ml=1,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=5000,
        meta_learning_rate=0.001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=True,
        experiment_name='miniimagenet-isic',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    maml.train(iterations=300)
    maml.evaluate(50, seed=42)


if __name__ == '__main__':
    run_isic()
