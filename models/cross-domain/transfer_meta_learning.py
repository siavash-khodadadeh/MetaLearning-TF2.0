from typing import Callable

import tensorflow as tf

from decorators import name_repr
from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import get_transfer_net
from tf_datasets import MiniImagenetDatabase, ISICDatabase


@name_repr('VGG16')
def get_network(num_classes):
    base_model = getattr(tf.keras.applications, 'VGG16')(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False
    flatten = tf.keras.layers.Flatten(name='flatten')(base_model.output)
    fc = tf.keras.layers.Dense(num_classes, name='fc', activation=None)(flatten)
    model = tf.keras.models.Model(inputs=[base_model.input], outputs=[fc], name='VGG16')

    return model


class TransferMetaLearning(ModelAgnosticMetaLearningModel):
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
        model(tf.zeros(shape=(1, 224, 224, 3)))
        return model

    def get_parse_function(self) -> Callable:
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, (224, 224))
            image = tf.keras.applications.vgg16.preprocess_input(image)
            return image

        return parse_function


def run_transfer_learning_miniimagent_isic():
    transfer_meta_learning = TransferMetaLearning(
        database=None,
        network_cls=get_transfer_net,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=1,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.005,
        num_steps_validation=5,
        save_after_iterations=15000,
        meta_learning_rate=0.001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=True,
        experiment_name='fixed_vgg_16_with_three_layers',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
    )

    transfer_meta_learning.train(iterations=60000)
    transfer_meta_learning.evaluate(50, seed=42, use_val_batch_statistics=True)
    transfer_meta_learning.evaluate(50, seed=42, use_val_batch_statistics=False)


if __name__ == '__main__':
    run_transfer_learning_miniimagent_isic()
