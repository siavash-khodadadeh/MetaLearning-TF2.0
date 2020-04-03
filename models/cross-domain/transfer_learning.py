from typing import Callable

import tensorflow as tf

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import get_transfer_net
from tf_datasets import MiniImagenetDatabase


class TransferLearning(ModelAgnosticMetaLearningModel):
    pass


class MiniImagenetDatabasePreProcess(MiniImagenetDatabase):
    def _get_parse_function(self) -> Callable:
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, self.input_shape[:2])
            image = tf.keras.applications.vgg16.preprocess_input(image)
            return image

        return parse_function


def run_mini_imagenet():
    mini_imagenet_database = MiniImagenetDatabasePreProcess(input_shape=(224, 224, 3))
    transfer_learning = TransferLearning(
        database=mini_imagenet_database,
        network_cls=get_transfer_net,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=5,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.005,
        num_steps_validation=5,
        save_after_iterations=15000,
        meta_learning_rate=0.001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=100,
        clip_gradients=True,
        experiment_name='fixed_vgg_16_with_three_layers'
    )

    # transfer_learning.train(iterations=60000)
    transfer_learning.evaluate(50, seed=14)


if __name__ == '__main__':
    run_mini_imagenet()
