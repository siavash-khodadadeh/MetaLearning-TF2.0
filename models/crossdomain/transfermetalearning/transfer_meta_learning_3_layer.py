from typing import Callable

import tensorflow as tf
from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import get_transfer_net
from databases import MiniImagenetDatabase, ISICDatabase, EuroSatDatabase, PlantDiseaseDatabase


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

class EuroSatDatabasePreProcess(EuroSatDatabase):
    def _get_parse_function(self) -> Callable:
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, self.input_shape[:2])
            image = tf.keras.applications.vgg16.preprocess_input(image)
            return image

        return parse_function

class ISICDatabasePreProcess(ISICDatabase):
    def _get_parse_function(self) -> Callable:
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, self.input_shape[:2])
            image = tf.keras.applications.vgg16.preprocess_input(image)
            return image

        return parse_function

class PlantDiseaseDatabasePreProcess(PlantDiseaseDatabase):
    def _get_parse_function(self) -> Callable:
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, self.input_shape[:2])
            image = tf.keras.applications.vgg16.preprocess_input(image)
            if image.shape[-1] != 3:
                image = image[:,:,:3]
            return image

        return parse_function

def run_cross_domain():
    mini_imagenet_database = MiniImagenetDatabasePreProcess(input_shape=(224, 224, 3))
    transfer_learning = TransferLearning(
        database=mini_imagenet_database,
#         database=EuroSatDatabasePreProcess(input_shape=(224, 224, 3)),
#         database=PlantDiseaseDatabasePreProcess(input_shape=(224, 224, 3)),
#         database=ISICDatabasePreProcess(input_shape=(224, 224, 3)),
        network_cls=get_transfer_net,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=5,
        meta_batch_size=1,
        num_steps_ml=1,
        lr_inner_ml=0.001,
        num_steps_validation=5,
        save_after_iterations=1500,
        meta_learning_rate=0.0001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=100,
        clip_gradients=True,
        experiment_name='meta_mini_imagenet_3_layer',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
    )


    transfer_learning.train(iterations=60000)
#     from datetime import datetime
#     begin_time = datetime.now()
#     transfer_learning.evaluate(200, seed=14)
#     print(datetime.now() - begin_time)

if __name__ == '__main__':
    run_cross_domain()