import tensorflow as tf

from models.base_data_loader import BaseDataLoader
from decorators import name_repr
from models.maml.maml import ModelAgnosticMetaLearningModel
from databases import MiniImagenetDatabase, CUBDatabase


@name_repr('VGG19')
def get_network(num_classes):
    base_model = tf.keras.applications.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False
    flatten = tf.keras.layers.Flatten(name='flatten')(base_model.output)
    fc = tf.keras.layers.Dense(num_classes, name='fc', activation=None)(flatten)
    model = tf.keras.models.Model(inputs=[base_model.input], outputs=[fc], name='VGG19')

    return model


class DatasetParser(BaseDataLoader):
    def get_parse_function(self):
        @tf.function
        def parse(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, (224, 224))
            image = tf.keras.applications.vgg19.preprocess_input(image)
            return image

        return parse

    def get_val_parse_function(self):
        return self.get_parse_function()

    def get_test_parse_function(self):
        return self.get_parse_function()

class TransferLearning(ModelAgnosticMetaLearningModel):
    def initialize_network(self):
        model = self.network_cls(num_classes=self.n)
        model(tf.zeros(shape=(1, 224, 224, 3)))
        model.summary()
        return model

    def init_data_loader(self, data_loader_cls):
        return super(TransferLearning, self).init_data_loader(DatasetParser)


def run_transfer_learning():
    cub_database = CUBDatabase()
    transfer_learning = TransferLearning(
        database=cub_database,
        network_cls=get_network,
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val_val=15,
        k_val=1,
        k_test=15,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=15000,
        meta_learning_rate=0.001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='fixed_vgg_19',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
    )
    print(f'k: {transfer_learning.k_test}')
    transfer_learning.evaluate(50, num_tasks=1000, seed=42, use_val_batch_statistics=False)
    transfer_learning.evaluate(50, num_tasks=1000, seed=42, use_val_batch_statistics=True)


if __name__ == '__main__':
    run_transfer_learning()
