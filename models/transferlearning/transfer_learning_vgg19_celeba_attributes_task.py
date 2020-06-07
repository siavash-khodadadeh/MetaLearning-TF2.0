import tensorflow as tf

from decorators import name_repr
from models.maml.maml import ModelAgnosticMetaLearningModel
from databases import CelebADatabase


@name_repr('VGG16')
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


class TransferLearning(ModelAgnosticMetaLearningModel):
    def initialize_network(self):
        model = self.network_cls(num_classes=self.n)
        model(tf.zeros(shape=(1, 224, 224, 3)))
        return model

    def get_test_dataset(self, seed=-1):
        # dataset = super(MAMLGANProGAN, self).get_test_dataset(seed=seed)
        test_dataset = self.database.get_attributes_task_dataset(
            partition='test',
            k=self.k_test,
            k_val=self.k_val_test,
            meta_batch_size=1,
            parse_fn=self.get_parse_function(),
            seed=seed,
            default_shape=(224, 224, 3)
        )
        test_dataset = test_dataset.repeat(-1)
        test_dataset = test_dataset.take(self.number_of_tasks_test)
        return test_dataset

    def get_parse_function(self):
        def parse_function(example_address):
            example_address = tf.squeeze(example_address)
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, (224, 224))
            image = tf.keras.applications.vgg19.preprocess_input(image)
            return image

        return parse_function


if __name__ == '__main__':
    celeba_database = CelebADatabase()
    transfer_learning = TransferLearning(
        database=celeba_database,
        network_cls=get_network,
        n=2,
        k=5,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=5,
        k_test=5,
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
        experiment_name='fixed_vgg_19',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
    )
    transfer_learning.evaluate(50, seed=42, use_val_batch_statistics=True)
    # transfer_learning.evaluate(50, seed=42, use_val_batch_statistics=False)




