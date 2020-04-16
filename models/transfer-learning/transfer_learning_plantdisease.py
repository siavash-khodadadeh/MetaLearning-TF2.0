import tensorflow as tf

from decorators import name_repr
from models.maml.maml import ModelAgnosticMetaLearningModel
from tf_datasets import PlantDiseaseDatabase


@name_repr('VGG16')
def get_network(num_classes, random_layer_initialization_seed):
    base_model = getattr(tf.keras.applications, 'VGG16')(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False
    flatten = tf.keras.layers.Flatten(name='flatten')(base_model.output)

    tf.random.set_seed(random_layer_initialization_seed)
    fc = tf.keras.layers.Dense(num_classes, name='fc', activation=None)(flatten)
    tf.random.set_seed(None)

    model = tf.keras.models.Model(inputs=[base_model.input], outputs=[fc], name='VGG16')

    return model


class TransferLearning(ModelAgnosticMetaLearningModel):
    def __init__(self, random_layer_initialization_seed, *args, **kwargs):
        self.random_layer_initialization_seed = random_layer_initialization_seed
        super(TransferLearning, self).__init__(*args, **kwargs)
        # self.inner_opt = tf.keras.optimizers.SGD(self.lr_inner_ml, momentum=0.9)
        # self.inner_opt = tf.keras.optimizers.Adam(self.lr_inner_ml)

    @tf.function
    def _train_model_for_eval(self, train_ds, train_labels):
        with tf.GradientTape(persistent=False) as train_tape:
            train_tape.watch(self.eval_model.meta_trainable_variables)
            logits = self.eval_model(train_ds, training=True)
            loss = self.inner_loss(train_labels, logits)
            tf.print(loss)
        # gradients = self.inner_opt.get_gradients(loss, self.eval_model.meta_trainable_variables)
        # self.inner_opt.apply_gradients(zip(gradients, self.eval_model.meta_trainable_variables))
        gradients = train_tape.gradient(loss, self.eval_model.meta_trainable_variables)
        self.create_meta_model(self.eval_model, self.eval_model, gradients, assign=True)

    def initialize_network(self):
        model = self.network_cls(self.n, self.random_layer_initialization_seed)
        model(tf.zeros(shape=(1, 224, 224, 3)))
        return model

    def get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, (224, 224))
            image = tf.keras.applications.vgg16.preprocess_input(image)
            return image

        return parse_function


def run_transfer_learning():
    plantdisease = PlantDiseaseDatabase()
    transfer_learning = TransferLearning(
        database=plantdisease,
        network_cls=get_network,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=5,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.01,
        num_steps_validation=5,
        save_after_iterations=15000,
        meta_learning_rate=0.001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=100,
        clip_gradients=True,
        experiment_name='fixed_vgg_16_plant_disease',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
        random_layer_initialization_seed=42,
    )
    transfer_learning.evaluate(10, seed=42, use_val_batch_statistics=True)


if __name__ == '__main__':
    run_transfer_learning()
