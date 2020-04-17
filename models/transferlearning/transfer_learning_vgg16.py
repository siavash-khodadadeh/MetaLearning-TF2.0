import tensorflow as tf

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import get_transfer_net


class TransferLearningVGG16(ModelAgnosticMetaLearningModel):
    def __init__(
            self,
            database,
            n,
            k_val_test,
            k_test,
            lr_inner_ml,
            number_of_tasks_test,
            val_test_batch_norm_momentum,
            random_layer_initialization_seed,
            num_trainable_layers,
    ):
        self.random_layer_initialization_seed = random_layer_initialization_seed
        self.num_trainable_layers = num_trainable_layers

        super(TransferLearningVGG16, self).__init__(
            database=database,
            network_cls=None,
            n=n,
            k=-1,
            k_val_ml=-1,
            k_val_val=-1,
            k_val_test=k_val_test,
            k_test=k_test,
            meta_batch_size=-1,
            num_steps_ml=-1,
            lr_inner_ml=lr_inner_ml,
            num_steps_validation=-1,
            save_after_iterations=-1,
            meta_learning_rate=-1,
            report_validation_frequency=-1,
            log_train_images_after_iteration=-1,  # Set to -1 if you do not want to log train images.
            number_of_tasks_val=-1,  # Make sure the validation pick this many tasks.
            number_of_tasks_test=number_of_tasks_test,  # Make sure the validation pick this many tasks.
            val_seed=-1,  # The seed for validation dataset. -1 means change the samples for each report.
            clip_gradients=False,
            experiment_name=None,
            val_test_batch_norm_momentum=val_test_batch_norm_momentum,
        )
        # self.inner_opt = tf.keras.optimizers.SGD(self.lr_inner_ml, momentum=0.9)
        # self.inner_opt = tf.keras.optimizers.Adam(self.lr_inner_ml)

    def get_network_name(self):
        return 'VGG16'

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
        model = get_transfer_net(
            architecture='VGG16',
            num_trainable_layers=self.num_trainable_layers,
            num_classes=self.n,
            random_layer_initialization_seed=self.random_layer_initialization_seed
        )
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
