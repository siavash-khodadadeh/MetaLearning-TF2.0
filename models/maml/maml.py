import tensorflow as tf

from models.base_model import BaseModel
from utils import combine_first_two_axes
import settings


# TODO
# Visualize all validation tasks just once.

# Fix tests and add tests for UMTRA.

# Test visualization could be done on all images or some of them

# Make it possible to train on multiple GPUs (Not very necessary now), but we have to make it fast with tf.function.


class ModelAgnosticMetaLearningModel(BaseModel):
    def __init__(
        self,
        database,
        network_cls,
        n,
        k,
        k_val_ml,
        k_val_val,
        k_val_test,
        k_test,
        meta_batch_size,
        num_steps_ml,
        lr_inner_ml,
        num_steps_validation,
        save_after_iterations,
        meta_learning_rate,
        report_validation_frequency,
        log_train_images_after_iteration,  # Set to -1 if you do not want to log train images.
        number_of_tasks_val=-1,  # Make sure the validation pick this many tasks.
        number_of_tasks_test=-1,  # Make sure the validation pick this many tasks.
        val_seed=-1,  # The seed for validation dataset. -1 means change the samples for each report.
        clip_gradients=False,
        experiment_name=None,
        val_test_batch_norm_momentum=0.0,
        val_database=None,
        target_database=None,
        k_val_train=None,
    ):
        super(ModelAgnosticMetaLearningModel, self).__init__(
            database=database,
            network_cls=network_cls,
            n=n,
            k=k,
            k_val_ml=k_val_ml,
            k_val_val=k_val_val,
            k_val_train=k_val_train,
            k_val_test=k_val_test,
            k_test=k_test,
            meta_batch_size=meta_batch_size,
            meta_learning_rate=meta_learning_rate,
            save_after_iterations=save_after_iterations,
            report_validation_frequency=report_validation_frequency,
            log_train_images_after_iteration=log_train_images_after_iteration,
            number_of_tasks_val=number_of_tasks_val,
            number_of_tasks_test=number_of_tasks_test,
            val_seed=val_seed,
            experiment_name=experiment_name,
            val_database=val_database,
            target_database=target_database
        )

        self.num_steps_ml = num_steps_ml
        self.num_steps_validation = num_steps_validation
        self.lr_inner_ml = lr_inner_ml
        self.clip_gradients = clip_gradients
        self.updated_models = list()

        for _ in range(self.num_steps_ml + 1):
            updated_model = self.initialize_network()
            self.updated_models.append(updated_model)

        self.eval_model = self.initialize_network()
        self.val_test_batch_norm_momentum = val_test_batch_norm_momentum
        for layer in self.eval_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.momentum = self.val_test_batch_norm_momentum

    def initialize_network(self):
        model = self.network_cls(num_classes=self.n)
        model(tf.zeros(shape=(1, *self.database.input_shape)))
        return model

    def get_network_name(self):
        if self.network_cls is not None:
            return self.network_cls.name
        raise Exception('Model name is not defined. Please override get_network_name function.')

    def get_config_str(self):
        config_str = f'model-{self.get_network_name()}_' \
               f'mbs-{self.meta_batch_size}_' \
               f'n-{self.n}_' \
               f'k-{self.k}_' \
               f'kvalml-{self.k_val_ml}' \
               f'stp-{self.num_steps_ml}'
        return config_str

    def post_process_outer_gradients(self, outer_gradients):
        if self.clip_gradients:
            outer_gradients = [tf.clip_by_value(grad, -10, 10) for grad in outer_gradients]
        return outer_gradients

    def create_meta_model(self, updated_model, model, gradients, assign=False):
        k = 0
        variables = list()
        # TODO Maybe get layers by name

        for i in range(len(model.layers)):
            if model.layers[i].trainable:
                if (isinstance(model.layers[i], tf.keras.layers.Conv2D) or
                        isinstance(model.layers[i], tf.keras.layers.Dense)):
                    if assign:
                        updated_model.layers[i].kernel.assign(model.layers[i].kernel - self.lr_inner_ml * gradients[k])
                    else:
                        updated_model.layers[i].kernel = model.layers[i].kernel - self.lr_inner_ml * gradients[k]
                    k += 1
                    variables.append(updated_model.layers[i].kernel)

                    if assign:
                        updated_model.layers[i].bias.assign(model.layers[i].bias - self.lr_inner_ml * gradients[k])
                    else:
                        updated_model.layers[i].bias = model.layers[i].bias - self.lr_inner_ml * gradients[k]
                    k += 1
                    variables.append(updated_model.layers[i].bias)

                elif isinstance(model.layers[i], tf.keras.layers.BatchNormalization):
                    if hasattr(model.layers[i], 'moving_mean') and model.layers[i].moving_mean is not None:
                        if assign:
                            updated_model.layers[i].moving_mean.assign(model.layers[i].moving_mean)
                        else:
                            updated_model.layers[i].moving_mean = model.layers[i].moving_mean
                    if hasattr(model.layers[i], 'moving_variance') and model.layers[i].moving_variance is not None:
                        if assign:
                            updated_model.layers[i].moving_variance.assign(model.layers[i].moving_variance)
                        else:
                            updated_model.layers[i].moving_variance = model.layers[i].moving_variance
                    if hasattr(model.layers[i], 'gamma') and model.layers[i].gamma is not None:
                        if assign:
                            updated_model.layers[i].gamma.assign(
                                model.layers[i].gamma - self.lr_inner_ml * gradients[k]
                            )
                        else:
                            updated_model.layers[i].gamma = model.layers[i].gamma - self.lr_inner_ml * gradients[k]
                        k += 1
                        variables.append(updated_model.layers[i].gamma)
                    if hasattr(model.layers[i], 'beta') and model.layers[i].beta is not None:
                        if assign:
                            updated_model.layers[i].beta.assign(model.layers[i].beta - self.lr_inner_ml * gradients[k])
                        else:
                            updated_model.layers[i].beta = model.layers[i].beta - self.lr_inner_ml * gradients[k]
                        k += 1
                        variables.append(updated_model.layers[i].beta)

        setattr(updated_model, 'meta_trainable_variables', variables)

    def inner_loss(self, train_labels, logits):
        loss = tf.reduce_mean(
            tf.losses.categorical_crossentropy(train_labels, logits, from_logits=True)
        )
        return loss

    def inner_train_loop(self, train_ds, train_labels):
        num_iterations = self.num_steps_ml

        gradients = list()
        for variable in self.model.trainable_variables:
            gradients.append(tf.zeros_like(variable))

        self.create_meta_model(self.updated_models[0], self.model, gradients)

        for k in range(1, num_iterations + 1):
            with tf.GradientTape(persistent=False) as train_tape:
                train_tape.watch(self.updated_models[k - 1].meta_trainable_variables)
                logits = self.updated_models[k - 1](train_ds, training=True)
                loss = self.inner_loss(train_labels, logits)
            gradients = train_tape.gradient(loss, self.updated_models[k - 1].meta_trainable_variables)
            self.create_meta_model(self.updated_models[k], self.updated_models[k - 1], gradients)

        return self.updated_models[-1]

    def update_loss_and_accuracy(self, logits, labels, loss_metric, accuracy_metric):
        val_loss = self.outer_loss(labels, logits)
        loss_metric.update_state(val_loss)

        predicted_class_labels = self.predict_class_labels_from_logits(logits)
        real_labels = self.convert_labels_to_real_labels(labels)

        # print(predicted_class_labels)

        accuracy_metric.update_state(
            real_labels,
            predicted_class_labels
        )
        val_acc = tf.math.reduce_mean(tf.cast(tf.math.equal(real_labels, predicted_class_labels), tf.float32))

        return val_acc, val_loss

    def outer_loss(self, labels, logits):
        loss = tf.reduce_mean(
            tf.losses.categorical_crossentropy(labels, logits, from_logits=True)
        )
        return loss

    def get_losses_of_tasks_batch(self, method='train', **kwargs):
        if method == 'train':
            return self.get_losses_of_tasks_batch_maml()
        elif method == 'val':
            return self.get_losses_of_tasks_batch_eval(iterations=self.num_steps_validation, training=False)
        elif method == 'test':
            return self.get_losses_of_tasks_batch_eval(
                iterations=kwargs['iterations'],
                training=kwargs['use_val_batch_statistics']
            )

    @tf.function
    def _initialize_eval_model(self):
        gradients = list()
        for variable in self.model.trainable_variables:
            gradients.append(tf.zeros_like(variable))
        self.create_meta_model(self.eval_model, self.model, gradients, assign=True)

    @tf.function
    def _train_model_for_eval(self, train_ds, train_labels):
        with tf.GradientTape(persistent=False) as train_tape:
            train_tape.watch(self.eval_model.meta_trainable_variables)
            logits = self.eval_model(train_ds, training=True)
            loss = self.inner_loss(train_labels, logits)
            if settings.DEBUG:
                tf.print(loss)
        gradients = train_tape.gradient(loss, self.eval_model.meta_trainable_variables)
        self.create_meta_model(self.eval_model, self.eval_model, gradients, assign=True)

    @tf.function
    def _evaluate_model_for_eval(self, val_ds, val_labels, training):
        updated_model_logits = self.eval_model(val_ds, training=training)
        val_loss = self.outer_loss(val_labels, updated_model_logits)

        predicted_class_labels = self.predict_class_labels_from_logits(updated_model_logits)
        real_labels = self.convert_labels_to_real_labels(val_labels)

        val_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_class_labels, real_labels), tf.float32))
        return val_acc, val_loss

    def get_losses_of_tasks_batch_eval(self, iterations, training):
        def f(inputs):
            train_ds, val_ds, train_labels, val_labels = inputs
            train_ds = combine_first_two_axes(train_ds)
            val_ds = combine_first_two_axes(val_ds)

            self._initialize_eval_model()
            for i in range(iterations):
                self._train_model_for_eval(train_ds, train_labels)
            val_acc, val_loss = self._evaluate_model_for_eval(val_ds, val_labels, training)
            if settings.DEBUG:
                tf.print()
                tf.print(val_loss)
                tf.print(val_acc)
                tf.print()
            return val_acc, val_loss
        return f

    def get_losses_of_tasks_batch_maml(self):
        # TODO check if tf.function results in any imporvement in speed since this causes a lot of warning.
        @tf.function
        def f(inputs):
            train_ds, val_ds, train_labels, val_labels = inputs
            train_ds = combine_first_two_axes(train_ds)
            val_ds = combine_first_two_axes(val_ds)

            updated_model = self.inner_train_loop(train_ds, train_labels)
            # TODO test what happens when training=False
            updated_model_logits = updated_model(val_ds, training=True)
            val_loss = self.outer_loss(val_labels, updated_model_logits)

            predicted_class_labels = self.predict_class_labels_from_logits(updated_model_logits)
            real_labels = self.convert_labels_to_real_labels(val_labels)

            val_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_class_labels, real_labels), tf.float32))

            return val_acc, val_loss

        return f

    def convert_labels_to_real_labels(self, labels):
        return tf.argmax(labels, axis=-1)

    def predict_class_labels_from_logits(self, logits):
        return tf.argmax(logits, axis=-1)
