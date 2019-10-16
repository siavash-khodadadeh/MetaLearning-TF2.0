import os

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tf_datasets import OmniglotDatabase, MiniImagenetDatabase
from networks import SimpleModel, MiniImagenetModel
from models.base_model import  BaseModel
from utils import combine_first_two_axes, average_gradients
import settings


# TODO
# Train and get reasonable results to make sure that the implementation does not have any bugs.

# Visualize all validation tasks just once.


# Fix tests and add tests for UMTRA.

# Test visualization could be done on all images or some of them
# Train visualization could be done after some iterations.

# Check if the tf.function can help with improving speed.
# Make it possible to train on multiple GPUs (Not very necassary now), but we have to make it fast with tf.function.
# Add evaluation without implementing the inner loop to test the correctness.


class ModelAgnosticMetaLearningModel(BaseModel):
    def __init__(
        self,
        database,
        network_cls,
        n,
        k,
        meta_batch_size,
        num_steps_ml,
        lr_inner_ml,
        num_steps_validation,
        save_after_epochs,
        meta_learning_rate,
        log_train_images_after_iteration  # Set to -1 if you do not want to log train images.
    ):
        self.n = n
        self.k = k
        self.meta_batch_size = meta_batch_size
        self.num_steps_ml = num_steps_ml
        self.lr_inner_ml = lr_inner_ml
        self.num_steps_validation = num_steps_validation
        self.save_after_epochs = save_after_epochs
        self.log_train_images_after_iteration = log_train_images_after_iteration
        super(ModelAgnosticMetaLearningModel, self).__init__(database, network_cls)

        self.model = self.network_cls(num_classes=self.n)
        self.model(tf.zeros(shape=(n * k, *self.database.input_shape)))

        self.updated_model = self.network_cls(num_classes=self.n)
        self.updated_model(tf.zeros(shape=(n * k, *self.database.input_shape)))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=meta_learning_rate)
        self.val_accuracy_metric = tf.metrics.Accuracy()
        self.val_loss_metric = tf.metrics.Mean()

        self._root = self.get_root()
        self.train_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/train/')
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/val/')
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)

        self.checkpoint_dir = os.path.join(self._root, self.get_config_info(), 'saved_models/')

        self.train_accuracy_metric = tf.metrics.Accuracy()
        self.train_loss_metric = tf.metrics.Mean()

    def get_root(self):
        return os.path.dirname(__file__)

    def get_train_dataset(self):
        return self.database.get_supervised_meta_learning_dataset(
            self.database.train_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=self.meta_batch_size
        )

    def get_val_dataset(self):
        return self.database.get_supervised_meta_learning_dataset(
            self.database.val_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=1,
            reshuffle_each_iteration=False
        )

    def get_test_dataset(self):
        return self.database.get_supervised_meta_learning_dataset(
            self.database.test_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=1,
        )

    def get_config_info(self):
        return f'model-{self.network_cls.name}_' \
               f'mbs-{self.meta_batch_size}_' \
               f'n-{self.n}_' \
               f'k-{self.k}_' \
               f'stp-{self.num_steps_ml}'

    def create_meta_model(self, model, gradients):
        k = 0
        variables = list()

        for i in range(len(model.layers)):
            if isinstance(model.layers[i], tf.keras.layers.Conv2D) or \
             isinstance(model.layers[i], tf.keras.layers.Dense):
                self.updated_model.layers[i].kernel = model.layers[i].kernel - self.lr_inner_ml * gradients[k]
                k += 1
                variables.append(self.updated_model.layers[i].kernel)

                self.updated_model.layers[i].bias = model.layers[i].bias - self.lr_inner_ml * gradients[k]
                k += 1
                variables.append(self.updated_model.layers[i].bias)

            elif isinstance(model.layers[i], tf.keras.layers.BatchNormalization):
                if hasattr(model.layers[i], 'moving_mean') and model.layers[i].moving_mean is not None:
                    self.updated_model.layers[i].moving_mean.assign(model.layers[i].moving_mean)
                if hasattr(model.layers[i], 'moving_variance') and model.layers[i].moving_variance is not None:
                    self.updated_model.layers[i].moving_variance.assign(model.layers[i].moving_variance)
                if hasattr(model.layers[i], 'gamma') and model.layers[i].gamma is not None:
                    self.updated_model.layers[i].gamma = model.layers[i].gamma - self.lr_inner_ml * gradients[k]
                    k += 1
                    variables.append(self.updated_model.layers[i].gamma)
                if hasattr(model.layers[i], 'beta') and model.layers[i].beta is not None:
                    self.updated_model.layers[i].beta = \
                        model.layers[i].beta - self.lr_inner_ml * gradients[k]
                    k += 1
                    variables.append(self.updated_model.layers[i].beta)

        setattr(self.updated_model, 'meta_trainable_variables', variables)
        return self.updated_model

    def get_train_loss_and_gradients(self, train_ds, train_labels):
        with tf.GradientTape(persistent=True) as train_tape:
            # TODO compare between model.forward(train_ds) and model(train_ds)
            logits = self.model(train_ds, training=True)
            train_loss = tf.reduce_sum(tf.losses.categorical_crossentropy(train_labels, logits, from_logits=True))

        train_gradients = train_tape.gradient(train_loss, self.model.trainable_variables)
        return train_loss, train_gradients, train_tape

    def get_task_train_and_val_ds(self, task, labels):
        train_ds, val_ds = tf.split(task, num_or_size_splits=2)
        train_labels, val_labels = tf.split(labels, num_or_size_splits=2)
        
        train_ds = combine_first_two_axes(tf.squeeze(train_ds, axis=0))
        val_ds = combine_first_two_axes(tf.squeeze(val_ds, axis=0))
        train_labels = combine_first_two_axes(tf.squeeze(train_labels, axis=0))
        val_labels = combine_first_two_axes(tf.squeeze(val_labels, axis=0))

        return train_ds, val_ds, train_labels, val_labels

    def inner_train_loop(self, train_ds, train_labels, num_iterations):
        gradients = list()
        for variable in self.model.trainable_variables:
            gradients.append(tf.zeros_like(variable))

        updated_model = self.create_meta_model(self.model, gradients)
        for k in range(num_iterations):
            with tf.GradientTape(persistent=True) as train_tape:
                train_tape.watch(updated_model.meta_trainable_variables)
                logits = updated_model(train_ds, training=True)
                loss = tf.reduce_sum(
                    tf.losses.categorical_crossentropy(train_labels, logits, from_logits=True)
                )

            gradients = train_tape.gradient(loss, updated_model.meta_trainable_variables)
            updated_model = self.create_meta_model(updated_model, gradients)

        return updated_model

    def save_model(self, epochs):
        self.model.save_weights(os.path.join(self.checkpoint_dir, f'model.ckpt-{epochs}'))

    def load_model(self, epochs=None):
        if epochs is not None:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model.ckpt-{epochs}')
        else:
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)

        if checkpoint_path is not None:
            print('==================\nResuming Training\n==================')
            self.model.load_weights(checkpoint_path)
        else:
            print('No previous checkpoint found!')

    def evaluate(self, iterations):
        self.test_dataset = self.get_test_dataset()
        self.load_model()
        test_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/test/')
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        test_accuracy_metric = tf.metrics.Accuracy()
        test_loss_metric = tf.metrics.Mean()

        for tmb, lmb in self.test_dataset:
            for task, labels in zip(tmb, lmb):
                train_ds, val_ds, train_labels, val_labels = self.get_task_train_and_val_ds(task, labels)
                updated_model = self.inner_train_loop(train_ds, train_labels, iterations)
                updated_model_logits = updated_model(val_ds, training=False)

                self.update_loss_and_accuracy(updated_model_logits, val_labels, test_loss_metric, test_accuracy_metric)

            self.log_metric(test_summary_writer, 'Loss', test_loss_metric, step=1)
            self.log_metric(test_summary_writer, 'Accuracy', test_accuracy_metric, step=1)

            print('Test Loss: {}'.format(test_loss_metric.result().numpy()))
            print('Test Accuracy: {}'.format(test_accuracy_metric.result().numpy()))

    def log_images(self, summary_writer, train_ds, val_ds, step):
        with tf.device('cpu:0'):
            with summary_writer.as_default():
                tf.summary.image(
                    'train',
                    train_ds,
                    step=step,
                    max_outputs=5
                )
                tf.summary.image(
                    'validation',
                    val_ds,
                    step=step,
                    max_outputs=5
                )

    def update_loss_and_accuracy(self, logits, labels, loss_metric, accuracy_metric):
        print(tf.argmax(logits, axis=-1))
        val_loss = tf.reduce_sum(
            tf.losses.categorical_crossentropy(labels, logits, from_logits=True))
        loss_metric.update_state(val_loss)
        accuracy_metric.update_state(
            tf.argmax(labels, axis=-1),
            tf.argmax(logits, axis=-1)
        )

    def log_metric(self, summary_writer, name, metric, step):
        with summary_writer.as_default():
            tf.summary.scalar(name, metric.result(), step=step)

    def report_validation_loss_and_accuracy(self, epoch_count):
        self.val_loss_metric.reset_states()
        self.val_accuracy_metric.reset_states()

        val_counter = 0
        for tmb, lmb in self.val_dataset:
            val_counter += 1
            for task, labels in zip(tmb, lmb):
                train_ds, val_ds, train_labels, val_labels = self.get_task_train_and_val_ds(task, labels)
                if val_counter % 5 == 0:
                    step = epoch_count * self.val_dataset.steps_per_epoch + val_counter
                    self.log_images(self.val_summary_writer, train_ds, val_ds, step)

                updated_model = self.inner_train_loop(train_ds, train_labels, self.num_steps_validation)
                updated_model_logits = updated_model(val_ds, training=False)

                self.update_loss_and_accuracy(
                    updated_model_logits, val_labels, self.val_loss_metric, self.val_accuracy_metric
                )

        self.log_metric(self.val_summary_writer, 'Loss', self.val_loss_metric, step=epoch_count)
        self.log_metric(self.val_summary_writer, 'Accuracy', self.val_accuracy_metric, step=epoch_count)

        print('Validation Loss: {}'.format(self.val_loss_metric.result().numpy()))
        print('Validation Accuracy: {}'.format(self.val_accuracy_metric.result().numpy()))

    @tf.function
    def get_gradients_of_tasks_batch(self, inputs):
        task, labels, iteration_count = inputs

        train_ds, val_ds, train_labels, val_labels = self.get_task_train_and_val_ds(task, labels)

        if self.log_train_images_after_iteration != -1 and \
                iteration_count % self.log_train_images_after_iteration == 0:

            self.log_images(self.train_summary_writer, train_ds, val_ds, step=iteration_count)

            with tf.device('cpu:0'):
                with self.train_summary_writer.as_default():
                    for var in self.model.variables:
                        tf.summary.histogram(var.name, var, step=iteration_count)
                    for var in self.updated_model.variables:
                        tf.summary.histogram('updated_model' + var.name, var, step=iteration_count)

        with tf.GradientTape(persistent=True) as val_tape:
            updated_model = self.inner_train_loop(train_ds, train_labels, self.num_steps_ml)
            updated_model_logits = updated_model(val_ds, training=False)
            val_loss = tf.reduce_sum(
                tf.losses.categorical_crossentropy(val_labels, updated_model_logits, from_logits=True)
            )
            self.train_loss_metric.update_state(val_loss)
            self.train_accuracy_metric.update_state(
                tf.argmax(val_labels, axis=-1),
                tf.argmax(updated_model_logits, axis=-1)
            )
        val_gradients = val_tape.gradient(val_loss, self.model.trainable_variables)
        return val_gradients

    def train(self, epochs=5):
        self.train_dataset = self.get_train_dataset()
        self.val_dataset = self.get_val_dataset()
        self.load_model()
        iteration_count = 0

        pbar = tqdm(self.train_dataset)

        for epoch_count in range(epochs):
            self.report_validation_loss_and_accuracy(epoch_count)
            if epoch_count != 0:
                print('Train Loss: {}'.format(self.train_loss_metric.result().numpy()))
                print('Train Accuracy: {}'.format(self.train_accuracy_metric.result().numpy()))

                with self.train_summary_writer.as_default():
                    tf.summary.scalar('Loss', self.train_loss_metric.result(), step=epoch_count)
                    tf.summary.scalar('Accuracy', self.train_accuracy_metric.result(), step=epoch_count)

            self.train_accuracy_metric.reset_states()
            self.train_loss_metric.reset_states()

            if epoch_count != 0 and epoch_count % self.save_after_epochs == 0:
                self.save_model(epoch_count)

            for tasks_meta_batch, labels_meta_batch in self.train_dataset:
                tasks_final_gradients = tf.map_fn(
                    self.get_gradients_of_tasks_batch,
                    elems=(
                        tasks_meta_batch,
                        labels_meta_batch,
                        tf.cast(tf.ones(tasks_meta_batch.shape[0], 1) * iteration_count, tf.int64)
                    ),
                    dtype=[tf.float32] * 14,
                    parallel_iterations=tasks_meta_batch.shape[0]
                )

                final_gradients = average_gradients(tasks_final_gradients)
                self.optimizer.apply_gradients(zip(final_gradients, self.model.trainable_variables))
                iteration_count += 1
                pbar.set_description_str('Iteration{}: Train Loss: {}, Train Accuracy: {}'.format(
                    iteration_count,
                    self.train_loss_metric.result().numpy(),
                    self.train_accuracy_metric.result().numpy()
                ))
                pbar.update(1)


def run_omniglot():
    omniglot_database = OmniglotDatabase(
        random_seed=47,
        num_train_classes=1200,
        num_val_classes=100,
    )

    maml = ModelAgnosticMetaLearningModel(
        database=omniglot_database,
        network_cls=SimpleModel,
        n=5,
        k=1,
        meta_batch_size=4,
        num_steps_ml=1,
        lr_inner_ml=0.4,
        num_steps_validation=1,
        save_after_epochs=50,
        meta_learning_rate=0.001,
        log_train_images_after_iteration=-1
    )

    maml.train(epochs=1000)
    # maml.evaluate(iterations=50)


def run_mini_imagenet():
    mini_imagenet_database = MiniImagenetDatabase(random_seed=-1)

    maml = ModelAgnosticMetaLearningModel(
        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        meta_batch_size=8,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_epochs=20,
        meta_learning_rate=0.0001,
        log_train_images_after_iteration=-1
    )

    maml.train(epochs=100)
    # maml.evaluate(50)


if __name__ == '__main__':
    run_omniglot()
