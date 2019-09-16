import os

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tf_datasets import OmniglotDatabase
from networks import SimpleModel
from utils import combine_first_two_axes, average_gradients
import settings


# TODO
# Add tensorboard.
# Save the model and logs.
# Make batch norm settings in config if necassary.
# Add evaluation.
# Check if the tf.function can help with improving speed.
# Make it possible to train on multiple GPUs (Not very necassary now), but we have to make it fast with tf.function.


class ModelAgnosticMetaLearningModel(object):
    def __init__(self, config):
        self.config = config
        self.database_config = self.config['database']
        self.database = self.database_config['database_class'](
            self.database_config['random_seed'], 
            self.database_config['config']
        )

        self.model = self.config['model']['class'](**self.config['model']['init_kwargs'])
        self.model.predict(
            tf.zeros(
                shape=(
                    self.database_config['config']['train_dataset_kwargs']['n'] *
                    self.database_config['config']['train_dataset_kwargs']['k'], 28, 28, 1
                )
            )
        )
        self.updated_model = self.config['model']['class'](**self.config['model']['init_kwargs'])
        self.updated_model(
            tf.zeros(
                shape=(
                    self.database_config['config']['train_dataset_kwargs']['n'] *
                    self.database_config['config']['train_dataset_kwargs']['k'], 28, 28, 1
                )
            )
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.val_accuracy_metric = tf.metrics.Accuracy()
        self.val_loss_metric = tf.metrics.Mean()

        self._root = os.path.dirname(__file__)
        self.train_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/train/')
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/val/')
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)
        self.test_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/test/')
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)

        self.checkpoint_dir = os.path.join(self._root, self.get_config_info(), 'saved_models/')

        self.train_accuracy_metric = tf.metrics.Accuracy()
        self.train_loss_metric = tf.metrics.Mean()

    def get_config_info(self):
        return f'model-{self.config["model"]["class"].name}_' \
               f'mbs-{self.config["database"]["config"]["train_dataset_kwargs"]["meta_batch_size"]}_' \
               f'n-{self.config["database"]["config"]["train_dataset_kwargs"]["n"]}_' \
               f'k-{self.config["database"]["config"]["train_dataset_kwargs"]["k"]}_' \
               f'stp-{self.config["num_steps_ml"]}'

    def create_meta_model(self, model, gradients):
        k = 0
        variables = list()

        for i in range(len(model.layers)):
            if isinstance(model.layers[i], tf.keras.layers.Conv2D) or \
             isinstance(model.layers[i], tf.keras.layers.Dense):
                self.updated_model.layers[i].kernel = model.layers[i].kernel - self.config['lr_inner_ml'] * gradients[k]
                k += 1
                variables.append(self.updated_model.layers[i].kernel)

                self.updated_model.layers[i].bias = model.layers[i].bias - self.config['lr_inner_ml'] * gradients[k]
                k += 1
                variables.append(self.updated_model.layers[i].bias)

            elif isinstance(model.layers[i], tf.keras.layers.BatchNormalization):
                if hasattr(model.layers[i], 'moving_mean') and model.layers[i].moving_mean is not None:
                    self.updated_model.layers[i].moving_mean.assign(model.layers[i].moving_mean)
                if hasattr(model.layers[i], 'moving_variance') and model.layers[i].moving_variance is not None:
                    self.updated_model.layers[i].moving_variance.assign(model.layers[i].moving_variance)
                if hasattr(model.layers[i], 'gamma') and model.layers[i].gamma is not None:
                    self.updated_model.layers[i].gamma = model.layers[i].gamma - self.config['lr_inner_ml'] * gradients[k]
                    k += 1
                    variables.append(self.updated_model.layers[i].gamma)
                if hasattr(model.layers[i], 'beta') and model.layers[i].beta is not None:
                    self.updated_model.layers[i].beta = \
                        model.layers[i].beta - self.config['lr_inner_ml'] * gradients[k]
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

    def report_validation_loss_and_accuracy(self, epoch_count):
        self.val_loss_metric.reset_states()
        self.val_accuracy_metric.reset_states()

        val_counter = 0
        for tmb, lmb in self.database.val_ds:
            if val_counter == self.database.val_ds.steps_per_epoch:
                break
            val_counter += 1
            for task, labels in zip(tmb, lmb):
                train_ds, val_ds, train_labels, val_labels = self.get_task_train_and_val_ds(task, labels)

                updated_model = self.inner_train_loop(train_ds, train_labels, 1)
                updated_model_logits = updated_model(val_ds, training=False)

                print(tf.argmax(updated_model_logits, axis=-1))
                val_loss = tf.reduce_sum(
                    tf.losses.categorical_crossentropy(val_labels, updated_model_logits, from_logits=True))
                self.val_loss_metric.update_state(val_loss)
                self.val_accuracy_metric.update_state(
                    tf.argmax(val_labels, axis=-1),
                    tf.argmax(updated_model_logits, axis=-1)
                )

        with self.val_summary_writer.as_default():
            tf.summary.scalar('Loss', self.val_loss_metric.result(), step=epoch_count)
            tf.summary.scalar('Accuracy', self.val_accuracy_metric.result(), step=epoch_count)

        print('Validation Loss: {}'.format(self.val_loss_metric.result().numpy()))
        print('Validation Accuracy: {}'.format(self.val_accuracy_metric.result().numpy()))

    def train(self, epochs=5):
        self.load_model()
        epoch_count = -1
        counter = 0

        pbar = tqdm(self.database.train_ds)
        for tasks_meta_batch, labels_meta_batch in self.database.train_ds:
            if counter % self.database.train_ds.steps_per_epoch == 0:
                # one epoch ends here
                epoch_count += 1
                self.report_validation_loss_and_accuracy(epoch_count)
                if epoch_count != 0:
                    print('Train Loss: {}'.format(self.train_loss_metric.result().numpy()))
                    print('Train Accuracy: {}'.format(self.train_accuracy_metric.result().numpy()))

                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('Loss', self.train_loss_metric.result(), step=epoch_count)
                        tf.summary.scalar('Accuracy', self.train_accuracy_metric.result(), step=epoch_count)

                self.train_accuracy_metric.reset_states()
                self.train_loss_metric.reset_states()

                if epoch_count != 0 and epoch_count % self.config['save_after_epochs'] == 0:
                    self.save_model(epoch_count)

            if counter == self.database.train_ds.steps_per_epoch * epochs:
                break

            counter += 1
            tasks_final_gradients = list()
            
            for task, labels in zip(tasks_meta_batch, labels_meta_batch):
                train_ds, val_ds, train_labels, val_labels = self.get_task_train_and_val_ds(task, labels)

                with tf.GradientTape(persistent=True) as val_tape:
                    updated_model = self.inner_train_loop(train_ds, train_labels, 1)
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
                tasks_final_gradients.append(val_gradients)

            final_gradients = average_gradients(tasks_final_gradients)
            self.optimizer.apply_gradients(zip(final_gradients, self.model.trainable_variables))
            pbar.set_description_str('Iteration{}: Train Loss: {}, Train Accuracy: {}'.format(
                counter,
                self.train_loss_metric.result().numpy(),
                self.train_accuracy_metric.result().numpy()
            ))
            pbar.update(1)


if __name__ == '__main__':
    config = {
        'database': {
            'database_class': OmniglotDatabase,
            'random_seed': -1,
            'config': {
                'num_train_classes': 1200,
                'num_val_classes': 100,
                'train_dataset_kwargs': {
                    'n': 5, 'k': 5, 'meta_batch_size': 32
                },
                'val_dataset_kwargs': {
                    'n': 5, 'k': 5, 'meta_batch_size': 1
                },
                'test_dataset_kwargs': {
                    'n': 5, 'k': 5, 'meta_batch_size': 1
                },
            },
        },
        'model': {
            'class': SimpleModel,
            'init_kwargs': {'num_classes': 5},
        },
        'lr_inner_ml': 0.05,
        'num_steps_ml': 5,
        'save_after_epochs': 3,
    }

    maml = ModelAgnosticMetaLearningModel(config)
    maml.train(epochs=100)
