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
        report_validation_frequency,
        log_train_images_after_iteration,  # Set to -1 if you do not want to log train images.
        least_number_of_tasks_val_test=-1,  # Make sure the validaiton and test dataset pick at least this many tasks.
        clip_gradients=False
    ):
        self.n = n
        self.k = k
        self.meta_batch_size = meta_batch_size
        self.num_steps_ml = num_steps_ml
        self.lr_inner_ml = lr_inner_ml
        self.num_steps_validation = num_steps_validation
        self.save_after_epochs = save_after_epochs
        self.log_train_images_after_iteration = log_train_images_after_iteration
        self.report_validation_frequency = report_validation_frequency
        self.least_number_of_tasks_val_test = least_number_of_tasks_val_test
        self.clip_gradients = clip_gradients
        super(ModelAgnosticMetaLearningModel, self).__init__(database, network_cls)

        self.model = self.network_cls(num_classes=self.n)
        self.model(tf.zeros(shape=(n * k, *self.database.input_shape)))

        self.updated_models = list()
        for _ in range(self.num_steps_ml + 1):
            updated_model = self.network_cls(num_classes=self.n)
            updated_model(tf.zeros(shape=(n * k, *self.database.input_shape)))
            self.updated_models.append(updated_model)

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
        dataset = self.database.get_supervised_meta_learning_dataset(
            self.database.train_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=self.meta_batch_size
        )
        # steps_per_epoch = dataset.steps_per_epoch
        # dataset = dataset.prefetch(1)
        # setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset

    def get_val_dataset(self):
        val_dataset = self.database.get_supervised_meta_learning_dataset(
            self.database.val_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=1,
            reshuffle_each_iteration=True
        )
        steps_per_epoch = max(val_dataset.steps_per_epoch, self.least_number_of_tasks_val_test)
        val_dataset = val_dataset.repeat(-1)
        val_dataset = val_dataset.take(steps_per_epoch)
        setattr(val_dataset, 'steps_per_epoch', steps_per_epoch)
        return val_dataset

    def get_test_dataset(self):
        test_dataset = self.database.get_supervised_meta_learning_dataset(
            self.database.test_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=1,
        )
        steps_per_epoch = max(test_dataset.steps_per_epoch, self.least_number_of_tasks_val_test)
        test_dataset = test_dataset.repeat(-1)
        test_dataset = test_dataset.take(steps_per_epoch)
        setattr(test_dataset, 'steps_per_epoch', steps_per_epoch)
        return test_dataset

    def get_config_info(self):
        return f'model-{self.network_cls.name}_' \
               f'mbs-{self.meta_batch_size}_' \
               f'n-{self.n}_' \
               f'k-{self.k}_' \
               f'stp-{self.num_steps_ml}'

    def create_meta_model(self, updated_model, model, gradients):
        k = 0
        variables = list()

        for i in range(len(model.layers)):
            if isinstance(model.layers[i], tf.keras.layers.Conv2D) or \
             isinstance(model.layers[i], tf.keras.layers.Dense):
                updated_model.layers[i].kernel = model.layers[i].kernel - self.lr_inner_ml * gradients[k]
                k += 1
                variables.append(updated_model.layers[i].kernel)

                updated_model.layers[i].bias = model.layers[i].bias - self.lr_inner_ml * gradients[k]
                k += 1
                variables.append(updated_model.layers[i].bias)

            elif isinstance(model.layers[i], tf.keras.layers.BatchNormalization):
                if hasattr(model.layers[i], 'moving_mean') and model.layers[i].moving_mean is not None:
                    updated_model.layers[i].moving_mean.assign(model.layers[i].moving_mean)
                if hasattr(model.layers[i], 'moving_variance') and model.layers[i].moving_variance is not None:
                    updated_model.layers[i].moving_variance.assign(model.layers[i].moving_variance)
                if hasattr(model.layers[i], 'gamma') and model.layers[i].gamma is not None:
                    updated_model.layers[i].gamma = model.layers[i].gamma - self.lr_inner_ml * gradients[k]
                    k += 1
                    variables.append(updated_model.layers[i].gamma)
                if hasattr(model.layers[i], 'beta') and model.layers[i].beta is not None:
                    updated_model.layers[i].beta = \
                        model.layers[i].beta - self.lr_inner_ml * gradients[k]
                    k += 1
                    variables.append(updated_model.layers[i].beta)

        setattr(updated_model, 'meta_trainable_variables', variables)

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

    def inner_train_loop(self, train_ds, train_labels, num_iterations=-1):
        if num_iterations == -1:
            num_iterations = self.num_steps_ml

            gradients = list()
            for variable in self.model.trainable_variables:
                gradients.append(tf.zeros_like(variable))

            self.create_meta_model(self.updated_models[0], self.model, gradients)

            for k in range(1, num_iterations + 1):
                with tf.GradientTape(persistent=True) as train_tape:
                    train_tape.watch(self.updated_models[k - 1].meta_trainable_variables)
                    logits = self.updated_models[k - 1](train_ds, training=True)
                    loss = tf.reduce_sum(
                        tf.losses.categorical_crossentropy(train_labels, logits, from_logits=True)
                    )
                gradients = train_tape.gradient(loss, self.updated_models[k - 1].meta_trainable_variables)
                self.create_meta_model(self.updated_models[k], self.updated_models[k - 1], gradients)

            return self.updated_models[-1]

        else:
            gradients = list()
            for variable in self.model.trainable_variables:
                gradients.append(tf.zeros_like(variable))

            self.create_meta_model(self.updated_models[0], self.model, gradients)
            copy_model = self.updated_models[0]

            for k in range(num_iterations):
                with tf.GradientTape(persistent=True) as train_tape:
                    train_tape.watch(copy_model.meta_trainable_variables)
                    logits = copy_model(train_ds, training=True)
                    loss = tf.reduce_sum(
                        tf.losses.categorical_crossentropy(train_labels, logits, from_logits=True)
                    )
                gradients = train_tape.gradient(loss, copy_model.meta_trainable_variables)
                self.create_meta_model(copy_model, copy_model, gradients)

            return copy_model

    def save_model(self, epochs):
        self.model.save_weights(os.path.join(self.checkpoint_dir, f'model.ckpt-{epochs}'))

    def load_model(self, epochs=None):
        epoch_count = 0
        if epochs is not None:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model.ckpt-{epochs}')
            epoch_count = epochs
        else:
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)

        if checkpoint_path is not None:
            try:
                self.model.load_weights(checkpoint_path)
                print('==================\nResuming Training\n==================')
                epoch_count = int(checkpoint_path[checkpoint_path.rindex('-') + 1:])
            except Exception as e:
                print('Could not load the previous checkpoint!')

        else:
            print('No previous checkpoint found!')

        return epoch_count

    def evaluate(self, iterations, epochs_to_load_from=None):
        self.test_dataset = self.get_test_dataset()
        self.load_model(epochs=epochs_to_load_from)
        test_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/test/')
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        test_accuracy_metric = tf.metrics.Accuracy()
        test_loss_metric = tf.metrics.Mean()

        for tmb, lmb in self.test_dataset:
            for task, labels in zip(tmb, lmb):
                train_ds, val_ds, train_labels, val_labels = self.get_task_train_and_val_ds(task, labels)
                updated_model = self.inner_train_loop(train_ds, train_labels, iterations)
                # If you want to compare with MAML paper, please set the training=True in the following line
                # In that paper the assumption is that we have access to all of test data together and we can evaluate
                # mean and variance from the batch which is given. Make sure to do the same thing in validation.
                updated_model_logits = updated_model(val_ds, training=True)

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
                # If you want to compare with MAML paper, please set the training=True in the following line
                # In that paper the assumption is that we have access to all of test data together and we can evaluate
                # mean and variance from the batch which is given. Make sure to do the same thing in evaluation.
                updated_model_logits = updated_model(val_ds, training=True)

                self.update_loss_and_accuracy(
                    updated_model_logits, val_labels, self.val_loss_metric, self.val_accuracy_metric
                )

        self.log_metric(self.val_summary_writer, 'Loss', self.val_loss_metric, step=epoch_count)
        self.log_metric(self.val_summary_writer, 'Accuracy', self.val_accuracy_metric, step=epoch_count)

        print('Validation Loss: {}'.format(self.val_loss_metric.result().numpy()))
        print('Validation Accuracy: {}'.format(self.val_accuracy_metric.result().numpy()))

    @tf.function
    def get_losses_of_tasks_batch(self, inputs):
        task, labels, iteration_count = inputs

        train_ds, val_ds, train_labels, val_labels = self.get_task_train_and_val_ds(task, labels)

        if self.log_train_images_after_iteration != -1 and \
                iteration_count % self.log_train_images_after_iteration == 0:

            self.log_images(self.train_summary_writer, train_ds, val_ds, step=iteration_count)

            with tf.device('cpu:0'):
                with self.train_summary_writer.as_default():
                    for var in self.model.variables:
                        tf.summary.histogram(var.name, var, step=iteration_count)

                    for k in range(len(self.updated_models)):
                        var_count = 0
                        if hasattr(self.updated_models[k], 'meta_trainable_variables'):
                            for var in self.updated_models[k].meta_trainable_variables:
                                var_count += 1
                                tf.summary.histogram(f'updated_model_{k}_' + str(var_count), var, step=iteration_count)

        updated_model = self.inner_train_loop(train_ds, train_labels, -1)
        updated_model_logits = updated_model(val_ds, training=True)
        val_loss = tf.reduce_sum(
            tf.losses.categorical_crossentropy(val_labels, updated_model_logits, from_logits=True)
        )
        self.train_loss_metric.update_state(val_loss)
        self.train_accuracy_metric.update_state(
            tf.argmax(val_labels, axis=-1),
            tf.argmax(updated_model_logits, axis=-1)
        )
        return val_loss

    def meta_train_loop(self, tasks_meta_batch, labels_meta_batch, iteration_count):
        with tf.GradientTape(persistent=True) as outer_tape:
            tasks_final_losses = tf.map_fn(
                self.get_losses_of_tasks_batch,
                elems=(
                    tasks_meta_batch,
                    labels_meta_batch,
                    tf.cast(tf.ones(self.meta_batch_size, 1) * iteration_count, tf.int64)
                ),
                dtype=tf.float32,
                parallel_iterations=self.meta_batch_size
            )
            final_loss = tf.reduce_mean(tasks_final_losses)
        outer_gradients = outer_tape.gradient(final_loss, self.model.trainable_variables)
        if self.clip_gradients:
            outer_gradients = [tf.clip_by_value(grad, -10, 10) for grad in outer_gradients]
        self.optimizer.apply_gradients(zip(outer_gradients, self.model.trainable_variables))

    def train(self, epochs=5):
        self.train_dataset = self.get_train_dataset()
        self.val_dataset = self.get_val_dataset()
        start_epoch = self.load_model()
        iteration_count = start_epoch * self.train_dataset.steps_per_epoch

        pbar = tqdm(self.train_dataset)

        for epoch_count in range(start_epoch, epochs):
            if epoch_count != 0:
                if epoch_count % self.report_validation_frequency == 0:
                    self.report_validation_loss_and_accuracy(epoch_count)
                    if epoch_count != 0:
                        print('Train Loss: {}'.format(self.train_loss_metric.result().numpy()))
                        print('Train Accuracy: {}'.format(self.train_accuracy_metric.result().numpy()))

                with self.train_summary_writer.as_default():
                    tf.summary.scalar('Loss', self.train_loss_metric.result(), step=epoch_count)
                    tf.summary.scalar('Accuracy', self.train_accuracy_metric.result(), step=epoch_count)

                if epoch_count % self.save_after_epochs == 0:
                    self.save_model(epoch_count)

            self.train_accuracy_metric.reset_states()
            self.train_loss_metric.reset_states()

            for tasks_meta_batch, labels_meta_batch in self.train_dataset:
                self.meta_train_loop(tasks_meta_batch, labels_meta_batch, iteration_count)
                iteration_count += 1
                pbar.set_description_str('Epoch{}, Iteration{}: Train Loss: {}, Train Accuracy: {}'.format(
                    epoch_count,
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
        meta_batch_size=32,
        num_steps_ml=10,
        lr_inner_ml=0.4,
        num_steps_validation=10,
        save_after_epochs=500,
        meta_learning_rate=0.001,
        report_validation_frequency=50,
        log_train_images_after_iteration=-1,
    )

    maml.train(epochs=4000)
    # maml.evaluate(iterations=50, epochs_to_load_from=500)


def run_mini_imagenet():
    mini_imagenet_database = MiniImagenetDatabase(random_seed=-1)

    maml = ModelAgnosticMetaLearningModel(
        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.001,
        num_steps_validation=5,
        save_after_epochs=500,
        meta_learning_rate=0.001,
        report_validation_frequency=50,
        log_train_images_after_iteration=1000,
        least_number_of_tasks_val_test=50,
        clip_gradients=True
    )

    # maml.train(epochs=4000)
    maml.evaluate(50)


if __name__ == '__main__':
    run_omniglot()
    # run_mini_imagenet()
