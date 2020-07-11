import os
import sys
from abc import abstractmethod

import tensorflow as tf
import numpy as np
from tqdm import tqdm

import settings
from utils import combine_first_two_axes, keep_keys_with_greater_than_equal_k_items


class SetupCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.setup()
        return obj


class BaseModel(metaclass=SetupCaller):
    def __init__(
        self,
        database,
        data_loader_cls,
        network_cls,
        n,
        k_ml,
        k_val_ml,
        k_val,
        k_val_val,
        k_test,
        k_val_test,
        meta_batch_size,
        meta_learning_rate,
        save_after_iterations,
        report_validation_frequency,
        log_train_images_after_iteration,  # Set to -1 if you do not want to log train images.
        num_tasks_val,
        val_seed=-1,  # The seed for validation dataset. -1 means change the samples for each report.
        experiment_name=None,
        val_database=None,
        test_database=None,
    ):
        self.database = database
        self.val_database = val_database if val_database is not None else self.database
        self.test_database = test_database if test_database is not None else self.database

        self.n = n
        self.k_ml = k_ml
        self.k_val_ml = k_val_ml
        self.k_val = k_val if k_val is not None else self.k_ml
        self.k_val_val = k_val_val
        self.k_test = k_test
        self.k_val_test = k_val_test

        self.meta_batch_size = meta_batch_size
        self.num_tasks_val = num_tasks_val
        self.val_seed = val_seed
        self.data_loader = self.init_data_loader(data_loader_cls)

        self.experiment_name = experiment_name
        self.meta_learning_rate = meta_learning_rate
        self.save_after_iterations = save_after_iterations
        self.log_train_images_after_iteration = log_train_images_after_iteration
        self.report_validation_frequency = report_validation_frequency

        self._root = self.get_root()
        self.train_log_dir = None
        self.train_summary_writer = None
        self.val_log_dir = None
        self.val_summary_writer = None
        self.checkpoint_dir = None

        self.network_cls = network_cls
        self.model = self.initialize_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=meta_learning_rate)

        self.val_accuracy_metric = tf.metrics.Mean()
        self.val_loss_metric = tf.metrics.Mean()

    def setup(self):
        """Setup is called right after init. This is to make sure that all the required fields are assigned.
        For example, num_steps in ml is in get_config_info(), however, it is not set in __init__ of the base model
        because it is a field for maml."""
        self.train_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/train/')
        self.val_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/val/')
        self.checkpoint_dir = os.path.join(self._root, self.get_config_info(), 'saved_models/')

    def init_data_loader(self, data_loader_cls):
        return data_loader_cls(
            database=self.database,
            val_database=self.val_database,
            test_database=self.test_database,
            n=self.n,
            k_ml=self.k_ml,
            k_val_ml=self.k_val_ml,
            k_val=self.k_val,
            k_val_val=self.k_val_val,
            k_test=self.k_test,
            k_val_test=self.k_val_test,
            meta_batch_size=self.meta_batch_size,
            num_tasks_val=self.num_tasks_val,
            val_seed=self.val_seed
        )

    def get_root(self):
        return os.path.dirname(sys.argv[0])

    def get_config_info(self):
        config_info = self.get_config_str()
        if self.experiment_name is not None:
            config_info += '_' + self.experiment_name

        return config_info

    def post_process_outer_gradients(self, outer_gradients):
        return outer_gradients

    def log_images(self, summary_writer, train_ds, val_ds, step):
        with tf.device('cpu:0'):
            with summary_writer.as_default():
                tf.summary.image(
                    'train',
                    train_ds,
                    step=step,
                    max_outputs=self.n * (self.k_ml + self.k_val_ml)
                )
                tf.summary.image(
                    'validation',
                    val_ds,
                    step=step,
                    max_outputs=self.n * (self.k_ml + self.k_val_ml)
                )

    def save_model(self, iterations):
        self.model.save_weights(os.path.join(self.checkpoint_dir, f'model.ckpt-{iterations}'))

    def load_model(self, iterations=None):
        iteration_count = 0
        if iterations is not None:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model.ckpt-{iterations}')
            iteration_count = iterations
        else:
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)

        if checkpoint_path is not None:
            try:
                self.model.load_weights(checkpoint_path)
                iteration_count = int(checkpoint_path[checkpoint_path.rindex('-') + 1:])
                print(f'==================\nResuming Training\n======={iteration_count}=======\n==================')
            except Exception as e:
                print('Could not load the previous checkpoint!')
                print(e)
                exit()

        else:
            print('No previous checkpoint found!')

        return iteration_count

    def log_histograms(self, step):
        with tf.device('cpu:0'):
            with self.train_summary_writer.as_default():
                for var in self.model.variables:
                    tf.summary.histogram(var.name, var, step=step)

                # for k in range(len(self.updated_models)):
                #     var_count = 0
                #     if hasattr(self.updated_models[k], 'meta_trainable_variables'):
                #         for var in self.updated_models[k].meta_trainable_variables:
                #             var_count += 1
                #     tf.summary.histogram(f'updated_model_{k}_' + str(var_count), var, step=iteration_count)

    def get_train_dataset(self):
        return self.data_loader.get_train_dataset()

    def get_val_dataset(self):
        return self.data_loader.get_val_dataset()

    def get_test_dataset(self, num_tasks, seed=-1):
        return self.data_loader.get_test_dataset(num_tasks, seed)

    def train(self, iterations=5):
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)
        train_dataset = self.get_train_dataset()
        iteration_count = self.load_model()
        epoch_count = iteration_count // tf.data.experimental.cardinality(train_dataset)
        pbar = tqdm(train_dataset)

        train_accuracy_metric = tf.metrics.Mean()
        train_accuracy_metric.reset_states()
        train_loss_metric = tf.metrics.Mean()
        train_loss_metric.reset_states()

        should_continue = iteration_count < iterations
        while should_continue:
            for (train_ds, val_ds), (train_labels, val_labels) in train_dataset:
                train_acc, train_loss = self.meta_train_loop(train_ds, val_ds, train_labels, val_labels)
                train_accuracy_metric.update_state(train_acc)
                train_loss_metric.update_state(train_loss)
                iteration_count += 1
                if (
                        self.log_train_images_after_iteration != -1 and
                        iteration_count % self.log_train_images_after_iteration == 0
                ):
                    self.log_images(
                        self.train_summary_writer,
                        combine_first_two_axes(train_ds[0, ...]),
                        combine_first_two_axes(val_ds[0, ...]),
                        step=iteration_count
                    )
                    self.log_histograms(step=iteration_count)

                if iteration_count != 0 and iteration_count % self.save_after_iterations == 0:
                    self.save_model(iteration_count)

                if iteration_count % self.report_validation_frequency == 0:
                    self.report_validation_loss_and_accuracy(iteration_count)
                    if iteration_count != 0:
                        print('Train Loss: {}'.format(train_loss_metric.result().numpy()))
                        print('Train Accuracy: {}'.format(train_accuracy_metric.result().numpy()))
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('Loss', train_loss_metric.result(), step=iteration_count)
                        tf.summary.scalar('Accuracy', train_accuracy_metric.result(), step=iteration_count)
                    train_accuracy_metric.reset_states()
                    train_loss_metric.reset_states()

                pbar.set_description_str('Epoch{}, Iteration{}: Train Loss: {}, Train Accuracy: {}'.format(
                    epoch_count,
                    iteration_count,
                    train_loss_metric.result().numpy(),
                    train_accuracy_metric.result().numpy()
                ))
                pbar.update(1)

                if iteration_count >= iterations:
                    should_continue = False
                    break

            epoch_count += 1

    def log_metric(self, summary_writer, name, metric, step):
        with summary_writer.as_default():
            tf.summary.scalar(name, metric.result(), step=step)

    @tf.function
    def meta_train_loop(self, train_ds, val_ds, train_labels, val_labels):
        with tf.GradientTape(persistent=False) as outer_tape:
            tasks_final_losses = list()
            tasks_final_accs = list()

            for i in range(self.meta_batch_size):
                task_final_acc, task_final_loss = self.get_losses_of_tasks_batch(method='train')(
                    (train_ds[i, ...], val_ds[i, ...], train_labels[i, ...], val_labels[i, ...])
                )
                tasks_final_losses.append(task_final_loss)
                tasks_final_accs.append(task_final_acc)

            final_acc = tf.reduce_mean(tasks_final_accs)
            # self.train_accuracy_metric.update_state(final_acc)
            final_loss = tf.reduce_mean(tasks_final_losses)
            # self.train_loss_metric.update_state(final_loss)

        outer_gradients = outer_tape.gradient(final_loss, self.model.trainable_variables)
        self.post_process_outer_gradients(outer_gradients)
        self.optimizer.apply_gradients(zip(outer_gradients, self.model.trainable_variables))

        return final_acc, final_loss

    def evaluate(self, iterations, num_tasks, iterations_to_load_from=None, seed=-1, use_val_batch_statistics=True):
        """If you set use val batch statistics to true, then the batch information from all the test samples will be
        used for batch normalization layers (like MAML experiments), otherwise batch normalization layers use the
        average and variance which they learned during the updates."""
        # TODO add ability to set batch norm momentum if use_val_batch_statistics=False
        self.test_dataset = self.get_test_dataset(num_tasks=num_tasks, seed=seed)
        self.load_model(iterations=iterations_to_load_from)

        accs = list()
        losses = list()
        losses_func = self.get_losses_of_tasks_batch(
            method='test',
            iterations=iterations,
            use_val_batch_statistics=use_val_batch_statistics
        )
        counter = 0
        for (train_ds, val_ds), (train_labels, val_labels) in self.test_dataset:
            remainder_num = num_tasks // 20
            if remainder_num == 0:
                remainder_num = 1
            if counter % remainder_num == 0:
                print(f'{counter} / {num_tasks} are evaluated.')

            counter += 1
            tasks_final_accuracy, tasks_final_losses = tf.map_fn(
                losses_func,
                elems=(
                    train_ds,
                    val_ds,
                    train_labels,
                    val_labels,
                ),
                dtype=(tf.float32, tf.float32),
                parallel_iterations=1
            )
            final_loss = tf.reduce_mean(tasks_final_losses)
            final_acc = tf.reduce_mean(tasks_final_accuracy)
            losses.append(final_loss)
            accs.append(final_acc)

        final_acc_mean = np.mean(accs)
        final_acc_std = np.std(accs)

        print(f'loss mean: {np.mean(losses)}')
        print(f'loss std: {np.std(losses)}')
        print(f'accuracy mean: {final_acc_mean}')
        print(f'accuracy std: {final_acc_std}')
        # Free the seed :D
        if seed != -1:
            np.random.seed(None)

        confidence_interval = 1.96 * final_acc_std / np.sqrt(num_tasks)

        print(
            f'final acc: {final_acc_mean} +- {confidence_interval}'
        )
        print(
            f'final acc: {final_acc_mean * 100:0.2f} +- {confidence_interval * 100:0.2f}'
        )
        return np.mean(accs)

    def report_validation_loss_and_accuracy(self, epoch_count):
        self.val_loss_metric.reset_states()
        self.val_accuracy_metric.reset_states()

        val_counter = 0
        loss_func = self.get_losses_of_tasks_batch(method='val')
        val_dataset = self.get_val_dataset()
        for (train_ds, val_ds), (train_labels, val_labels) in val_dataset:
            val_counter += 1
            # TODO fix validation logging
            if settings.DEBUG:
                if val_counter % 5 == 0:
                    step = epoch_count * val_dataset.steps_per_epoch + val_counter
                    # pick the first task in meta batch
                    log_train_ds = combine_first_two_axes(train_ds[0, ...])
                    log_val_ds = combine_first_two_axes(val_ds[0, ...])
                    self.log_images(self.val_summary_writer, log_train_ds, log_val_ds, step)

            tasks_final_accuracy, tasks_final_losses = tf.map_fn(
                loss_func,
                elems=(
                    train_ds,
                    val_ds,
                    train_labels,
                    val_labels,
                ),
                dtype=(tf.float32, tf.float32),
                parallel_iterations=1
            )
            final_loss = tf.reduce_mean(tasks_final_losses)
            final_acc = tf.reduce_mean(tasks_final_accuracy)
            self.val_loss_metric.update_state(final_loss)
            self.val_accuracy_metric.update_state(final_acc)

        self.log_metric(self.val_summary_writer, 'Loss', self.val_loss_metric, step=epoch_count)
        self.log_metric(self.val_summary_writer, 'Accuracy', self.val_accuracy_metric, step=epoch_count)
        print('Validation Loss: {}'.format(self.val_loss_metric.result().numpy()))
        print('Validation Accuracy: {}'.format(self.val_accuracy_metric.result().numpy()))

    @abstractmethod
    def get_losses_of_tasks_batch(self, method='train', **kwargs):
        pass

    @abstractmethod
    def initialize_network(self):
        pass

    @abstractmethod
    def get_config_str(self):
        pass
