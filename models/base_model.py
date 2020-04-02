import os
import sys
from abc import abstractmethod

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils import combine_first_two_axes


class SetupCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.setup()
        return obj


class BaseModel(metaclass=SetupCaller):
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
        meta_learning_rate,
        save_after_iterations,
        report_validation_frequency,
        log_train_images_after_iteration,  # Set to -1 if you do not want to log train images.
        number_of_tasks_val,
        number_of_tasks_test,
        val_seed,  # The seed for validation dataset. -1 means change the samples for each report.
        experiment_name=None,
    ):
        self.database = database
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.experiment_name = experiment_name
        self.n = n
        self.k = k
        self.k_val_ml = k_val_ml
        self.k_val_val = k_val_val
        self.k_val_test = k_val_test
        self.k_test = k_test
        self.meta_batch_size = meta_batch_size
        self.meta_learning_rate = meta_learning_rate
        self.save_after_iterations = save_after_iterations
        self.log_train_images_after_iteration = log_train_images_after_iteration
        self.report_validation_frequency = report_validation_frequency
        self.number_of_tasks_val = number_of_tasks_val
        self.number_of_tasks_test = number_of_tasks_test
        self.val_seed = val_seed

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
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/val/')
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)
        self.checkpoint_dir = os.path.join(self._root, self.get_config_info(), 'saved_models/')

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
                    max_outputs=self.n * (self.k + self.k_val_ml)
                )
                tf.summary.image(
                    'validation',
                    val_ds,
                    step=step,
                    max_outputs=self.n * (self.k + self.k_val_ml)
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
        dataset = self.database.get_supervised_meta_learning_dataset(
            self.database.train_folders,
            n=self.n,
            k=self.k,
            k_validation=self.k_val_ml,
            meta_batch_size=self.meta_batch_size
        )
        # steps_per_epoch = dataset.steps_per_epoch
        # setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset

    def get_val_dataset(self):
        val_dataset = self.database.get_supervised_meta_learning_dataset(
            self.database.val_folders,
            n=self.n,
            k=self.k,
            k_validation=self.k_val_val,
            meta_batch_size=1,
            seed=self.val_seed,
        )
        val_dataset = val_dataset.repeat(-1)
        val_dataset = val_dataset.take(self.number_of_tasks_val)
        setattr(val_dataset, 'steps_per_epoch', self.number_of_tasks_val)
        return val_dataset

    def get_test_dataset(self, seed=-1):
        test_dataset = self.database.get_supervised_meta_learning_dataset(
            self.database.test_folders,
            n=self.n,
            k=self.k_test,
            k_validation=self.k_val_test,
            meta_batch_size=1,
            seed=seed
        )
        test_dataset = test_dataset.repeat(-1)
        test_dataset = test_dataset.take(self.number_of_tasks_test)
        setattr(test_dataset, 'steps_per_epoch', self.number_of_tasks_test)
        return test_dataset

    def train(self, iterations=5):
        self.train_dataset = self.get_train_dataset()
        iteration_count = self.load_model()
        epoch_count = iteration_count // self.train_dataset.steps_per_epoch
        pbar = tqdm(self.train_dataset)

        train_accuracy_metric = tf.metrics.Mean()
        train_accuracy_metric.reset_states()
        train_loss_metric = tf.metrics.Mean()
        train_loss_metric.reset_states()

        should_continue = iteration_count < iterations
        while should_continue:
            for (train_ds, val_ds), (train_labels, val_labels) in self.train_dataset:
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
                    if epoch_count != 0:
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
                task_final_loss, task_final_acc = self.get_losses_of_tasks_batch(
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

    @abstractmethod
    def get_losses_of_tasks_batch(self, inputs):
        pass

    @abstractmethod
    def initialize_network(self):
        pass

    @abstractmethod
    def get_config_str(self):
        pass

    @abstractmethod
    def report_validation_loss_and_accuracy(self, iteration_count):
        pass
