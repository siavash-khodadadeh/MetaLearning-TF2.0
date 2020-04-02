import os

import tensorflow as tf

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.proto_networks import SimpleModelProto
from tf_datasets import OmniglotDatabase


class PrototypicalNetworks(ModelAgnosticMetaLearningModel):
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
            save_after_iterations,
            meta_learning_rate,
            report_validation_frequency,
            log_train_images_after_iteration,  # Set to -1 if you do not want to log train images.
            number_of_tasks_val=-1,
            number_of_tasks_test=-1,
            val_seed=-1,
            experiment_name=None
    ):
        self.experiment_name = experiment_name if experiment_name is not None else ''
        self.n = n
        self.k = k
        self.k_val_ml = k_val_ml
        self.k_val_val = k_val_val
        self.k_val_test = k_val_test
        self.k_test = k_test
        self.meta_batch_size = meta_batch_size
        self.save_after_iterations = save_after_iterations
        self.log_train_images_after_iteration = log_train_images_after_iteration
        self.report_validation_frequency = report_validation_frequency
        self.number_of_tasks_val = number_of_tasks_val
        self.number_of_tasks_test = number_of_tasks_test
        self.val_seed = val_seed
        super(ModelAgnosticMetaLearningModel, self).__init__(database, network_cls)

        self.model = self.initialize_network()

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

    def get_config_info(self):
        config_str = f'model-{self.network_cls.name}_' \
                     f'mbs-{self.meta_batch_size}_' \
                     f'n-{self.n}_' \
                     f'k-{self.k}_'
        if self.experiment_name != '':
            config_str += '_' + self.experiment_name

        return config_str

    def initialize_network(self):
        model = self.network_cls()
        model(tf.zeros(shape=(self.n * self.k, *self.database.input_shape)))

        return model

    def report_validation_loss_and_accuracy(self, epoch_count):
        self.val_loss_metric.reset_states()
        self.val_accuracy_metric.reset_states()

        val_counter = 0
        for tmb, lmb in self.val_dataset:
            val_counter += 1
            for task, labels in zip(tmb, lmb):
                support_set, query_set, support_labels, query_labels = self.get_task_train_and_val_ds(task, labels)
                if val_counter % 5 == 0:
                    step = epoch_count * self.val_dataset.steps_per_epoch + val_counter
                    self.log_images(self.val_summary_writer, support_set, query_set, step)

                ce_loss, predictions, query_classes = self.proto_net(
                    support_set,
                    query_set,
                    query_labels,
                    training=False
                )
                self.train_loss_metric.update_state(ce_loss)
                self.val_loss_metric.update_state(ce_loss)
                self.val_accuracy_metric.update_state(query_classes, predictions)

        self.log_metric(self.val_summary_writer, 'Loss', self.val_loss_metric, step=epoch_count)
        self.log_metric(self.val_summary_writer, 'Accuracy', self.val_accuracy_metric, step=epoch_count)

        print('Validation Loss: {}'.format(self.val_loss_metric.result().numpy()))
        print('Validation Accuracy: {}'.format(self.val_accuracy_metric.result().numpy()))

    def euclidean_distance(self, a, b):
        # a.shape = N x D
        # b.shape = M x D
        N, D = tf.shape(a)[0], tf.shape(a)[1]
        M = tf.shape(b)[0]
        a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
        b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
        return tf.reduce_mean(tf.square(a - b), axis=2)

    def proto_net(self, support_set, query_set, query_labels, training):
        support_set = self.model(support_set, training=True)
        query_set = self.model(query_set, training=training)
        support_set = tf.reshape(support_set, (self.n, self.k, -1))
        support_set = tf.reduce_mean(support_set, axis=1)
        dists = self.euclidean_distance(support_set, query_set)
        log_p_y = tf.transpose(tf.nn.log_softmax(-dists))

        ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.math.multiply(query_labels, log_p_y), axis=-1), (-1, )))
        predictions = tf.argmax(log_p_y, axis=-1)
        query_classes = tf.argmax(query_labels, axis=-1)

        return ce_loss, predictions, query_classes

    @tf.function
    def get_losses_of_tasks_batch(self, inputs):
        task, labels, iteration_count = inputs

        support_set, query_set, support_labels, query_labels = self.get_task_train_and_val_ds(task, labels)

        if self.log_train_images_after_iteration != -1 and \
                iteration_count % self.log_train_images_after_iteration == 0:

            self.log_images(self.train_summary_writer, support_set, query_set, step=iteration_count)

            with tf.device('cpu:0'):
                with self.train_summary_writer.as_default():
                    for var in self.model.variables:
                        tf.summary.histogram(var.name, var, step=iteration_count)

        # implement prototypcial network for one task
        ce_loss, predictions, query_classes = self.proto_net(support_set, query_set, query_labels, training=True)

        self.train_loss_metric.update_state(ce_loss)
        self.train_accuracy_metric.update_state(
            query_classes,
            predictions
        )
        return ce_loss

    def evaluate(self, epochs_to_load_from=None):
        self.test_dataset = self.get_test_dataset()
        self.load_model(epochs=epochs_to_load_from)
        test_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/test/')
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        test_accuracy_metric = tf.metrics.Accuracy()
        test_loss_metric = tf.metrics.Mean()

        for tmb, lmb in self.test_dataset:
            for task, labels in zip(tmb, lmb):
                support_set, query_set, support_labels, query_labels = self.get_task_train_and_val_ds(task, labels)
                ce_loss, predictions, query_classes = self.proto_net(
                    support_set,
                    query_set,
                    query_labels,
                    training=False
                )
                test_loss_metric.update_state(ce_loss)
                test_accuracy_metric.update_state(
                    query_classes,
                    predictions
                )

            self.log_metric(test_summary_writer, 'Loss', test_loss_metric, step=1)
            self.log_metric(test_summary_writer, 'Accuracy', test_accuracy_metric, step=1)

            print('Test Loss: {}'.format(test_loss_metric.result().numpy()))
            print('Test Accuracy: {}'.format(test_accuracy_metric.result().numpy()))

        return test_accuracy_metric.result().numpy()


def run_omniglot():
    omniglot_database = OmniglotDatabase(
        random_seed=47,
        num_train_classes=1200,
        num_val_classes=100,
    )

    proto_net = PrototypicalNetworks(
        database=omniglot_database,
        network_cls=SimpleModelProto,
        n=5,
        k=5,
        meta_batch_size=32,
        save_after_epochs=300,
        meta_learning_rate=0.001,
        report_validation_frequency=50,
        log_train_images_after_iteration=100,
    )

    proto_net.train(epochs=1501)
    proto_net.evaluate()


if __name__ == '__main__':
    run_omniglot()
    # run_mini_imagenet()
    # run_celeba()
