import tensorflow as tf
import numpy as np

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.proto_networks import SimpleModelProto
from tf_datasets import OmniglotDatabase
from utils import combine_first_two_axes


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

        super(ModelAgnosticMetaLearningModel, self).__init__(
            database=database,
            network_cls=network_cls,
            n=n,
            k=k,
            k_val_ml=k_val_ml,
            k_val_val=k_val_val,
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
        )

    def get_config_str(self):
        return f'model-{self.network_cls.name}_' \
               f'mbs-{self.meta_batch_size}_' \
               f'n-{self.n}_' \
               f'k-{self.k}_' \
               f'kvalml-{self.k_val_ml}'

    def initialize_network(self):
        model = self.network_cls()
        model(tf.zeros(shape=(self.n * self.k, *self.database.input_shape)))

        return model

    def convert_labels_to_real_labels(self, labels):
        return tf.argmax(labels, axis=-1)

    def get_losses_of_tasks_batch(self, inputs):
        pass

    def get_loss_func(self, use_val_batch_statistics=True):
        @tf.function
        def f(inputs):
            train_ds, val_ds, train_labels, val_labels = inputs
            train_ds = combine_first_two_axes(train_ds)
            val_ds = combine_first_two_axes(val_ds)

            ce_loss, predictions, query_labels = self.proto_net(
                training=use_val_batch_statistics,
                support_set=train_ds,
                query_set=val_ds,
                query_labels=val_labels
            )
            real_labels = self.convert_labels_to_real_labels(val_labels)
            val_acc = tf.reduce_mean(tf.cast(tf.equal(predictions, real_labels), tf.float32))
            return val_acc, ce_loss

        return f

    def report_validation_loss_and_accuracy(self, epoch_count):
        loss_func = self.get_loss_func()
        self.val_loss_metric.reset_states()
        self.val_accuracy_metric.reset_states()

        val_counter = 0
        for (train_ds, val_ds), (train_labels, val_labels) in self.get_val_dataset():
            val_counter += 1
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

    def evaluate(self, iterations, iterations_to_load_from=None, seed=-1, use_val_batch_statistics=True):
        self.test_dataset = self.get_test_dataset()
        self.load_model(iterations=iterations_to_load_from)

        accs = list()
        losses = list()
        loss_func = self.get_loss_func(use_val_batch_statistics=use_val_batch_statistics)

        counter = 0
        for (train_ds, val_ds), (train_labels, val_labels) in self.test_dataset:
            remainder_num = self.number_of_tasks_test // 20
            if remainder_num == 0:
                remainder_num = 1
            if counter % remainder_num == 0:
                print(f'{counter} / {self.number_of_tasks_test} are evaluated.')
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
            losses.append(final_loss)
            accs.append(final_acc)

        print(f'loss mean: {np.mean(losses)}')
        print(f'loss std: {np.std(losses)}')
        print(f'accuracy mean: {np.mean(accs)}')
        print(f'accuracy std: {np.std(accs)}')
        # Free the seed :D
        if seed != -1:
            np.random.seed(None)

        print(
            f'final acc: {np.mean(accs)} +- {1.96 * np.std(accs) / np.sqrt(self.number_of_tasks_test)}'
        )
        return np.mean(accs)


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
