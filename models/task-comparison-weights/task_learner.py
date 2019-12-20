import tensorflow as tf

from models.base_model import BaseModel
from networks import SimpleModel
from tf_datasets import OmniglotDatabase
from utils import combine_first_two_axes


class TaskLearner(BaseModel):
    def __init__(self, database, network_cls, n, k):
        self.n = n
        self.k = k
        super(TaskLearner, self).__init__(database, network_cls)

    def get_task_train_and_val_ds(self, task, labels):
        train_ds, val_ds = tf.split(task, num_or_size_splits=2)
        train_labels, val_labels = tf.split(labels, num_or_size_splits=2)

        train_ds = combine_first_two_axes(tf.squeeze(train_ds, axis=0))
        val_ds = combine_first_two_axes(tf.squeeze(val_ds, axis=0))
        train_labels = combine_first_two_axes(tf.squeeze(train_labels, axis=0))
        val_labels = combine_first_two_axes(tf.squeeze(val_labels, axis=0))

        return train_ds, val_ds, train_labels, val_labels

    def train(self):
        self.train_dataset = self.get_train_dataset()

        for tasks_meta_batch, labels_meta_batch in self.train_dataset:
            task = tf.squeeze(tasks_meta_batch, axis=0)
            labels = tf.squeeze(labels_meta_batch, axis=0)

            train_ds, val_ds, train_labels, val_labels = self.get_task_train_and_val_ds(task, labels)

            train_network = self.network_cls(self.n)

            for layer in train_network.layers[:-1]:
                layer.trainable = False

            train_network.compile(
                optimizer='sgd',
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.categorical_accuracy]
            )
            train_network.fit(train_ds, train_labels, epochs=100)
            train_network.evaluate(train_ds, train_labels)
            train_network.evaluate(val_ds, val_labels)

            # task_data = tf.concat((train_ds, val_ds), axis=0)
            # task_labels = tf.concat((train_labels, val_labels), axis=0)
            #
            # task_network = self.network_cls(self.n)
            # task_network.compile(
            #     optimizer='sgd',
            #     loss=tf.losses.CategoricalCrossentropy(from_logits=True),
            #     metrics=[tf.metrics.categorical_accuracy]
            # )
            #
            # task_network.fit(task_data, task_labels, epochs=100)
            #
            # task_network.evaluate(train_ds, train_labels)
            # task_network.evaluate(val_ds, val_labels)

            break

    def evaluate(self, iterations):
        pass

    def get_train_dataset(self):
        dataset = self.database.get_supervised_meta_learning_dataset(
            self.database.train_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=1
        )
        return dataset

    def get_val_dataset(self):
        pass

    def get_test_dataset(self):
        pass

    def get_config_info(self):
        pass


def run_omniglot():
    omniglot_database = OmniglotDatabase(
        random_seed=-1,
        num_train_classes=1200,
        num_val_classes=100,
    )
    tl = TaskLearner(omniglot_database, SimpleModel, n=5, k=9)
    tl.train()


if __name__ == '__main__':
    run_omniglot()

