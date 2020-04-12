import tensorflow as tf
import numpy as np

from models.base_model import BaseModel
from networks.proto_networks import SimpleModelProto, VGGSmallModel
from tf_datasets import OmniglotDatabase, VGGFace2Database
from utils import combine_first_two_axes


class PrototypicalNetworks(BaseModel):
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

        super(PrototypicalNetworks, self).__init__(
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

    def get_losses_of_tasks_batch(self, method='train', **kwargs):
        if method == 'train':
            return self.get_loss_func(training=True, k=self.k)
        elif method == 'val':
            return self.get_loss_func(training=True, k=self.k)
        elif method == 'test':
            return self.get_loss_func(training=kwargs['use_val_batch_statistics'], k=self.k_test)

    def get_loss_func(self, training, k):
        @tf.function
        def f(inputs):
            train_ds, val_ds, train_labels, val_labels = inputs
            train_ds = combine_first_two_axes(train_ds)
            val_ds = combine_first_two_axes(val_ds)

            ce_loss, predictions, query_labels = self.proto_net(
                training=training,
                support_set=train_ds,
                query_set=val_ds,
                query_labels=val_labels,
                k=k
            )
            real_labels = self.convert_labels_to_real_labels(val_labels)
            val_acc = tf.reduce_mean(tf.cast(tf.equal(predictions, real_labels), tf.float32))
            return val_acc, ce_loss

        return f

    def euclidean_distance(self, a, b):
        # a.shape = N x D
        # b.shape = M x D
        N, D = tf.shape(a)[0], tf.shape(a)[1]
        M = tf.shape(b)[0]
        a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
        b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
        return tf.reduce_mean(tf.square(a - b), axis=2)

    def proto_net(self, support_set, query_set, query_labels, training, k):
        support_set = self.model(support_set, training=True)
        query_set = self.model(query_set, training=training)
        support_set = tf.reshape(support_set, (self.n, k, -1))
        support_set = tf.reduce_mean(support_set, axis=1)
        dists = self.euclidean_distance(support_set, query_set)
        log_p_y = tf.transpose(tf.nn.log_softmax(-dists))

        ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.math.multiply(query_labels, log_p_y), axis=-1), (-1, )))
        predictions = tf.argmax(log_p_y, axis=-1)
        query_classes = tf.argmax(query_labels, axis=-1)

        return ce_loss, predictions, query_classes


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
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=3,
        meta_batch_size=32,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,  # Set to -1 if you do not want to log train images.
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        val_seed=-1,
        experiment_name=None
    )

    proto_net.train(iterations=1000)
    # proto_net.evaluate(-1)


def run_celeba():
    from models.protonets.inception_resnet_v1 import InceptionResNetV1

    celeba_database = VGGFace2Database(input_shape=(224, 224, 3))
    # celeba_database = CelebADatabase(input_shape=(224, 224, 3))
    # celeba_database = LFWDatabase(input_shape=(224, 224, 3))
    # celeba_database = CelebADatabase(input_shape=(84, 84, 3))
    proto_net = PrototypicalNetworks(
        database=celeba_database,
        network_cls=VGGSmallModel,
        # network_cls=InceptionResNetV1,
        # network_cls=MiniImagenetModel,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=1,
        meta_batch_size=4,
        save_after_iterations=500,
        # meta_learning_rate=0.001,
        meta_learning_rate=0.0001,
        report_validation_frequency=500,
        log_train_images_after_iteration=1000,  # Set to -1 if you do not want to log train images.
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        val_seed=42,
        # experiment_name='mb_16_mlr001'
        experiment_name='mb_16mlr000005'
    )
    # Start with 0.00005
    # From 44000 train with smaller learning rate 0.000001
    # From 59000 train with smaller learning rate 0.0000005
    proto_net.train(iterations=90000)
    # proto_net.evaluate(-1, seed=42)


if __name__ == '__main__':
    # run_omniglot()
    # run_mini_imagenet()
    # from datetime import datetime
    # begin_time = datetime.now()
    run_celeba()
    # print(datetime.now() - begin_time)
