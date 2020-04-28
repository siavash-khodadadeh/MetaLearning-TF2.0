import os

import tensorflow as tf

from models.maml.maml import ModelAgnosticMetaLearningModel
from databases import OmniglotDatabase, MiniImagenetDatabase, CelebADatabase
from networks.maml_umtra_networks import SimpleModel, MiniImagenetModel


class MAMLAbstractLearner(ModelAgnosticMetaLearningModel):
    def get_root(self):
        return os.path.dirname(__file__)

    def initialize_network(self):
        model = self.network_cls(num_classes=1)
        model(tf.zeros(shape=(self.n * self.k, * self.database.input_shape)))

        return model

    def convert_labels_to_real_labels(self, labels):
        return labels

    def predict_class_labels_from_logits(self, logits):
        logits = tf.reshape(logits, (-1,))
        labels = tf.sigmoid(logits)
        labels = tf.math.round(labels)
        return labels

    def outer_loss(self, labels, logits):
        logits = tf.reshape(logits, (-1, ))
        loss = tf.reduce_sum(
            # tf.losses.binary_crossentropy(labels, logits, from_logits=True)
            tf.nn.weighted_cross_entropy_with_logits(
                labels,
                logits,
                pos_weight=self.n - 1,
                name=None
            )
        )
        return loss

    def inner_loss(self, train_labels, logits):
        logits = tf.reshape(logits, (-1, ))
        loss = tf.reduce_sum(
            # tf.losses.binary_crossentropy(train_labels, logits, from_logits=True)
            tf.nn.weighted_cross_entropy_with_logits(
                train_labels,
                logits,
                pos_weight=self.n - 1,
                name=None
            )
        )
        return loss

    def create_meta_model(self, updated_model, model, gradients):
        k = 0
        variables = list()
        layers_to_consider = 1

        for i in range(len(model.layers)):
            if isinstance(model.layers[i], tf.keras.layers.Conv2D) or \
                    isinstance(model.layers[i], tf.keras.layers.Dense):
                if i >= len(model.layers) - layers_to_consider:
                    updated_model.layers[i].kernel = model.layers[i].kernel - self.lr_inner_ml * gradients[k]
                else:
                    updated_model.layers[i].kernel = model.layers[i].kernel
                k += 1
                variables.append(updated_model.layers[i].kernel)

                if i >= len(model.layers) - layers_to_consider:
                    updated_model.layers[i].bias = model.layers[i].bias - self.lr_inner_ml * gradients[k]
                else:
                    updated_model.layers[i].bias = model.layers[i].bias
                k += 1
                variables.append(updated_model.layers[i].bias)

            elif isinstance(model.layers[i], tf.keras.layers.BatchNormalization):
                if hasattr(model.layers[i], 'moving_mean') and model.layers[i].moving_mean is not None:
                    updated_model.layers[i].moving_mean.assign(model.layers[i].moving_mean)
                if hasattr(model.layers[i], 'moving_variance') and model.layers[i].moving_variance is not None:
                    updated_model.layers[i].moving_variance.assign(model.layers[i].moving_variance)
                if hasattr(model.layers[i], 'gamma') and model.layers[i].gamma is not None:
                    if i >= len(model.layers) - layers_to_consider:
                        updated_model.layers[i].gamma = model.layers[i].gamma - self.lr_inner_ml * gradients[k]
                    else:
                        updated_model.layers[i].gamma = model.layers[i].gamma
                    k += 1
                    variables.append(updated_model.layers[i].gamma)
                if hasattr(model.layers[i], 'beta') and model.layers[i].beta is not None:
                    if i >= len(model.layers) - layers_to_consider:
                        updated_model.layers[i].beta = model.layers[i].beta - self.lr_inner_ml * gradients[k]
                    else:
                        updated_model.layers[i].beta = model.layers[i].beta
                    k += 1
                    variables.append(updated_model.layers[i].beta)

            elif isinstance(model.layers[i], tf.keras.layers.LayerNormalization):
                if hasattr(model.layers[i], 'gamma') and model.layers[i].gamma is not None:
                    if i >= len(model.layers) - layers_to_consider:
                        updated_model.layers[i].gamma = model.layers[i].gamma - self.lr_inner_ml * gradients[k]
                    else:
                        updated_model.layers[i].gamma = model.layers[i].gamma
                    k += 1
                    variables.append(updated_model.layers[i].gamma)
                if hasattr(model.layers[i], 'beta') and model.layers[i].beta is not None:
                    if i >= len(model.layers) - layers_to_consider:
                        updated_model.layers[i].beta = model.layers[i].beta - self.lr_inner_ml * gradients[k]
                    else:
                        updated_model.layers[i].beta = model.layers[i].beta
                    k += 1
                    variables.append(updated_model.layers[i].beta)

        setattr(updated_model, 'meta_trainable_variables', variables)

    def get_train_dataset(self):
        dataset = self.database.get_abstract_learning_dataset(
            self.database.train_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=self.meta_batch_size,
        )

        return dataset

    def get_val_dataset(self):
        val_dataset = self.database.get_abstract_learning_dataset(
            self.database.val_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=1,
            reshuffle_each_iteration=False
        )
        steps_per_epoch = max(val_dataset.steps_per_epoch, self.least_number_of_tasks_val_test)
        val_dataset = val_dataset.repeat(-1)
        val_dataset = val_dataset.take(steps_per_epoch)
        setattr(val_dataset, 'steps_per_epoch', steps_per_epoch)
        return val_dataset


def run_omniglot():
    omniglot_database = OmniglotDatabase(
        random_seed=47,
        num_train_classes=1200,
        num_val_classes=100,
    )

    maml = MAMLAbstractLearner(
        database=omniglot_database,
        network_cls=SimpleModel,
        n=5,
        k=1,
        meta_batch_size=32,
        num_steps_ml=1,
        lr_inner_ml=0.4,
        num_steps_validation=10,
        save_after_epochs=500,
        meta_learning_rate=0.001,
        report_validation_frequency=10,
        log_train_images_after_iteration=-1,
    )

    maml.train(epochs=4000)
    maml.evaluate(iterations=50)


def run_mini_imagenet():
    mini_imagenet_database = MiniImagenetDatabase(random_seed=-1)

    maml = MAMLAbstractLearner(
        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=4,
        meta_batch_size=1,
        num_steps_ml=5,
        lr_inner_ml=0.01,
        num_steps_validation=5,
        save_after_epochs=500,
        meta_learning_rate=0.001,
        report_validation_frequency=50,
        log_train_images_after_iteration=1000,
        least_number_of_tasks_val_test=50,
        clip_gradients=True
    )

    maml.train(epochs=30000)
    maml.evaluate(50)


def run_celeba():
    celeba_database = CelebADatabase(random_seed=-1)
    maml = MAMLAbstractLearner(
        database=celeba_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=4,
        meta_batch_size=1,
        num_steps_ml=5,
        lr_inner_ml=0.01,
        num_steps_validation=5,
        save_after_epochs=20,
        meta_learning_rate=0.001,
        report_validation_frequency=1,
        log_train_images_after_iteration=1000,
        least_number_of_tasks_val_test=50,
        clip_gradients=True,
        experiment_name='celeba'
    )
    maml.train(epochs=100)


if __name__ == '__main__':
    # run_omniglot()
    # run_mini_imagenet()
    run_celeba()
