import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tf_datasets import OmniglotDatabase
from networks import SimpleModel
from utils import combine_first_two_axes, average_gradients


# TODO
# Save the model and logs.
# Make batch norm settings in config if necassary.
# Add evaluation.
# Add tensorboard.
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
        self.updated_model = self.config['model']['class'](**self.config['model']['init_kwargs'])
        # self.updated_model.trainable = False
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.val_accuracy_metric = tf.metrics.Accuracy()
        self.val_loss_metric = tf.metrics.Mean()

    def update_updated_model(self, gradients):
        # self.updated_model.set_weights(self.model.get_weights())
        k = 0
        for i in range(len(self.model.layers)):
            if isinstance(self.model.layers[i], tf.keras.layers.Conv2D) or \
             isinstance(self.model.layers[i], tf.keras.layers.Dense):
                if self.model.layers[i].kernel.trainable:
                    self.updated_model.layers[i].kernel = \
                        self.model.layers[i].kernel - self.config['lr_inner_ml'] * gradients[k]
                    k += 1
                if self.model.layers[i].bias.trainable:
                    self.updated_model.layers[i].bias = \
                        self.model.layers[i].bias - self.config['lr_inner_ml'] * gradients[k]
                    k += 1
            elif isinstance(self.model.layers[i], tf.keras.layers.BatchNormalization):
                if hasattr(self.model.layers[i], 'moving_mean') and self.model.layers[i].moving_mean is not None:
                    self.updated_model.layers[i].moving_mean.assign(self.model.layers[i].moving_mean)
                if hasattr(self.model.layers[i], 'moving_variance') and self.model.layers[i].moving_variance is not None:
                    self.updated_model.layers[i].moving_variance.assign(self.model.layers[i].moving_variance)
                if hasattr(self.model.layers[i], 'gamma') and self.model.layers[i].gamma is not None:
                    self.updated_model.layers[i].gamma = \
                        self.model.layers[i].gamma - self.config['lr_inner_ml'] * gradients[k]
                    k += 1
                if hasattr(self.model.layers[i], 'beta') and self.model.layers[i].beta is not None:
                    self.updated_model.layers[i].beta = \
                        self.model.layers[i].beta - self.config['lr_inner_ml'] * gradients[k]
                    k += 1

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

    def train(self, epochs=5):
        tf.config.experimental_run_functions_eagerly(True)
        counter = 0
        self.model.predict(np.zeros(shape=(self.database_config['config']['train_dataset_kwargs']['n'] * self.database_config['config']['train_dataset_kwargs']['k'], 28, 28, 1)))
        self.updated_model.predict(np.zeros(shape=(self.database_config['config']['train_dataset_kwargs']['n'] * self.database_config['config']['train_dataset_kwargs']['k'], 28, 28, 1)))
        
        for tasks_meta_batch, labels_meta_batch in tqdm(self.database.train_ds):
            if counter % self.database.train_ds.steps_per_epoch == 0:
                # one epoch ends here
                self.val_loss_metric.reset_states()
                self.val_accuracy_metric.reset_states()

                val_counter = 0
                for tmb, lmb in self.database.val_ds:
                    if val_counter == self.database.val_ds.steps_per_epoch:
                        break
                    val_counter += 1
                    for task, labels in zip(tmb, lmb):
                        train_ds, val_ds, train_labels, val_labels = self.get_task_train_and_val_ds(task, labels)
                        train_loss, train_gradients, train_tape = self.get_train_loss_and_gradients(train_ds, train_labels)
                        self.update_updated_model(train_gradients)
                        updated_model_logits = self.updated_model(val_ds, training=False)
                        print(tf.argmax(updated_model_logits, axis=-1))
                        val_loss = tf.reduce_sum(tf.losses.categorical_crossentropy(val_labels, updated_model_logits, from_logits=True))
                        self.val_loss_metric.update_state(val_loss)
                        self.val_accuracy_metric.update_state(tf.argmax(val_labels, axis=-1), tf.argmax(updated_model_logits, axis=-1))
                
                print('Validation Error: {}'.format(self.val_loss_metric.result().numpy()))
                print('Validation Accuracy: {}'.format(self.val_accuracy_metric.result().numpy()))

            if counter == self.database.train_ds.steps_per_epoch * epochs:
                break

            counter += 1
            tasks_final_gradients = list()
            
            for task, labels in zip(tasks_meta_batch, labels_meta_batch):
                train_ds, val_ds, train_labels, val_labels = self.get_task_train_and_val_ds(task, labels)

                # self.model.forward(train_ds) # run forward pass to initialize weights
                # self.updated_model.forward(train_ds, training=False) # run forward pass to initialize weights
                with tf.GradientTape(persistent=True) as val_tape:
                    train_loss, train_gradients, train_tape = self.get_train_loss_and_gradients(train_ds, train_labels)

                    with tf.GradientTape(persistent=True) as train_tape:
                        logits = self.model(train_ds, training=True)
                        loss = tf.reduce_sum(
                            tf.losses.categorical_crossentropy(train_labels, logits, from_logits=True)
                        )

                    train_gradients = train_tape.gradient(loss, self.model.trainable_variables)

                    self.update_updated_model(train_gradients)

                    for k in range(5 - 1):
                        with tf.GradientTape(persistent=True) as train_tape:
                            logits = self.updated_model(train_ds, training=True)
                            loss = tf.reduce_sum(
                                tf.losses.categorical_crossentropy(train_labels, logits, from_logits=True)
                            )

                        train_gradients = train_tape.gradient(loss, self.updated_model.get_weights())
                        self.update_updated_model(train_gradients)

                    updated_model_logits = self.updated_model(val_ds, training=False)
                    val_loss = tf.reduce_sum(tf.losses.categorical_crossentropy(val_labels, updated_model_logits, from_logits=True))
                    # print(val_loss)
                
                # val_gradients = val_tape.gradient(val_loss, updated_model_logits)

                val_gradients = val_tape.gradient(val_loss, self.model.trainable_variables)
                # print(np.max(val_gradients.numpy()))
                self.optimizer.apply_gradients(zip(val_gradients, self.model.trainable_variables))
                # tasks_final_gradients.append(val_gradients)

            # final_gradients = average_gradients(tasks_final_gradients)
            # self.optimizer.apply_gradients(zip(final_gradients, self.model.trainable_variables))


if __name__ == '__main__':
    config = {
        'database': {
            'database_class': OmniglotDatabase,
            'random_seed': -1,
            'config': {
                'num_train_classes': 1200,
                'num_val_classes': 100,
                'train_dataset_kwargs': {
                    'n': 5, 'k': 5, 'meta_batch_size': 25
                },
                'val_dataset_kwargs': {
                    'n': 5, 'k': 5, 'meta_batch_size': 3
                },
                'test_dataset_kwargs': {
                    'n': 5, 'k': 5, 'meta_batch_size': 3
                },
            },
        },
        'model': {
            'class': SimpleModel,
            'init_kwargs': {'num_classes': 5},
        },
        'lr_inner_ml': 0.1,
        'num_steps_ml': 5,
    }

    maml = ModelAgnosticMetaLearningModel(config)
    maml.train(epochs=100)
