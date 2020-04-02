import os

import tensorflow as tf
import numpy as np

from tf_datasets import OmniglotDatabase, MiniImagenetDatabase, CelebADatabase, LFWDatabase, VGGFace2Database, \
    ISICDatabase, EuroSatDatabase, PlantDiseaseDatabase, ChestXRay8Database
from networks.maml_umtra_networks import SimpleModel, MiniImagenetModel, VGG19Model, VGGSmallModel
from models.base_model import BaseModel
from utils import combine_first_two_axes, average_gradients
import settings


# TODO
# Visualize all validation tasks just once.

# Fix tests and add tests for UMTRA.

# Test visualization could be done on all images or some of them

# Make it possible to train on multiple GPUs (Not very necessary now), but we have to make it fast with tf.function.


class ModelAgnosticMetaLearningModel(BaseModel):
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
        num_steps_ml,
        lr_inner_ml,
        num_steps_validation,
        save_after_iterations,
        meta_learning_rate,
        report_validation_frequency,
        log_train_images_after_iteration,  # Set to -1 if you do not want to log train images.
        number_of_tasks_val=-1,  # Make sure the validation pick this many tasks.
        number_of_tasks_test=-1,  # Make sure the validation pick this many tasks.
        val_seed=-1,  # The seed for validation dataset. -1 means change the samples for each report.
        clip_gradients=False,
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

        self.num_steps_ml = num_steps_ml
        self.num_steps_validation = num_steps_validation
        self.lr_inner_ml = lr_inner_ml
        self.clip_gradients = clip_gradients
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=meta_learning_rate)
        self.network_cls = network_cls
        self.model = self.initialize_network()
        self.updated_models = list()

        for _ in range(self.num_steps_ml + 1):
            updated_model = self.initialize_network()
            self.updated_models.append(updated_model)

    def initialize_network(self):
        model = self.network_cls(num_classes=self.n)
        model(tf.zeros(shape=(self.n * self.k, *self.database.input_shape)))
        return model

    def get_config_str(self):
        config_str = f'model-{self.network_cls.name}_' \
               f'mbs-{self.meta_batch_size}_' \
               f'n-{self.n}_' \
               f'k-{self.k}_' \
               f'kvalml-{self.k_val_ml}' \
               f'stp-{self.num_steps_ml}'
        return config_str

    def post_process_outer_gradients(self, outer_gradients):
        if self.clip_gradients:
            outer_gradients = [tf.clip_by_value(grad, -10, 10) for grad in outer_gradients]
        return outer_gradients

    def create_meta_model(self, updated_model, model, gradients):
        k = 0
        variables = list()

        for i in range(len(model.layers)):
            if (isinstance(model.layers[i], tf.keras.layers.Conv2D) or
                    isinstance(model.layers[i], tf.keras.layers.Dense)):
                updated_model.layers[i].kernel = model.layers[i].kernel - self.lr_inner_ml * gradients[k]
                k += 1
                variables.append(updated_model.layers[i].kernel)

                updated_model.layers[i].bias = model.layers[i].bias - self.lr_inner_ml * gradients[k]
                k += 1
                variables.append(updated_model.layers[i].bias)

            elif isinstance(model.layers[i], tf.keras.layers.BatchNormalization):
                if hasattr(model.layers[i], 'moving_mean') and model.layers[i].moving_mean is not None:
                    # updated_model.layers[i].moving_mean.assign(model.layers[i].moving_mean)
                    updated_model.layers[i].moving_mean = model.layers[i].moving_mean
                if hasattr(model.layers[i], 'moving_variance') and model.layers[i].moving_variance is not None:
                    # updated_model.layers[i].moving_variance.assign(model.layers[i].moving_variance)
                    updated_model.layers[i].moving_variance = model.layers[i].moving_variance
                if hasattr(model.layers[i], 'gamma') and model.layers[i].gamma is not None:
                    updated_model.layers[i].gamma = model.layers[i].gamma - self.lr_inner_ml * gradients[k]
                    k += 1
                    variables.append(updated_model.layers[i].gamma)
                if hasattr(model.layers[i], 'beta') and model.layers[i].beta is not None:
                    updated_model.layers[i].beta = model.layers[i].beta - self.lr_inner_ml * gradients[k]
                    k += 1
                    variables.append(updated_model.layers[i].beta)

            elif isinstance(model.layers[i], tf.keras.layers.LayerNormalization):
                if hasattr(model.layers[i], 'gamma') and model.layers[i].gamma is not None:
                    updated_model.layers[i].gamma = model.layers[i].gamma - self.lr_inner_ml * gradients[k]
                    k += 1
                    variables.append(updated_model.layers[i].gamma)
                if hasattr(model.layers[i], 'beta') and model.layers[i].beta is not None:
                    updated_model.layers[i].beta = model.layers[i].beta - self.lr_inner_ml * gradients[k]
                    k += 1
                    variables.append(updated_model.layers[i].beta)

        setattr(updated_model, 'meta_trainable_variables', variables)

    def inner_loss(self, train_labels, logits):
        loss = tf.reduce_mean(
            tf.losses.categorical_crossentropy(train_labels, logits, from_logits=True)
        )
        return loss

    def inner_train_loop(self, train_ds, train_labels, num_iterations=-1):
        if num_iterations == -1:
            num_iterations = self.num_steps_ml

            gradients = list()
            for variable in self.model.trainable_variables:
                gradients.append(tf.zeros_like(variable))

            self.create_meta_model(self.updated_models[0], self.model, gradients)

            for k in range(1, num_iterations + 1):
                with tf.GradientTape(persistent=False) as train_tape:
                    train_tape.watch(self.updated_models[k - 1].meta_trainable_variables)
                    logits = self.updated_models[k - 1](train_ds, training=True)
                    loss = self.inner_loss(train_labels, logits)
                gradients = train_tape.gradient(loss, self.updated_models[k - 1].meta_trainable_variables)
                self.create_meta_model(self.updated_models[k], self.updated_models[k - 1], gradients)

            return self.updated_models[-1]

        else:
            gradients = list()
            for variable in self.model.trainable_variables:
                gradients.append(tf.zeros_like(variable))

            copy_model = self.updated_models[0]
            self.create_meta_model(self.updated_models[0], self.model, gradients)

            for k in range(num_iterations):
                with tf.GradientTape(persistent=False) as train_tape:
                    train_tape.watch(copy_model.meta_trainable_variables)
                    logits = copy_model(train_ds, training=True)
                    loss = self.inner_loss(train_labels, logits)
                gradients = train_tape.gradient(loss, copy_model.meta_trainable_variables)
                self.create_meta_model(copy_model, copy_model, gradients)

            return copy_model

    def update_loss_and_accuracy(self, logits, labels, loss_metric, accuracy_metric):
        val_loss = self.outer_loss(labels, logits)
        loss_metric.update_state(val_loss)

        predicted_class_labels = self.predict_class_labels_from_logits(logits)
        real_labels = self.convert_labels_to_real_labels(labels)

        # print(predicted_class_labels)

        accuracy_metric.update_state(
            real_labels,
            predicted_class_labels
        )
        val_acc = tf.math.reduce_mean(tf.cast(tf.math.equal(real_labels, predicted_class_labels), tf.float32))

        return val_acc, val_loss

    def outer_loss(self, labels, logits):
        loss = tf.reduce_mean(
            tf.losses.categorical_crossentropy(labels, logits, from_logits=True)
        )
        return loss

    @tf.function
    def get_losses_of_tasks_batch(self, inputs, method='train'):
        train_ds, val_ds, train_labels, val_labels, = inputs
        train_ds = combine_first_two_axes(train_ds)
        val_ds = combine_first_two_axes(val_ds)

        updated_model = self.inner_train_loop(train_ds, train_labels, -1)
        updated_model_logits = updated_model(val_ds, training=True)
        val_loss = self.outer_loss(val_labels, updated_model_logits)

        predicted_class_labels = self.predict_class_labels_from_logits(updated_model_logits)
        real_labels = self.convert_labels_to_real_labels(val_labels)

        val_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_class_labels, real_labels), tf.float32))

        return val_loss, val_acc

    def get_losses_of_tasks_batch_evaluation(self, iterations, use_val_batch_statistics=True):
        @tf.function
        def f(inputs):
            train_ds, val_ds, train_labels, val_labels = inputs
            train_ds = combine_first_two_axes(train_ds)
            val_ds = combine_first_two_axes(val_ds)

            updated_model = self.inner_train_loop(train_ds, train_labels, iterations)
            updated_model_logits = updated_model(val_ds, training=use_val_batch_statistics)
            val_loss = self.outer_loss(val_labels, updated_model_logits)

            predicted_class_labels = self.predict_class_labels_from_logits(updated_model_logits)
            real_labels = self.convert_labels_to_real_labels(val_labels)

            val_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_class_labels, real_labels), tf.float32))

            return val_acc, val_loss

        return f


    def evaluate(self, iterations, iterations_to_load_from=None, seed=-1, use_val_batch_statistics=True):
        """If you set use val batch statistics to true, then the batch information from all the test samples will be
        used for batch normalization layers (like MAML experiments), otherwise batch normalization layers use the
        average and variance which they learned during the updates."""
        self.test_dataset = self.get_test_dataset(seed=seed)
        self.load_model(iterations=iterations_to_load_from)

        accs = list()
        losses = list()
        losses_func = self.get_losses_of_tasks_batch_evaluation(
            iterations,
            use_val_batch_statistics=use_val_batch_statistics
        )
        counter = 0
        for (train_ds, val_ds), (train_labels, val_labels) in self.test_dataset:
            remainder_num = self.number_of_tasks_test // 20
            if remainder_num == 0:
                remainder_num = 1
            if counter % remainder_num == 0:
                print(f'{counter} / {self.number_of_tasks_test} are evaluated.')

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

    def report_validation_loss_and_accuracy(self, epoch_count):
        self.val_loss_metric.reset_states()
        self.val_accuracy_metric.reset_states()

        val_counter = 0
        loss_func = self.get_losses_of_tasks_batch_evaluation(self.num_steps_validation, use_val_batch_statistics=False)
        for (train_ds, val_ds), (train_labels, val_labels) in self.get_val_dataset():
            val_counter += 1
            # TODO fix validation logging
            # if val_counter % 5 == 0:
            #     step = epoch_count * self.val_dataset.steps_per_epoch + val_counter
            #     # pick the first task in meta batch
            #     log_train_ds = combine_first_two_axes(train_ds[0, ...])
            #     log_val_ds = combine_first_two_axes(val_ds[0, ...])
            #     self.log_images(self.val_summary_writer, log_train_ds, log_val_ds, step)

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

    def convert_labels_to_real_labels(self, labels):
        return tf.argmax(labels, axis=-1)

    def predict_class_labels_from_logits(self, logits):
        return tf.argmax(logits, axis=-1)


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
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=1,
        meta_batch_size=32,
        num_steps_ml=1,
        lr_inner_ml=0.4,
        num_steps_validation=1,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=50,
        log_train_images_after_iteration=200,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=False,
        experiment_name='omniglot'
    )

    # maml.train(iterations=5000)
    maml.evaluate(iterations=50)


def run_mini_imagenet():
    mini_imagenet_database = MiniImagenetDatabase()

    maml = ModelAgnosticMetaLearningModel(
        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=1,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=15000,
        meta_learning_rate=0.001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=10,
        clip_gradients=True,
        experiment_name='mini_imagenet_with_batch_norm_exp2'
    )

    maml.train(iterations=60010)
    maml.evaluate(50, seed=42, use_val_batch_statistics=True)
    maml.evaluate(50, seed=42, use_val_batch_statistics=False)  # TODO add momentum=0.0 to evaluate


def run_celeba():
    celeba_database = VGGFace2Database(input_shape=(224, 224, 3))
    # celeba_database = CelebADatabase(input_shape=(224, 224, 3))
    # celeba_database = LFWDatabase(input_shape=(224, 224, 3))
    # celeba_database = CelebADatabase(input_shape=(84, 84, 3))
    maml = ModelAgnosticMetaLearningModel(
        database=celeba_database,
        network_cls=VGGSmallModel,
        # network_cls=MiniImagenetModel,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=1,
        meta_batch_size=4,
        num_steps_ml=1,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=5000,
        meta_learning_rate=0.0001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=10,
        clip_gradients=True,
        experiment_name='vgg_face2_conv128_mlr_0.0001'
    )

    # 260000 iteration with meta learning rate 0.0001
    # from there start with 0.00001
    # from 310000 start with 0.000005
    # maml.train(iterations=500000)
    maml.evaluate(50, seed=42)


def run_isic():
    # isic_database = ISICDatabase()
    # eurosat_database = EuroSatDatabase()
    # plant_disease_database = PlantDiseaseDatabase()
    chestx_ray_database = ChestXRay8Database()
    maml = ModelAgnosticMetaLearningModel(
        # database=isic_database,
        # database=eurosat_database,
        # database=plant_disease_database,
        database=chestx_ray_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=5,
        meta_batch_size=1,
        num_steps_ml=1,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=5000,
        meta_learning_rate=0.001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=True,
        experiment_name='chestx'
    )

    # maml.train(iterations=60000)
    maml.evaluate(50, seed=42)


if __name__ == '__main__':
    # tf.config.set_visible_devices([], 'GPU')
    # run_omniglot()
    run_mini_imagenet()
    # run_celeba()
    # run_isic()
