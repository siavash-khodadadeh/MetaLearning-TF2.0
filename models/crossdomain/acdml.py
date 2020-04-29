import os
import tensorflow as tf
import numpy as np

from models.maml.maml import ModelAgnosticMetaLearningModel
from databases import MiniImagenetDatabase, PlantDiseaseDatabase, ISICDatabase, AirplaneDatabase, CUBDatabase
from databases.data_bases import Database
from models.crossdomain.attention import MiniImagenetModel, AttentionModel, decompose_attention_model, assemble_model

from tqdm import tqdm
from typing import List
from decorators import name_repr
from utils import combine_first_two_axes, keep_keys_with_greater_than_equal_k_items

class AttentionCrossDomainMetaLearning(ModelAgnosticMetaLearningModel):
    def get_train_dataset(self):
        databases = [MiniImagenetDatabase(), AirplaneDatabase(), CUBDatabase()]

        dataset = self.get_cross_domain_meta_learning_dataset(
            databases=databases,
            n=self.n,
            k=self.k,
            k_validation=self.k_val_ml,
            meta_batch_size=self.meta_batch_size
        )
        return dataset

    def get_val_dataset(self):
        databases = [ISICDatabase()]

        val_dataset = self.get_cross_domain_meta_learning_dataset(
            databases=databases,
            n=self.n,
            k=self.k,
            k_validation=self.k_val_val,
            meta_batch_size=1
        )
        val_dataset = val_dataset.repeat(-1)
        val_dataset = val_dataset.take(self.number_of_tasks_val)
        setattr(val_dataset, 'steps_per_epoch', self.number_of_tasks_val)
        return val_dataset

    def get_test_dataset(self, seed=-1):
        databases = [ISICDatabase()]

        test_dataset = self.get_cross_domain_meta_learning_dataset(
            databases=databases,
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

    def get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, (84, 84))
            image = tf.cast(image, tf.float32)
            return image / 255.
        return parse_function

    def get_cross_domain_meta_learning_dataset(
            self,
            databases: List[Database],
            n: int,
            k: int,
            k_validation: int,
            meta_batch_size: int,
            one_hot_labels: bool = True,
            reshuffle_each_iteration: bool = True,
            seed: int = -1,
            dtype=tf.float32,  # The input dtype
    ) -> tf.data.Dataset:
        datasets = list()
        steps_per_epoch = 1000000
        for database in databases:
            dataset = self.get_supervised_meta_learning_dataset(
                database.train_folders,
                n,
                k,
                k_validation,
                meta_batch_size=2,
                one_hot_labels=one_hot_labels,
                reshuffle_each_iteration=reshuffle_each_iteration,
                seed=seed,
                dtype=tf.string,
                instance_parse_function=lambda x: x
            )
            steps_per_epoch = min(steps_per_epoch, dataset.steps_per_epoch)
            datasets.append(dataset)
        datasets = tuple(datasets)

        def choose_two_domains(*domains):
            tensors = []
            for domain in domains:
                (tr_ds, val_ds), (tr_labels, val_labels) = domain
                tensors.append(tr_ds)
                tensors.append(val_ds)
                tensors.append(tr_labels)
                tensors.append(val_labels)

            def f(*args):
#                 np.random.seed(42)
                indices = np.random.choice(range(len(datasets)), size=2, replace=False)
                tr_ds = args[indices[0] * 4][0, ...]
                tr_labels = args[indices[0] * 4 + 2][0, ...]
                tr_domain = args[indices[0] * 4][1, ...]
                val_ds = args[indices[1] * 4 + 1][0, ...]
                val_labels = args[indices[1] * 4 + 3][0, ...]
                val_domain = args[indices[1] * 4][1, ...]
                return tr_ds, tr_labels, tr_domain, val_ds, val_labels, val_domain

            return tf.py_function(f, inp=tensors, Tout=[tf.string, tf.float32, tf.string] * 2)

        # TODO handle the seed
        parallel_iterations = None

        def parse_function(
            tr_task_imgs_addresses,
            tr_task_labels,
            tr_domain_imgs_addresses,
            val_task_imgs_addresses,
            val_task_labels,
            val_domain_imgs_addresses
        ):
            def parse_batch_imgs(img_addresses, shape):
                img_addresses = tf.reshape(img_addresses, (-1,))
                imgs = tf.map_fn(
                    self.get_parse_function(),
                    img_addresses,
                    dtype=dtype,
                    parallel_iterations=parallel_iterations
                )
                return tf.reshape(imgs, shape)

            tr_task_imgs = parse_batch_imgs(tr_task_imgs_addresses, (n, k, 84, 84, 3))
            tr_dom_imgs = parse_batch_imgs(tr_domain_imgs_addresses, (n, k, 84, 84, 3))
            val_task_imgs = parse_batch_imgs(val_task_imgs_addresses, (n, k_validation, 84, 84, 3))
            val_dom_imgs = parse_batch_imgs(val_domain_imgs_addresses, (n, k, 84, 84, 3))

            return (tr_task_imgs, tr_dom_imgs, val_task_imgs, val_dom_imgs), (tr_task_labels, val_task_labels)

        dataset = tf.data.Dataset.zip(datasets)
        # TODO steps per epoch can be inferred from tf.data.experimental.cardinality(dataset)
        dataset = dataset.map(choose_two_domains)
        dataset = dataset.map(parse_function)

        # TODO fix this for different meta batch sizes. Check to see if it works with removing the following line.
        meta_batch_size = 1
        dataset = dataset.batch(batch_size=meta_batch_size, drop_remainder=False)
        steps_per_epoch = steps_per_epoch // meta_batch_size

        # import matplotlib.pyplot as plt
        #
        # for item in dataset:
        #     (tr_ds, tr_dom, val_ds, val_dom), (tr_labels, val_labels) = item
        #     print(tr_ds.shape)
        #     print(val_ds.shape)
        #     print(tr_dom.shape)
        #     print(val_dom.shape)
        #     print(tr_labels.shape)
        #     print(val_labels.shape)
        #
        #     for i in range(n):
        #         for j in range(k):
        #             plt.imshow(tr_ds[i, j, ...])
        #             plt.show()
        #     for i in range(n):
        #         for j in range(k):
        #             plt.imshow(tr_dom[i, j, ...])
        #             plt.show()
        #     for i in range(n):
        #         for j in range(k):
        #             plt.imshow(val_dom[i, j, ...])
        #             plt.show()
        #     for i in range(n):
        #         for j in range(k_validation):
        #             plt.imshow(val_ds[i, j, ...])
        #             plt.show()
        #     break

        setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset

    def initialize_network(self):
        model = self.network_cls(num_classes=self.n)
        model([tf.zeros(shape=(1, *self.database.input_shape)), tf.zeros(shape=(1, *self.database.input_shape))])
        return model

    def train(self, iterations=5):
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)
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
            for (train_ds, train_dom, val_ds, val_dom), (train_labels, val_labels) in self.train_dataset:
                train_acc, train_loss = self.meta_train_loop(train_ds, train_dom, val_ds, val_dom, train_labels, val_labels)
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
                        combine_first_two_axes(train_dom[0, ...]),
                        combine_first_two_axes(val_ds[0, ...]),
                        combine_first_two_axes(val_dom[0, ...]),
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

    def log_images(self, summary_writer, train_ds, train_dom, val_ds, val_dom, step):
        with tf.device('cpu:0'):
            with summary_writer.as_default():
                tf.summary.image(
                    'train_ds',
                    train_ds,
                    step=step,
                    max_outputs=self.n * (self.k + self.k_val_ml)
                )
                tf.summary.image(
                    'train_dom',
                    train_dom,
                    step=step,
                    max_outputs=self.n * (self.k + self.k_val_ml)
                )
                tf.summary.image(
                    'val_ds',
                    val_ds,
                    step=step,
                    max_outputs=self.n * (self.k + self.k_val_ml)
                )
                tf.summary.image(
                    'val_dom',
                    val_dom,
                    step=step,
                    max_outputs=self.n * (self.k + self.k_val_ml)
                )

    @tf.function
    def meta_train_loop(self, train_ds, train_dom, val_ds, val_dom, train_labels, val_labels):
        with tf.GradientTape(persistent=False) as outer_tape:
            tasks_final_losses = list()
            tasks_final_accs = list()

            for i in range(self.meta_batch_size):
                task_final_acc, task_final_loss = self.get_losses_of_tasks_batch(method='train')(
                    (train_ds[i, ...], train_dom[i, ...], val_ds[i, ...], val_dom[i, ...],
                     train_labels[i, ...], val_labels[i, ...])
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

    def get_losses_of_tasks_batch(self, method='train', **kwargs):
        if method == 'train':
            return self.get_losses_of_tasks_batch_maml()
        elif method == 'val':
            return self.get_losses_of_tasks_batch_eval(iterations=self.num_steps_validation, training=False)
        elif method == 'test':
            return self.get_losses_of_tasks_batch_eval(
                iterations=kwargs['iterations'],
                training=kwargs['use_val_batch_statistics']
            )

    def get_losses_of_tasks_batch_maml(self):
        # TODO check if tf.function results in any imporvement in speed since this causes a lot of warning.
        @tf.function
        def f(inputs):
            train_ds, train_dom, val_ds, val_dom, train_labels, val_labels = inputs
            train_ds = combine_first_two_axes(train_ds)
            train_dom = combine_first_two_axes(train_dom)
            val_ds = combine_first_two_axes(val_ds)
            val_dom = combine_first_two_axes(val_dom)

            updated_model = self.inner_train_loop(train_ds, train_dom, train_labels)
            # TODO test what happens when training=False
            updated_model_logits = updated_model([val_dom, val_ds], training=True)
            val_loss = self.outer_loss(val_labels, updated_model_logits)

            predicted_class_labels = self.predict_class_labels_from_logits(updated_model_logits)
            real_labels = self.convert_labels_to_real_labels(val_labels)

            val_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_class_labels, real_labels), tf.float32))

            return val_acc, val_loss

        return f

    def inner_train_loop(self, train_ds, train_dom, train_labels):
        num_iterations = self.num_steps_ml

        gradients = list()
        for variable in self.model.trainable_variables:
            gradients.append(tf.zeros_like(variable))

        self.create_meta_model(self.updated_models[0], self.model, gradients)

        for k in range(1, num_iterations + 1):
            with tf.GradientTape(persistent=False) as train_tape:
                train_tape.watch(self.updated_models[k - 1].meta_trainable_variables)
                logits = self.updated_models[k - 1]([train_dom, train_ds], training=True)
                loss = self.inner_loss(train_labels, logits)
            gradients = train_tape.gradient(loss, self.updated_models[k - 1].meta_trainable_variables)
            self.create_meta_model(self.updated_models[k], self.updated_models[k - 1], gradients)

        return self.updated_models[-1]

@name_repr('AssembledModel')
def get_assembled_model(num_classes, ind=7, architecture=MiniImagenetModel, input_shape=(84, 84, 3)):
    attention_model = AttentionModel()
    attention_model = decompose_attention_model(attention_model, input_shape)

    base_model = architecture(num_classes)
    base_input = tf.keras.Input(shape=input_shape, name='base_input')
    base_model(base_input)

    return assemble_model(attention_model, base_model, ind)

def run_acdml():
    acdml = AttentionCrossDomainMetaLearning(
        database=None,
        network_cls=get_assembled_model,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=1,
        meta_batch_size=1,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=15000,
        meta_learning_rate=0.001,
        report_validation_frequency=1000,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=True,
        experiment_name='acdml',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
    )

    acdml.train(iterations=60000)
    acdml.evaluate(100, seed=14)


if __name__ == '__main__':
    run_acdml()
