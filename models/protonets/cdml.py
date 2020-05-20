import os
import tensorflow as tf
import numpy as np

from models.protonets.proto_nets import PrototypicalNetworks
from networks.maml_umtra_networks import MiniImagenetModel
from databases import MiniImagenetDatabase, AirplaneDatabase, CUBDatabase, OmniglotDatabase, DTDDatabase, \
    FungiDatabase, VGGFlowerDatabase, PlantDiseaseDatabase, ISICDatabase, EuroSatDatabase, MSCOCODatabase, \
    ChestXRay8Database
from databases.data_bases import Database

from typing import List


class CombinedCrossDomainMetaLearning(PrototypicalNetworks):
    def get_train_dataset(self):
        databases = [
            MiniImagenetDatabase(),
            AirplaneDatabase(),
            CUBDatabase(),
            OmniglotDatabase(random_seed=47, num_train_classes=1200, num_val_classes=100),
            DTDDatabase(),
            FungiDatabase(),
            VGGFlowerDatabase()
        ]

        dataset = self.get_cross_domain_meta_learning_dataset(
            databases=databases,
            n=self.n,
            k=self.k,
            k_validation=self.k_val_ml,
            meta_batch_size=self.meta_batch_size
        )
        return dataset

    def get_parse_function(self):
        @tf.function
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, (84, 84))
            if tf.shape(image)[2] == 1:
                image = tf.squeeze(image, axis=2)
                image = tf.stack((image, image, image), axis=2)
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
                meta_batch_size=1,
                one_hot_labels=one_hot_labels,
                reshuffle_each_iteration=reshuffle_each_iteration,
                seed=seed,
                dtype=tf.string,
                instance_parse_function=lambda x: x
            )
            steps_per_epoch = min(steps_per_epoch, dataset.steps_per_epoch)
            datasets.append(dataset)
        datasets = tuple(datasets)

        def choose_one_domain(*domains):
            tensors = []
            for domain in domains:
                (tr_ds, val_ds), (tr_labels, val_labels) = domain
                tensors.append(tr_ds)
                tensors.append(val_ds)
                tensors.append(tr_labels)
                tensors.append(val_labels)

            def f(*args):
                index = np.random.choice(range(len(datasets)), size=1, replace=False)[0]
                tr_ds = args[index * 4]
                val_ds = args[index * 4 + 1]
                tr_labels = args[index * 4 + 2]
                val_labels = args[index * 4 + 3]
                return tr_ds, tr_labels, val_ds, val_labels

            return tf.py_function(f, inp=tensors, Tout=[tf.string, tf.float32] * 2)

        # TODO handle the seed
        parallel_iterations = None

        def parse_function(
            tr_task_imgs_addresses,
            tr_task_labels,
            val_task_imgs_addresses,
            val_task_labels,
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
            val_task_imgs = parse_batch_imgs(val_task_imgs_addresses, (n, k_validation, 84, 84, 3))
            tr_task_labels = tf.squeeze(tr_task_labels, axis=0)
            val_task_labels = tf.squeeze(val_task_labels, axis=0)

            return (tr_task_imgs, val_task_imgs), (tr_task_labels, val_task_labels)

        dataset = tf.data.Dataset.zip(datasets)
        # TODO steps per epoch can be inferred from tf.data.experimental.cardinality(dataset)
        dataset = dataset.map(choose_one_domain)
        dataset = dataset.map(parse_function)

        dataset = dataset.batch(batch_size=meta_batch_size, drop_remainder=True)
        steps_per_epoch = steps_per_epoch // meta_batch_size

        # import matplotlib.pyplot as plt
        # for item in dataset:
        #
        #     (tr_ds, val_ds), (tr_lbls, val_lbls) = item
        #     print(tr_ds.shape)
        #     print(val_ds.shape)
        #     print(tr_lbls.shape)
        #     print(val_lbls.shape)
        #     for i in range(n):
        #         for j in range(k):
        #             plt.imshow(tr_ds[i, j, ...])
        #             plt.show()
        #     for i in range(n):
        #         for j in range(k_validation):
        #             plt.imshow(val_ds[i, j, ...])
        #             plt.show()
        #     print(item)
        #     exit()

        setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset


def run_ccdml():
    test_database = EuroSatDatabase()
    ccdml = CombinedCrossDomainMetaLearning(
        database=test_database,
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
        report_validation_frequency=1000,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=True,
        experiment_name='cdml',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
    )

    ccdml.train(iterations=60000)
    ccdml.evaluate(50, seed=14)


if __name__ == '__main__':
    run_ccdml()
