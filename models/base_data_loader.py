from typing import Dict, List
import random

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from utils import keep_keys_with_greater_than_equal_k_items


class BaseDataLoader(object):
    def __init__(
            self,
            database,
            val_database,
            test_database,
            n,
            k_ml,
            k_val_ml,
            k_val,
            k_val_val,
            k_test,
            k_val_test,
            meta_batch_size,
            num_tasks_val,
            val_seed
    ):
        self.database = database
        self.val_database = val_database
        self.test_database = test_database
        self.n = n
        self.k_ml = k_ml
        self.k_val_ml = k_val_ml
        self.k_val = k_val
        self.k_val_val = k_val_val
        self.k_test = k_test
        self.k_val_test = k_val_test
        self.meta_batch_size = meta_batch_size
        self.num_tasks_val = num_tasks_val
        self.val_seed = val_seed

    def get_train_dataset(self):
        dataset = self.get_supervised_meta_learning_dataset(
            self.database.train_folders,
            n=self.n,
            k=self.k_ml,
            k_validation=self.k_val_ml,
            meta_batch_size=self.meta_batch_size
        )
        return dataset

    def get_val_dataset(self):
        val_dataset = self.get_supervised_meta_learning_dataset(
            self.val_database.val_folders,
            n=self.n,
            k=self.k_val,
            k_validation=self.k_val_val,
            meta_batch_size=1,
            seed=self.val_seed,
        )
        val_dataset = val_dataset.repeat(-1)
        val_dataset = val_dataset.take(self.num_tasks_val)

        return val_dataset

    def get_test_dataset(self, num_tasks, seed=-1):
        test_dataset = self.get_supervised_meta_learning_dataset(
            self.test_database.test_folders,
            n=self.n,
            k=self.k_test,
            k_validation=self.k_val_test,
            meta_batch_size=1,
            seed=seed
        )
        test_dataset = test_dataset.repeat(-1)
        test_dataset = test_dataset.take(num_tasks)

        return test_dataset

    def make_labels_dataset(self, n: int, k: int, k_validation: int, one_hot_labels: bool) -> tf.data.Dataset:
        """
            Creates a tf.data.Dataset which generates corresponding labels to meta-learning inputs.
            This method just creates this dataset for one task and repeats it. You can use zip to combine this dataset
            with your desired dataset.
            Note that the repeat is set to -1 so that the dataset will repeat itself. This will allow us to
            zip it with any other dataset and it will generate labels as much as needed.
            Also notice that this dataset is not batched into meta batch size, so this will just generate labels for one
            task.
        """
        tr_labels_ds = tf.data.Dataset.from_tensor_slices(np.expand_dims(np.repeat(np.arange(n), k), 0))
        val_labels_ds = tf.data.Dataset.from_tensor_slices(np.expand_dims(np.repeat(np.arange(n), k_validation), 0))

        if one_hot_labels:
            tr_labels_ds = tr_labels_ds.map(lambda example: tf.one_hot(example, depth=n))
            val_labels_ds = val_labels_ds.map(lambda example: tf.one_hot(example, depth=n))

        labels_dataset = tf.data.Dataset.zip((tr_labels_ds, val_labels_ds))
        labels_dataset = labels_dataset.repeat(-1)

        return labels_dataset

    def get_unsupervised_dataset(
        self,
        folders: Dict[str, List[str]],
        n: int,
        meta_batch_size: int,
        one_hot_labels: bool = True,
        reshuffle_each_iteration: bool = True,
        seed: int = -1,
        instance_parse_function=None
    ):
        k = 1
        """This function generates a dataset that uses the same image for both training and validation"""
        if instance_parse_function is None:
            instance_parse_function = self.get_parse_function()

        # TODO handle seed
        if seed != -1:
            np.random.seed(seed)

        train_indices = [i // k + i % k * n for i in range(n * k)]
        val_indices = [n * k + i // k + i % k * n for i in range(n * k)]

        def generate_same_samples(instances):
            new_instances = list()
            for i in range(2 * k - 1):
                new_instance = instances + tf.zeros_like(instances)
                new_instances.append(new_instance)

            new_instances = tf.concat(new_instances, axis=0)
            all_instances = tf.concat((instances, new_instances), axis=0)
            train_instances = tf.gather(all_instances, train_indices, axis=0)
            val_instances = tf.gather(all_instances, val_indices, axis=0)

            return (
                tf.reshape(train_instances, (n, k, 84, 84, 3)),
                tf.reshape(val_instances, (n, k, 84, 84, 3)),
            )

        instances = list()

        for folder, folder_instances in folders.items():
            instances.extend(folder_instances)

        random.shuffle(instances)

        dataset = tf.data.Dataset.from_tensor_slices(instances)
        dataset = dataset.map(instance_parse_function)
        # dataset = dataset.shuffle(buffer_size=len(instances))
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=reshuffle_each_iteration)

        dataset = dataset.batch(n, drop_remainder=True)
        dataset = dataset.map(generate_same_samples)

        labels_dataset = self.make_labels_dataset(n, k, k, one_hot_labels=one_hot_labels)

        dataset = tf.data.Dataset.zip((dataset, labels_dataset))
        dataset = dataset.batch(meta_batch_size)

        setattr(dataset, 'steps_per_epoch', tf.data.experimental.cardinality(dataset))
        return dataset

    def get_umtra_dataset(
        self,
        folders: Dict[str, List[str]],
        n: int,
        k: int,
        k_validation: int,
        meta_batch_size: int,
        one_hot_labels: bool = True,
        reshuffle_each_iteration: bool = True,
        seed: int = -1,
        instance_parse_function=None
    ):
        if instance_parse_function is None:
            instance_parse_function = self.get_parse_function()

        # TODO handle seed
        if seed != -1:
            np.random.seed(seed)

        handle = 'https://tfhub.dev/google/image_augmentation/nas_imagenet/1'
        hub_model = hub.load(handle).signatures['from_decoded_images']

        train_indices = [i // k + i % k * n for i in range(n * k)]
        val_indices = [
            n * k + i // k_validation + i % k_validation * n for i in range(n * k_validation)
        ]

        def generate_new_samples_with_auto_augment(instances):
            new_instances = list()
            for i in range(k + k_validation - 1):
                new_instance = hub_model(
                    images=instances,
                    image_size=tf.constant([84, 84]),
                    augmentation=tf.constant(True)
                )['default']
                new_instances.append(new_instance)

            new_instances = tf.concat(new_instances, axis=0)
            all_instances = tf.concat((instances, new_instances), axis=0)
            train_instances = tf.gather(all_instances, train_indices, axis=0)
            val_instances = tf.gather(all_instances, val_indices, axis=0)

            return (
                tf.reshape(train_instances, (n, k, *train_instances.shape[1:])),
                tf.reshape(val_instances, (n, k_validation, *val_instances.shape[1:])),
            )

        instances = list()

        for folder, folder_instances in folders.items():
            instances.extend(folder_instances)

        random.shuffle(instances)

        dataset = tf.data.Dataset.from_tensor_slices(instances)
        dataset = dataset.map(instance_parse_function)
        # dataset = dataset.shuffle(buffer_size=len(instances))
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = dataset.batch(n, drop_remainder=True)

        dataset = dataset.map(generate_new_samples_with_auto_augment)
        # dataset = dataset.map(generate_new_samples_with_augmentation)

        labels_dataset = self.make_labels_dataset(n, k, k_validation, one_hot_labels=one_hot_labels)

        dataset = tf.data.Dataset.zip((dataset, labels_dataset))
        dataset = dataset.batch(meta_batch_size)

        setattr(dataset, 'steps_per_epoch', tf.data.experimental.cardinality(dataset))
        return dataset

    def get_supervised_meta_learning_dataset(
            self,
            folders: Dict[str, List[str]],
            n: int,
            k: int,
            k_validation: int,
            meta_batch_size: int,
            one_hot_labels: bool = True,
            reshuffle_each_iteration: bool = True,
            seed: int = -1,
            dtype=tf.float32,  # The input dtype
            instance_parse_function=None
    ) -> tf.data.Dataset:
        """
            Folders are dictionary
            If it is a dictionary then each item is the class name and the corresponding values are the file addresses
            of images of that class.
        """
        if instance_parse_function is None:
            instance_parse_function = self.get_parse_function()

        if seed != -1:
            np.random.seed(seed)

        def _get_instances(class_dir_address):
            def get_instances(class_dir_address):
                class_dir_address = class_dir_address.numpy().decode('utf-8')
                instance_names = folders[class_dir_address]
                instances = np.random.choice(instance_names, size=k + k_validation, replace=False)
                return instances[:k], instances[k:k + k_validation]

            return tf.py_function(get_instances, inp=[class_dir_address], Tout=[tf.string, tf.string])

        if seed != -1:
            parallel_iterations = 1
        else:
            parallel_iterations = None

        def parse_function(tr_imgs_addresses, val_imgs_addresses):
            tr_imgs = tf.map_fn(
                instance_parse_function,
                tr_imgs_addresses,
                dtype=dtype,
                parallel_iterations=parallel_iterations
            )
            val_imgs = tf.map_fn(
                instance_parse_function,
                val_imgs_addresses,
                dtype=dtype,
                parallel_iterations=parallel_iterations
            )

            return tf.stack(tr_imgs), tf.stack(val_imgs)

        keep_keys_with_greater_than_equal_k_items(folders, k + k_validation)

        dataset = tf.data.Dataset.from_tensor_slices(sorted(list(folders.keys())))
        if seed != -1:
            dataset = dataset.shuffle(
                buffer_size=len(folders.keys()),
                reshuffle_each_iteration=reshuffle_each_iteration,
                seed=seed
            )
            # When using a seed the map should be done in the same order so no parallel execution
            dataset = dataset.map(_get_instances, num_parallel_calls=1)
        else:
            dataset = dataset.shuffle(
                buffer_size=len(folders.keys()),
                reshuffle_each_iteration=reshuffle_each_iteration
            )
            dataset = dataset.map(_get_instances, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(n, drop_remainder=True)

        labels_dataset = self.make_labels_dataset(n, k, k_validation, one_hot_labels=one_hot_labels)

        dataset = tf.data.Dataset.zip((dataset, labels_dataset))

        steps_per_epoch = tf.data.experimental.cardinality(dataset)
        if steps_per_epoch == 0:
            dataset = dataset.repeat(-1).take(meta_batch_size).batch(meta_batch_size)
        else:
            dataset = dataset.batch(meta_batch_size, drop_remainder=True)

        return dataset

    def get_parse_function(self):
        """Returns a function which get an example_address
         and processes it such that it will be input to the network."""
        return self.database._get_parse_function()
