import os
import unittest
from unittest.mock import patch
from collections import deque

import tensorflow as tf

from databases import OmniglotDatabase, MiniImagenetDatabase, CelebADatabase, LFWDatabase, EuroSatDatabase, \
    ISICDatabase, ChestXRay8Database
from utils import combine_first_two_axes


class TestDatabases(unittest.TestCase):
    def setUp(self):
        def parse_function(example_address):
            return example_address

        self.parse_function = parse_function

        self.omniglot_database = OmniglotDatabase(random_seed=-1, num_train_classes=1200, num_val_classes=100)
        self.mini_imagenet_database = MiniImagenetDatabase()
        self.celeba_database = CelebADatabase()
        self.lfw_database = LFWDatabase()
        self.euro_sat_database = EuroSatDatabase()
        self.isic_database = ISICDatabase()
        self.chest_x_ray_database = ChestXRay8Database()

    @property
    def databases(self):
        return (
            self.omniglot_database,
            self.mini_imagenet_database,
            self.celeba_database,
            self.lfw_database,
            self.euro_sat_database,
            self.isic_database,
            self.chest_x_ray_database,
        )

    def test_train_val_test_folders_are_separate(self):
        for database in (self.omniglot_database, self.mini_imagenet_database, self.celeba_database):
            train_folders = set(database.train_folders)
            val_folders = set(database.val_folders)
            test_folders = set(database.test_folders)

            for folder in train_folders:
                self.assertNotIn(folder, val_folders)
                self.assertNotIn(folder, test_folders)

            for folder in val_folders:
                self.assertNotIn(folder, train_folders)
                self.assertNotIn(folder, test_folders)
            
            for folder in test_folders:
                self.assertNotIn(folder, train_folders)
                self.assertNotIn(folder, val_folders)

    def test_train_val_test_folders_are_comprehensive(self):
        self.assertEqual(
            1623,
            len(
                list(self.omniglot_database.train_folders.keys()) +
                list(self.omniglot_database.val_folders.keys()) +
                list(self.omniglot_database.test_folders.keys())
            )
        )
        self.assertEqual(
            100,
            len(
                list(self.mini_imagenet_database.train_folders) +
                list(self.mini_imagenet_database.val_folders) +
                list(self.mini_imagenet_database.test_folders)
            )
        )
        self.assertEqual(
            10177,
            len(
                list(self.celeba_database.train_folders) +
                list(self.celeba_database.val_folders) +
                list(self.celeba_database.test_folders)
            )
        )

    @patch('tf_datasets.OmniglotDatabase._get_parse_function')
    def test_covering_all_classes_in_one_epoch(self, mocked_parse_function):
        # Make a new database so that the number of classes are dividable by the number of meta batches * n.
        mocked_parse_function.return_value = self.parse_function
        ds = self.omniglot_database.get_supervised_meta_learning_dataset(
            self.omniglot_database.train_folders,
            n=5,
            k=5,
            k_validation=5,
            meta_batch_size=4,
            dtype=tf.string
        )

        # Check for covering all classes and no duplicates
        classes = set()
        for (x, x_val), (y, y_val) in ds:
            train_ds = combine_first_two_axes(x)
            for i in range(train_ds.shape[0]):
                class_instances = train_ds[i, ...]
                class_instance_address = class_instances[0].numpy().decode('utf-8')
                class_address = os.path.split(class_instance_address)[0]
                self.assertNotIn(class_address, classes)
                classes.add(class_address)

        self.assertSetEqual(classes, set(self.omniglot_database.train_folders.keys()))

    @patch('tf_datasets.OmniglotDatabase._get_parse_function')
    def test_covering_all_classes_in_subsequent_epochs(self, mocked_parse_function):
        # This test might not pass because of the random selection of classes at the beginning of the class, but the
        # chances are low.  Specially if we increase the number of epochs, the chance of not covering classes will
        # decrease. Make a new database so that the number of classes are dividable by the number of meta batches * n.
        # This test should always fail with num_epochs = 1
        num_epochs = 3
        mocked_parse_function.return_value = self.parse_function
        ds = self.omniglot_database.get_supervised_meta_learning_dataset(
            self.omniglot_database.train_folders,
            n=7,
            k=5,
            k_validation=5,
            meta_batch_size=4,
            seed=42,
            dtype=tf.string
        )
        # Check for covering all classes
        classes = set()
        for epoch in range(num_epochs):
            for (x, x_val), (y, y_val) in ds:
                train_ds = combine_first_two_axes(x)
                for i in range(train_ds.shape[0]):
                    class_instances = train_ds[i, ...]
                    class_instance_address = class_instances[0].numpy().decode('utf-8')
                    class_address = os.path.split(class_instance_address)[0]
                    classes.add(class_address)

        self.assertSetEqual(classes, set(self.omniglot_database.train_folders.keys()))

    @patch('tf_datasets.OmniglotDatabase._get_parse_function')
    def test_labels_are_correct_in_train_and_val_for_every_task(self, mocked_parse_function):
        mocked_parse_function.return_value = self.parse_function
        n = 7
        k = 5
        k_validation = 6
        ds = self.omniglot_database.get_supervised_meta_learning_dataset(
            self.omniglot_database.train_folders,
            n=n,
            k=k,
            k_validation=k_validation,
            meta_batch_size=4,
            one_hot_labels=False,
            dtype=tf.string
        )

        for epoch in range(2):
            for (x, x_val), (y, y_val) in ds:
                for meta_batch_index in range(x.shape[0]):
                    train_ds = x[meta_batch_index, ...]
                    val_ds = x_val[meta_batch_index, ...]
                    train_labels = y[meta_batch_index, ...]
                    val_labels = y_val[meta_batch_index, ...]

                    class_label_dict = dict()

                    for class_index in range(n):
                        for instance_index in range(k):
                            instance_name = train_ds[class_index, instance_index, ...]
                            label = train_labels[class_index * k + instance_index, ...].numpy()

                            class_name = os.path.split(instance_name.numpy().decode('utf-8'))[0]
                            if class_name in class_label_dict:
                                self.assertEqual(class_label_dict[class_name], label)
                            else:
                                class_label_dict[class_name] = label

                    for class_index in range(n):
                        for instance_index in range(k_validation):
                            instance_name = val_ds[class_index, instance_index, ...]
                            label = val_labels[class_index * k_validation + instance_index, ...].numpy()
                            class_name = os.path.split(instance_name.numpy().decode('utf-8'))[0]
                            self.assertIn(class_name, class_label_dict)
                            self.assertEqual(class_label_dict[class_name], label)

    @patch('tf_datasets.OmniglotDatabase._get_parse_function')
    def test_train_and_val_have_different_samples_in_every_task(self, mocked_parse_function):
        mocked_parse_function.return_value = self.parse_function
        n = 6
        k = 4
        k_validation = 5
        ds = self.omniglot_database.get_supervised_meta_learning_dataset(
            self.omniglot_database.train_folders,
            n=n,
            k=k,
            k_validation=k_validation,
            meta_batch_size=4,
            one_hot_labels=False,
            dtype=tf.string
        )

        for epoch in range(4):
            for (x, x_val), (y, y_val) in ds:
                for meta_batch_index in range(x.shape[0]):
                    train_ds = x[meta_batch_index, ...]
                    val_ds = x_val[meta_batch_index, ...]

                    class_instances_dict = dict()

                    for class_index in range(n):
                        for instance_index in range(k):
                            instance_name = train_ds[class_index, instance_index, ...]

                            class_name, instance_name = os.path.split(instance_name.numpy().decode('utf-8'))
                            if class_name not in class_instances_dict:
                                class_instances_dict[class_name] = set()

                            class_instances_dict[class_name].add(instance_name)

                    for class_index in range(n):
                        for instance_index in range(k_validation):
                            instance_name = val_ds[class_index, instance_index, ...]
                            class_name, instance_name = os.path.split(instance_name.numpy().decode('utf-8'))
                            self.assertIn(class_name, class_instances_dict)
                            self.assertNotIn(instance_name, class_instances_dict[class_name])

    @patch('tf_datasets.OmniglotDatabase._get_parse_function')
    def test_no_two_class_in_the_same_task(self, mocked_parse_function):
        mocked_parse_function.return_value = self.parse_function
        n = 6
        k = 4
        k_validation = 5
        ds = self.omniglot_database.get_supervised_meta_learning_dataset(
            self.omniglot_database.train_folders,
            n=n,
            k=k,
            k_validation=k_validation,
            meta_batch_size=4,
            one_hot_labels=False,
            dtype=tf.string
        )

        for epoch in range(4):
            for (x, x_val), (y, y_val) in ds:
                for base_data, k_value in zip((x, x_val), (k, k_validation)):
                    for meta_batch_index in range(x.shape[0]):
                        train_ds = base_data[meta_batch_index, ...]
                        classes = dict()

                        for class_index in range(n):
                            for instance_index in range(k_value):
                                instance_name = train_ds[class_index, instance_index, ...]
                                class_name = os.path.split(instance_name.numpy().decode('utf-8'))[0]
                                classes[class_name] = classes.get(class_name, 0) + 1

                        for class_name, num_class_instances in classes.items():
                            self.assertEqual(k_value, num_class_instances)

    @patch('tf_datasets.OmniglotDatabase._get_parse_function')
    def test_different_instances_are_selected_from_each_class_for_train_and_val_each_time(self, mocked_parse_function):
        #  Random seed is selected such that instances are not selected the same for the whole epoch.
        #  This test might fail due to change in random or behaviour of selecting the samples and it might not mean that the code
        #  does not work properly. Maybe from one task in two different times the same train and val data will be selected
        mocked_parse_function.return_value = self.parse_function
        n = 6
        k = 4
        ds = self.omniglot_database.get_supervised_meta_learning_dataset(
            self.omniglot_database.train_folders,
            n=n,
            k=k,
            k_validation=8,
            meta_batch_size=4,
            one_hot_labels=False,
            dtype=tf.string
        )

        class_instances = dict()
        class_instances[0] = dict()
        class_instances[1] = dict()

        for epoch in range(2):
            for (x, x_val), (y, y_val) in ds:
                for meta_batch_index in range(x.shape[0]):
                    train_ds = x[meta_batch_index, ...]

                    for class_index in range(n):
                        for instance_index in range(k):
                            instance_address = train_ds[class_index, instance_index, ...]
                            class_name, instance_name = os.path.split(instance_address.numpy().decode('utf-8'))
                            if class_name not in class_instances[epoch]:
                                class_instances[epoch][class_name] = set()
                            class_instances[epoch][class_name].add(instance_name)

        first_epoch_class_instances = class_instances[0]
        second_epoch_class_instances = class_instances[1]

        for class_name in first_epoch_class_instances.keys():
            self.assertIn(class_name, second_epoch_class_instances)
            self.assertNotEqual(0, first_epoch_class_instances[class_name].difference(second_epoch_class_instances[class_name]))

    @patch('tf_datasets.MiniImagenetDatabase._get_parse_function')
    def test_1000_tasks_are_completely_different(self, mocked_parse_function):
        # This test should run on MAML get validation dataset.
        # For now, I just copied the code from there here to make sure that everything works fine.
        # However, if that code changes, this test does not have any value.
        # TODO Make this test work with maml.get_validation_dataset and get_test_dataset functions.
        mocked_parse_function.return_value = self.parse_function

        def get_val_dataset():
            val_dataset = self.mini_imagenet_database.get_supervised_meta_learning_dataset(
                self.mini_imagenet_database.val_folders,
                n=6,
                k=4,
                k_validation=3,
                meta_batch_size=1,
                seed=42,
                dtype=tf.string
            )
            val_dataset = val_dataset.repeat(-1)
            val_dataset = val_dataset.take(1000)
            return val_dataset

        counter = 0
        xs = set()
        x_vals = set()

        xs_queue = deque()
        x_vals_queue = deque()

        for (x, x_val), (y, y_val) in get_val_dataset():
            x_string = ','.join(list(map(str, x.numpy().reshape(-1))))
            # print(x_string)
            self.assertNotIn(x_string, xs)
            xs.add(x_string)

            x_val_string = ','.join(list(map(str, x_val.numpy().reshape(-1))))
            self.assertNotIn(x_val_string, x_vals)
            x_vals.add(x_val_string)

            xs_queue.append(x_string)
            x_vals_queue.append(x_val_string)
            counter += 1

        self.assertEqual(counter, 1000)
        for (x, x_val), (y, y_val) in get_val_dataset():
            x_string = ','.join(list(map(str, x.numpy().reshape(-1))))
            self.assertEqual(xs_queue.popleft(), x_string)

            x_val_string = ','.join(list(map(str, x_val.numpy().reshape(-1))))
            self.assertEqual(x_vals_queue.popleft(), x_val_string)
