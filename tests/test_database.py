import os
import unittest
import copy
from unittest.mock import patch

import tensorflow as tf

from tf_datasets import OmniglotDatabase


class TestOmniglotDatabase(unittest.TestCase):
    def setUp(self):
        def parse_function(example_address):
            return example_address

        self.parse_function = parse_function
        self.configs = [
            {
                'num_train_classes': 1200,
                'num_val_classes': 100,
                'train_dataset_kwargs': {
                    'n': 6, 'k': 4, 'meta_batch_size': 5
                },
                'val_dataset_kwargs': {
                    'n': 6, 'k': 4, 'meta_batch_size': 3
                },
                'test_dataset_kwargs': {
                    'n': 6, 'k': 4, 'meta_batch_size': 3
                },
            },
            {
                'num_train_classes': 1200,
                'num_val_classes': 100,
                'train_dataset_kwargs': {
                    'n': 650, 'k': 4, 'meta_batch_size': 1
                },
                'val_dataset_kwargs': {
                    'n': 6, 'k': 4, 'meta_batch_size': 3
                },
                'test_dataset_kwargs': {
                    'n': 6, 'k': 4, 'meta_batch_size': 3
                },
            },
            {
                'num_train_classes': 1200,
                'num_val_classes': 100,
                'train_dataset_kwargs': {
                    'n': 7, 'k': 4, 'meta_batch_size': 600
                },
                'val_dataset_kwargs': {
                    'n': 6, 'k': 4, 'meta_batch_size': 3
                },
                'test_dataset_kwargs': {
                    'n': 6, 'k': 4, 'meta_batch_size': 3
                },
            },
            {
                'num_train_classes': 1200,
                'num_val_classes': 100,
                'train_dataset_kwargs': {
                    'n': 7, 'k': 4, 'meta_batch_size': 11
                },
                'val_dataset_kwargs': {
                    'n': 6, 'k': 4, 'meta_batch_size': 3
                },
                'test_dataset_kwargs': {
                    'n': 6, 'k': 4, 'meta_batch_size': 3
                },
            },
        ]
    
    def test_raise_exception_when_k_is_too_large(self):
        config = {
            'num_train_classes': 1200,
            'num_val_classes': 100,
            'train_dataset_kwargs': {
                'n': 7, 'k': 12, 'meta_batch_size': 3
            },
            'val_dataset_kwargs': {
                'n': 6, 'k': 4, 'meta_batch_size': 3
            },
            'test_dataset_kwargs': {
                'n': 6, 'k': 4, 'meta_batch_size': 3
            },
        }
        with self.assertRaises(Exception):
            OmniglotDatabase(random_seed=1, config=config)

    def test_train_val_test_folders_are_separate(self):
        for config in self.configs:
            database = OmniglotDatabase(random_seed=1, config=config)
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
        for config in self.configs:
            database = OmniglotDatabase(random_seed=1, config=config)

            all_folders = [os.path.join(database.database_address, class_name) for class_name in os.listdir(database.database_address)]
            self.assertListEqual(
                sorted(all_folders), 
                sorted(database.train_folders + database.val_folders + database.test_folders)
            )

    @patch('tf_datasets.OmniglotDatabase._get_parse_function')
    def test_covering_all_classes_in_one_epoch(self, mocked_parse_function):
        # Make a new database so that the number of classes are dividable by the number of meta batches * n.
        mocked_parse_function.return_value = self.parse_function
        config = copy.copy(self.configs[0])
        config['train_dataset_kwargs'] = {'n': 6, 'k': 4, 'meta_batch_size': 4}
        database = OmniglotDatabase(random_seed=1, config=config)

        # Check for covering all classes
        classes = set()
        counter = 0
        for task_meta_batch, labels_meta_batch in database.train_ds:
            if counter == database.train_ds.steps_per_epoch:
                break
            counter += 1
            for task in task_meta_batch:
                train_ds, val_ds = tf.split(task, num_or_size_splits=2)
                train_ds = tf.squeeze(train_ds, axis=0)
                for class_instances in tf.split(train_ds, num_or_size_splits=config['train_dataset_kwargs']['n']):
                    class_instances = tf.squeeze(class_instances, axis=0)
                    class_instance_address = class_instances[0].numpy().decode('utf-8')
                    class_address = os.path.split(class_instance_address)[0]
                    classes.add(class_address)
        
        self.assertSetEqual(classes, set(database.train_folders))

    @patch('tf_datasets.OmniglotDatabase._get_parse_function')
    def test_covering_all_classes_in_subsequent_epochs(self, mocked_parse_function):
        # This test might not pass because of the random selection of classes at the beginning of the class, but the chances are low
        # Specially if we increase the number of epochs, the chance of not covering classes will decrease.
        # Make a new database so that the number of classes are dividable by the number of meta batches * n.
        num_epochs = 3
        mocked_parse_function.return_value = self.parse_function
        config = copy.copy(self.configs[0])
        config['train_dataset_kwargs'] = {'n': 7, 'k': 4, 'meta_batch_size': 4}
        database = OmniglotDatabase(random_seed=1, config=config)

        # Check for covering all classes
        classes = set()
        counter = 0
        for task_meta_batch, labels_meta_batch in database.train_ds:
            if counter == database.train_ds.steps_per_epoch * num_epochs:
                break
            counter += 1
            for task in task_meta_batch:
                train_ds, val_ds = tf.split(task, num_or_size_splits=2)
                train_ds = tf.squeeze(train_ds, axis=0)
                for class_instances in tf.split(train_ds, num_or_size_splits=config['train_dataset_kwargs']['n']):
                    class_instances = tf.squeeze(class_instances, axis=0)
                    class_instance_address = class_instances[0].numpy().decode('utf-8')
                    class_address = os.path.split(class_instance_address)[0]
                    classes.add(class_address)
        
        self.assertSetEqual(classes, set(database.train_folders))

    @patch('tf_datasets.OmniglotDatabase._get_parse_function')
    def test_labels_are_correct_in_train_and_val_for_every_task(self, mocked_parse_function):
        mocked_parse_function.return_value = self.parse_function
        for config in self.configs:
            config = copy.copy(config)
            config['train_dataset_kwargs']['one_hot_labels'] = False
            database = OmniglotDatabase(random_seed=1, config=config)
            n = config['train_dataset_kwargs']['n']
            k = config['train_dataset_kwargs']['k']
            mbs = config['train_dataset_kwargs']['meta_batch_size']

            counter = 0
            for task_meta_batch, labels_meta_batch in database.train_ds:
                if counter == 2 * database.train_ds.steps_per_epoch:
                    break
                counter += 1
                
                for task_index in range(mbs):
                    task = task_meta_batch[task_index, ...]
                    task_labels = labels_meta_batch[task_index, ...]

                    train_ds, val_ds = tf.split(task, num_or_size_splits=2)
                    train_labels, val_labels = tf.split(task_labels, num_or_size_splits=2)

                    train_ds = tf.squeeze(train_ds, axis=0)
                    val_ds = tf.squeeze(val_ds, axis=0)
                    train_labels = tf.squeeze(train_labels, axis=0)
                    val_labels = tf.squeeze(val_labels, axis=0)

                    class_label_dict = dict()

                    for class_index in range(n):
                        for instance_index in range(k):
                            instance_name = train_ds[class_index, instance_index, ...]
                            label = train_labels[class_index, instance_index, ...].numpy()

                            class_name = os.path.split(instance_name.numpy().decode('utf-8'))[0]
                            if class_name in class_label_dict:
                                self.assertEqual(class_label_dict[class_name], label)
                            else:
                                class_label_dict[class_name] = label

                    for class_index in range(n):
                        for instance_index in range(k):
                            instance_name = val_ds[class_index, instance_index, ...]
                            label = val_labels[class_index, instance_index, ...].numpy()
                            class_name = os.path.split(instance_name.numpy().decode('utf-8'))[0]
                            self.assertIn(class_name, class_label_dict)
                            self.assertEqual(class_label_dict[class_name], label)

    @patch('tf_datasets.OmniglotDatabase._get_parse_function')
    def test_train_and_val_have_different_samples_in_every_task(self, mocked_parse_function):
        mocked_parse_function.return_value = self.parse_function
        for config in self.configs:
            database = OmniglotDatabase(random_seed=1, config=config)
            n = config['train_dataset_kwargs']['n']
            k = config['train_dataset_kwargs']['k']
            mbs = config['train_dataset_kwargs']['meta_batch_size']

            counter = 0
            for task_meta_batch, labels_meta_batch in database.train_ds:
                if counter == 4 * database.train_ds.steps_per_epoch:
                    break
                counter += 1
                
                for task_index in range(mbs):
                    task = task_meta_batch[task_index, ...]
                    task_labels = labels_meta_batch[task_index, ...]

                    train_ds, val_ds = tf.split(task, num_or_size_splits=2)
                    train_labels, val_labels = tf.split(task_labels, num_or_size_splits=2)

                    train_ds = tf.squeeze(train_ds, axis=0)
                    val_ds = tf.squeeze(val_ds, axis=0)
                    train_labels = tf.squeeze(train_labels, axis=0)
                    val_labels = tf.squeeze(val_labels, axis=0)

                    class_instances_dict = dict()

                    for class_index in range(n):
                        for instance_index in range(k):
                            instance_name = train_ds[class_index, instance_index, ...]

                            class_name, instance_name = os.path.split(instance_name.numpy().decode('utf-8'))
                            if not class_name in class_instances_dict:
                                class_instances_dict[class_name] = set()

                            class_instances_dict[class_name].add(instance_name)

                    for class_index in range(n):
                        for instance_index in range(k):
                            instance_name = val_ds[class_index, instance_index, ...]
                            class_name, instance_name = os.path.split(instance_name.numpy().decode('utf-8'))
                            self.assertIn(class_name, class_instances_dict)
                            self.assertNotIn(instance_name, class_instances_dict[class_name])

    @patch('tf_datasets.OmniglotDatabase._get_parse_function')
    def test_no_two_class_in_the_same_task(self, mocked_parse_function):
        mocked_parse_function.return_value = self.parse_function
        for config in self.configs:
            database = OmniglotDatabase(random_seed=1, config=config)
            n = config['train_dataset_kwargs']['n']
            k = config['train_dataset_kwargs']['k']
            mbs = config['train_dataset_kwargs']['meta_batch_size']

            counter = 0
            for task_meta_batch, labels_meta_batch in database.train_ds:
                if counter == 4 * database.train_ds.steps_per_epoch:
                    break
                counter += 1
                
                for task_index in range(mbs):
                    task = task_meta_batch[task_index, ...]
                    task_labels = labels_meta_batch[task_index, ...]

                    train_ds, val_ds = tf.split(task, num_or_size_splits=2)
                    train_labels, val_labels = tf.split(task_labels, num_or_size_splits=2)

                    train_ds = tf.squeeze(train_ds, axis=0)
                    val_ds = tf.squeeze(val_ds, axis=0)
                    train_labels = tf.squeeze(train_labels, axis=0)
                    val_labels = tf.squeeze(val_labels, axis=0)

                    classes = dict()

                    for class_index in range(n):
                        for instance_index in range(k):
                            instance_name = train_ds[class_index, instance_index, ...]
                            class_name = os.path.split(instance_name.numpy().decode('utf-8'))[0]
                            classes[class_name] = classes.get(class_name, 0) + 1
                    
                    for class_name, num_class_instances in classes.items():
                        self.assertEqual(k, num_class_instances)

    @patch('tf_datasets.OmniglotDatabase._get_parse_function')
    def test_different_instances_are_selected_from_each_class_for_train_and_val_each_time(self, mocked_parse_function):
        #  Random seed is selected such that instances are not selected the same for the whole epoch.
        #  This test might fail due to change in random or behaviour of selecting the samples and it might not mean that the code
        #  does not work properly. Maybe from one task in two different times the same train and val data will be selected
        mocked_parse_function.return_value = self.parse_function
        for config in self.configs:
            database = OmniglotDatabase(random_seed=2, config=config)
            n = config['train_dataset_kwargs']['n']
            k = config['train_dataset_kwargs']['k']
            mbs = config['train_dataset_kwargs']['meta_batch_size']

            class_instances = dict()
            class_instances[0] = dict()
            class_instances[1] = dict()

            for epoch in range(2):
                iteartion = 0
                for task_meta_batch, labels_meta_batch in database.train_ds:
                    if iteartion == database.train_ds.steps_per_epoch:
                        break
                    iteartion += 1

                    for task_index in range(mbs):
                        task = task_meta_batch[task_index, ...]
                        task_labels = labels_meta_batch[task_index, ...]

                        train_ds, val_ds = tf.split(task, num_or_size_splits=2)
                        train_labels, val_labels = tf.split(task_labels, num_or_size_splits=2)

                        train_ds = tf.squeeze(train_ds, axis=0)
                        val_ds = tf.squeeze(val_ds, axis=0)
                        train_labels = tf.squeeze(train_labels, axis=0)
                        val_labels = tf.squeeze(val_labels, axis=0)

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
