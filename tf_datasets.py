import os
import shutil
from abc import ABC, abstractclassmethod
import random

import tensorflow as tf

import settings


class Database(ABC):
    def __init__(self, raw_database_address, database_address, random_seed=-1, config=None):
        if random_seed != -1:
            random.seed(random_seed)
            tf.random.set_seed(random_seed)
        
        if config is None:
            config = {}

        self.raw_database_address = raw_database_address
        self.database_address = database_address
        self.config = config

        self.prepare_database()
        self.train_folders, self.val_folders, self.test_folders = None, None, None
        self.train_ds, self.val_ds, self.test_ds = self.get_train_val_test_datasets()

    @abstractclassmethod
    def prepare_database(self):
        pass

    def check_number_of_samples_at_each_class_meet_minimum(self, folders, minimum):
        for folder in folders:
            if len(os.listdir(folder)) < 2 * minimum:
                raise Exception(f'There should be at least {2 * minimum} examples in each class. Class {folder} does not have that many examples')

    def get_train_val_test_datasets(self):
        """Returns 3 tf.data.dataset instances for train, validation and test."""
        num_train_classes = self.config['num_train_classes']
        num_val_classes = self.config['num_val_classes']

        folders = [os.path.join(self.database_address, class_name) for class_name in os.listdir(self.database_address)]
        folders.sort()
        random.shuffle(folders)
        self.train_folders = folders[:num_train_classes]
        self.val_folders = folders[num_train_classes:num_train_classes + num_val_classes]
        self.test_folders = folders[num_train_classes + num_val_classes:]

        self.check_number_of_samples_at_each_class_meet_minimum(self.train_folders, self.config['train_dataset_kwargs']['k'])
        self.check_number_of_samples_at_each_class_meet_minimum(self.val_folders, self.config['val_dataset_kwargs']['k'])
        self.check_number_of_samples_at_each_class_meet_minimum(self.test_folders, self.config['test_dataset_kwargs']['k'])

        assert(self.config['val_dataset_kwargs']['meta_batch_size'] == 1)
        assert (self.config['test_dataset_kwargs']['meta_batch_size'] == 1)

        return self.get_supervised_meta_learning_dataset(
            self.train_folders, 
            **self.config['train_dataset_kwargs']
        ), self.get_supervised_meta_learning_dataset(
            self.val_folders, 
            **self.config['val_dataset_kwargs']
        ), self.get_supervised_meta_learning_dataset(
            self.test_folders, 
            **self.config['test_dataset_kwargs']
        )

    def _get_instances(self, k):
        def get_instances(class_dir_address):
            return tf.data.Dataset.list_files(class_dir_address, shuffle=False).take(2 * k)
        return get_instances

    def _get_parse_function(self):
        def parse_function(example_address):
            return example_address
        return parse_function

    def get_supervised_meta_learning_dataset(
            self,
            folders,
            n,
            k,
            meta_batch_size,
            one_hot_labels=True,
            reshuffle_each_iteration=True,
            # num_repeats=-1
    ):
        classes = [class_name + '/*' for class_name in folders]
        steps_per_epoch = len(classes) // n // meta_batch_size

        labels_dataset = tf.data.Dataset.range(n)
        if one_hot_labels:
            labels_dataset = labels_dataset.map(lambda example: tf.one_hot(example, depth=n))

        labels_dataset = labels_dataset.interleave(
            lambda x: tf.data.Dataset.from_tensors(x).repeat(2 * k),
            cycle_length=n, 
            block_length=k
        )
        labels_dataset = labels_dataset.repeat(meta_batch_size)
        labels_dataset = labels_dataset.repeat(steps_per_epoch)

        dataset = tf.data.Dataset.from_tensor_slices(classes)
        dataset = dataset.shuffle(buffer_size=len(folders), reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = dataset.interleave(self._get_instances(k), cycle_length=n, block_length=k)
        dataset = dataset.map(self._get_parse_function())

        dataset = tf.data.Dataset.zip((dataset, labels_dataset))
        
        dataset = dataset.batch(k, drop_remainder=False)
        dataset = dataset.batch(n, drop_remainder=True)
        dataset = dataset.batch(2, drop_remainder=True)
        dataset = dataset.batch(meta_batch_size, drop_remainder=True)

        steps_per_epoch = len(classes) // n // meta_batch_size
        setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset

    def get_umtra_dataset(self, folders, n, k, meta_batch_size, one_hot_labels=True):
        pass


class OmniglotDatabase(Database):
    def __init__(self, random_seed=-1, config=None):
        super(OmniglotDatabase, self).__init__(
            settings.OMNIGLOT_RAW_DATA_ADDRESS,
            os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/omniglot'),
            random_seed=random_seed,
            config=config
        )

    def _get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, (28, 28))
            image = tf.cast(image, tf.float32)

            return 1 - (image / 255.)

        return parse_function

    def prepare_database(self):
        for item in ('images_background', 'images_evaluation'):
            alphabets = os.listdir(os.path.join(self.raw_database_address, item))
            for alphabet in alphabets:
                alphabet_address = os.path.join(self.raw_database_address, item, alphabet)
                for character in os.listdir(alphabet_address):
                    character_address = os.path.join(alphabet_address, character)
                    destination_address = os.path.join(self.database_address, alphabet + '_' + character)
                    if not os.path.exists(destination_address):
                        shutil.copytree(character_address, destination_address)
