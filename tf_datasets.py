import os
import shutil
from abc import ABC, abstractmethod
import random

import tensorflow as tf

import settings


# TODO
# Make the mini imagenet database such that it will use the files in mini-imagenet raw dataset.

class Database(ABC):
    def __init__(self, raw_database_address, database_address, random_seed=-1):
        if random_seed != -1:
            random.seed(random_seed)
            tf.random.set_seed(random_seed)
        
        self.raw_database_address = raw_database_address
        self.database_address = database_address

        self.prepare_database()
        self.train_folders, self.val_folders, self.test_folders = self.get_train_val_test_folders()

        self.input_shape = self.get_input_shape()

    @abstractmethod
    def get_input_shape(self):
        pass

    @abstractmethod
    def prepare_database(self):
        pass

    @abstractmethod
    def get_train_val_test_folders(self):
        pass

    def check_number_of_samples_at_each_class_meet_minimum(self, folders, minimum):
        for folder in folders:
            if len(os.listdir(folder)) < 2 * minimum:
                raise Exception(f'There should be at least {2 * minimum} examples in each class. Class {folder} does not have that many examples')

    def _get_instances(self, k):
        def get_instances(class_dir_address):
            return tf.data.Dataset.list_files(class_dir_address, shuffle=False).take(2 * k)
        return get_instances

    def _get_parse_function(self):
        def parse_function(example_address):
            return example_address
        return parse_function

    def make_labels_dataset(self, n, k, meta_batch_size, steps_per_epoch, one_hot_labels):
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
        return labels_dataset

    def get_supervised_meta_learning_dataset(
            self,
            folders,
            n,
            k,
            meta_batch_size,
            one_hot_labels=True,
            reshuffle_each_iteration=True,
    ):
        for class_name in folders:
            assert(len(os.listdir(class_name)) > 2 * k), f'The number of instances in each class should be larger ' \
                                                         f'than {2 * k}, however, the number of instances in' \
                                                         f' {class_name} are: {len(os.listdir(class_name))}'

        classes = [class_name + '/*' for class_name in folders]
        steps_per_epoch = len(classes) // n // meta_batch_size

        labels_dataset = self.make_labels_dataset(n, k, meta_batch_size, steps_per_epoch, one_hot_labels)

        dataset = tf.data.Dataset.from_tensor_slices(classes)
        dataset = dataset.shuffle(buffer_size=len(folders), reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = dataset.interleave(
            self._get_instances(k),
            cycle_length=n,
            block_length=k,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.map(self._get_parse_function(), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = tf.data.Dataset.zip((dataset, labels_dataset))
        
        dataset = dataset.batch(k, drop_remainder=False)
        dataset = dataset.batch(n, drop_remainder=True)
        dataset = dataset.batch(2, drop_remainder=True)
        dataset = dataset.batch(meta_batch_size, drop_remainder=True)

        setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset

    def get_umtra_dataset(
        self,
        folders,
        n,
        meta_batch_size,
        augmentation_function=None,
        one_hot_labels=True,
        reshuffle_each_iteration=True
    ):
        if augmentation_function is None:
            def same(x):
                return x

            augmentation_function = same

        def parse_umtra(example, label):
            return tf.stack((example, augmentation_function(example))), tf.stack((label, label))

        instances = list()
        for class_name in folders:
            instances.extend(os.path.join(class_name, file_name) for file_name in os.listdir(class_name))
        instances.sort()

        steps_per_epoch = len(instances) // n // meta_batch_size
        labels_dataset = self.make_labels_dataset(n, 1, meta_batch_size, steps_per_epoch, one_hot_labels)

        dataset = tf.data.Dataset.from_tensor_slices(instances)
        dataset = dataset.shuffle(buffer_size=len(instances), reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = dataset.map(self._get_parse_function())

        dataset = tf.data.Dataset.zip((dataset, labels_dataset))

        dataset = dataset.batch(1, drop_remainder=False)
        dataset = dataset.batch(n, drop_remainder=True)
        dataset = dataset.map(parse_umtra)
        dataset = dataset.batch(meta_batch_size, drop_remainder=True)
        setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset


class OmniglotDatabase(Database):
    def __init__(
        self,
        random_seed,
        num_train_classes,
        num_val_classes,
    ):
        self.num_train_classes = num_train_classes
        self.num_val_classes = num_val_classes
        super(OmniglotDatabase, self).__init__(
            settings.OMNIGLOT_RAW_DATA_ADDRESS,
            os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/omniglot'),
            random_seed=random_seed,
        )

    def get_input_shape(self):
        return 28, 28, 1

    def get_train_val_test_folders(self):
        num_train_classes = self.num_train_classes
        num_val_classes = self.num_val_classes

        folders = [os.path.join(self.database_address, class_name) for class_name in os.listdir(self.database_address)]
        folders.sort()
        random.shuffle(folders)
        train_folders = folders[:num_train_classes]
        val_folders = folders[num_train_classes:num_train_classes + num_val_classes]
        test_folders = folders[num_train_classes + num_val_classes:]

        return train_folders, val_folders, test_folders

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


class MiniImagenetDatabase(Database):
    def get_input_shape(self):
        return 84, 84, 3

    def __init__(self, random_seed=-1, config=None):
        super(MiniImagenetDatabase, self).__init__(
            settings.MINI_IMAGENET_RAW_DATA_ADDRESS,
            os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/mini-imagenet'),
            random_seed=random_seed,
        )

    def get_train_val_test_folders(self):
        dataset_folders = list()
        for dataset_type in ('train', 'val', 'test'):
            dataset_base_address = os.path.join(self.database_address, dataset_type)
            folders = [
                os.path.join(dataset_base_address, class_name) for class_name in os.listdir(dataset_base_address)
            ]
            dataset_folders.append(folders)
        return dataset_folders[0], dataset_folders[1], dataset_folders[2]

    def _get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, (84, 84))
            image = tf.cast(image, tf.float32)

            return image / 255.
        return parse_function

    def prepare_database(self):
        if not os.path.exists(self.database_address):
            shutil.copytree(self.raw_database_address, self.database_address)
