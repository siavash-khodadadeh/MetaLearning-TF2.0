import os
from typing import Tuple, List, Callable, Dict
import itertools
import shutil

import tensorflow as tf
import tqdm

import settings

from .data_bases import Database


class CelebADatabase(Database):
    def get_train_val_test_partition(self) -> Dict:
        train_val_test_partition = dict()
        with open(os.path.join(settings.CELEBA_RAW_DATA_ADDRESS, 'list_eval_partition.txt')) as list_eval_partition:
            for line in list_eval_partition:
                line_data = line.split()
                if line_data[1] == '0':
                    train_val_test_partition[line_data[0]] = 'train'
                elif line_data[1] == '1':
                    train_val_test_partition[line_data[0]] = 'val'
                else:
                    train_val_test_partition[line_data[0]] = 'test'
        return train_val_test_partition

    def get_attributes_task_dataset(
            self,
            partition,
            k,
            k_val,
            meta_batch_size,
            parse_fn,
            seed,
            default_shape=(84, 84, 3)
    ):
        self.make_attributes_task_dataset()
        tasks_base_address = os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/celeba/attributes_task', partition)

        def get_images_from_task_file(file_address):
            negative_task_address = tf.strings.regex_replace(file_address, 'F', '@')
            negative_task_address = tf.strings.regex_replace(negative_task_address, 'T', 'F')
            negative_task_address = tf.strings.regex_replace(negative_task_address, '@', 'T')
            file_address = tf.strings.join((tasks_base_address, '/', file_address))
            negative_task_address = tf.strings.join((tasks_base_address, '/', negative_task_address))

            positive_lines = tf.strings.split(tf.io.read_file(file_address), '\n')[:-1]
            negative_lines = tf.strings.split(tf.io.read_file(negative_task_address), '\n')[:-1]

            # TODO replace this with choose instead of shuffle
            positive_lines = tf.random.shuffle(positive_lines, seed=seed)
            negative_lines = tf.random.shuffle(negative_lines, seed=seed)

            return (
                (
                    tf.reshape(
                        tf.map_fn(
                            parse_fn,
                            tf.reshape(tf.concat((positive_lines[:k], negative_lines[:k]), axis=0), (-1, 1)),
                            dtype=tf.float32
                        ),
                        (2, k, *default_shape)
                    ),
                    tf.reshape(
                        tf.map_fn(
                            parse_fn,
                            tf.reshape(
                                tf.concat((positive_lines[k:k + k_val], negative_lines[k:k + k_val]), axis=0), (-1, 1)
                            ),
                            dtype=tf.float32
                        ),
                        (2, k_val, *default_shape)
                    ),
                ),
                (
                    tf.one_hot(
                        tf.concat((tf.zeros(shape=(k,), dtype=tf.int32), tf.ones(shape=(k,), dtype=tf.int32)), axis=0),
                        depth=2
                    ),
                    tf.one_hot(
                        tf.concat(
                            (tf.zeros(shape=(k_val,), dtype=tf.int32), tf.ones(shape=(k_val,), dtype=tf.int32)), axis=0
                        ),
                        depth=2
                    ),
                )
            )

        all_tasks = os.listdir(tasks_base_address)
        dataset = tf.data.Dataset.from_tensor_slices(all_tasks)
        dataset = dataset.shuffle(len(all_tasks), seed=seed)
        dataset = dataset.map(get_images_from_task_file)

        steps_per_epoch = len(all_tasks) // meta_batch_size
        dataset = dataset.batch(meta_batch_size, drop_remainder=True)

        setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset

    def make_attributes_task_dataset(self):
        num_train_attributes = 20
        num_val_attributes = 10
        num_test_attributes = 10
        min_samples_for_each_class = 10  # For making exactly the same tasks as in CACTUs set this number to 9.

        attributes_task_base_address = os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/celeba/attributes_task')
        if not os.path.exists(attributes_task_base_address):
            os.makedirs(attributes_task_base_address)
        else:
            return

        train_folder = os.path.join(attributes_task_base_address, 'train')
        val_folder = os.path.join(attributes_task_base_address, 'val')
        test_folder = os.path.join(attributes_task_base_address, 'test')

        for folder_name in (train_folder, val_folder, test_folder):
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

        self.generate_task_data(
            start_attribute=0,
            end_attribute=num_train_attributes,
            min_samples=min_samples_for_each_class,
            partition='train'
        )
        self.generate_task_data(
            start_attribute=num_train_attributes,
            end_attribute=num_train_attributes + num_val_attributes,
            min_samples=min_samples_for_each_class,
            partition='val'
        )
        self.generate_task_data(
            start_attribute=num_train_attributes + num_val_attributes,
            end_attribute=num_train_attributes + num_val_attributes + num_test_attributes,
            min_samples=min_samples_for_each_class,
            partition='test'
        )

    def generate_task_data(self, start_attribute, end_attribute, min_samples, partition):
        train_val_test_partition = self.get_train_val_test_partition()
        identities = self.get_identities()
        with open(os.path.join(settings.CELEBA_RAW_DATA_ADDRESS, 'list_attr_celeba.txt')) as attributes_file:
            num_lines = int(attributes_file.readline())
            attributes = attributes_file.readline().split()
            attributes.sort()

            attributes_positive_sets = list()
            attributes_negative_sets = list()

            for i in range(40):
                attributes_positive_sets.append(set())
                attributes_negative_sets.append(set())

            for _ in range(num_lines):
                line_data = attributes_file.readline().split()
                example_name = line_data[0]

                if train_val_test_partition[example_name] != partition:
                    continue

                for i in range(40):
                    if line_data[i + 1] == '1':
                        attributes_positive_sets[i].add(example_name)
                    else:
                        attributes_negative_sets[i].add(example_name)

        all_combinations = list(itertools.combinations(range(start_attribute, end_attribute), 3))
        boolean_combinations = list(itertools.product((True, False), repeat=3))[4:]

        counter = 0
        for combination in tqdm.tqdm(all_combinations):
            for bool_combination in boolean_combinations:
                positive_sets = list()
                negative_sets = list()
                for i in range(3):
                    if bool_combination[i]:
                        positive_sets.append(attributes_positive_sets[combination[i]])
                        negative_sets.append(attributes_negative_sets[combination[i]])
                    else:
                        positive_sets.append(attributes_negative_sets[combination[i]])
                        negative_sets.append(attributes_positive_sets[combination[i]])

                positive_samples = [
                    item for item in positive_sets[0] if item in positive_sets[1] and item in positive_sets[2]
                ]
                negative_samples = [
                    item for item in negative_sets[0] if item in negative_sets[1] and item in negative_sets[2]
                ]

                if len(positive_samples) > min_samples and len(negative_samples) > min_samples:
                    counter += 1
                    self.make_task_file(combination, bool_combination, positive_samples, partition, identities)
                    bool_combination = [not item for item in bool_combination]
                    self.make_task_file(combination, bool_combination, negative_samples, partition, identities)
        print(f'Number of {partition} tasks : {counter}')

    def make_task_file(self, combination, bool_combinations, samples, partition, identities):
        task_name = f'{str(bool_combinations[0])[:1]}{combination[0]}_' \
                    f'{str(bool_combinations[1])[:1]}{combination[1]}_' \
                    f'{str(bool_combinations[2])[:1]}{combination[2]}.txt'

        task_file_name = os.path.join(
            settings.PROJECT_ROOT_ADDRESS,
            'data/celeba/attributes_task',
            partition,
            task_name
        )
        with open(task_file_name, 'w') as task_file:
            for sample in samples:
                sample_full_address = os.path.join(self.database_address, partition, identities[sample], sample)
                assert (os.path.exists(sample_full_address))
                task_file.write(
                    sample_full_address
                )
                task_file.write('\n')

    def get_identities(self):
        identities = dict()
        with open(os.path.join(settings.CELEBA_RAW_DATA_ADDRESS, 'identity_CelebA.txt')) as identity_file:
            for line in identity_file:
                line_data = line.split()
                identities[line_data[0]] = line_data[1]

        return identities

    def __init__(self, input_shape=(84, 84, 3)):
        # Celeba has tow different kind of tasks and we keep it backward compatible. In fact this is the only dataset
        # which makes a copy of itself in data directory.
        self.raw_database_address = settings.CELEBA_RAW_DATA_ADDRESS
        self.database_address = os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/celeba/identification_task')
        self.prepare_database()

        super(CelebADatabase, self).__init__(
            settings.CELEBA_RAW_DATA_ADDRESS,
            os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/celeba/identification_task'),
            random_seed=-1,
            input_shape=input_shape
        )

    def get_train_val_test_folders(self) -> Tuple[List[str], List[str], List[str]]:
        dataset_folders = list()
        for dataset_type in ('train', 'val', 'test'):
            dataset_base_address = os.path.join(self.database_address, dataset_type)
            folders = [
                os.path.join(dataset_base_address, class_name) for class_name in os.listdir(dataset_base_address)
            ]
            dataset_folders.append(folders)
        return dataset_folders[0], dataset_folders[1], dataset_folders[2]

    def _get_parse_function(self) -> Callable:
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            # For face recognition ablation study uncomment this line

            # for face net uncomment this line
            # image = tf.image.resize(image, (160, 160))
            #
            # image = tf.cast(image, tf.float32)
            # return (image - 127.5) / 128.0

            # The true parse function
            image = tf.image.resize(image, self.get_input_shape()[:2])
            image = tf.cast(image, tf.float32)
            return image / 255.

        return parse_function

    def put_images_in_train_val_and_test_folders(self):
        os.mkdir(os.path.join(self.database_address, 'train'))
        os.mkdir(os.path.join(self.database_address, 'val'))
        os.mkdir(os.path.join(self.database_address, 'test'))
        identity_map = dict()

        with open(os.path.join(self.raw_database_address, 'identity_CelebA.txt')) as f:
            for line in f:
                line_data = line.split()
                identity_map[line_data[0]] = line_data[1]

        with open(os.path.join(self.raw_database_address, 'list_eval_partition.txt')) as f:
            for line in f:
                line_data = line.split()

                if line_data[1] == '0':
                    target_address = os.path.join(self.database_address, 'train')
                elif line_data[1] == '1':
                    target_address = os.path.join(self.database_address, 'val')
                else:
                    target_address = os.path.join(self.database_address, 'test')

                target_dir = os.path.join(target_address, identity_map[line_data[0]])
                if not os.path.exists(target_dir):
                    os.mkdir(target_dir)

                target_address = os.path.join(target_dir, line_data[0])

                source_address = os.path.join(self.raw_database_address, 'img_align_celeba', line_data[0])
                shutil.copyfile(source_address, target_address)

    def prepare_database(self):
        if not os.path.exists(self.database_address):
            os.makedirs(self.database_address)
            self.put_images_in_train_val_and_test_folders()


class LFWDatabase(Database):
    def __init__(self, input_shape=(84, 84, 3)):
        super(LFWDatabase, self).__init__(
            settings.LFW_RAW_DATA_ADDRESS,
            settings.LFW_RAW_DATA_ADDRESS,
            random_seed=-1,
            input_shape=input_shape
        )

    def get_train_val_test_folders(self) -> Tuple[List[str], List[str], List[str]]:
        # TODO fix this
        dataset_folders = [
            os.path.join(self.database_address, class_name) for class_name in os.listdir(self.raw_database_address)
        ]

        return dataset_folders, dataset_folders, dataset_folders

    def _get_parse_function(self) -> Callable:
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            # For face recognition ablation study uncomment this line
            # image = tf.image.resize(image, (84, 84))

            # for face net uncomment this line
            # image = tf.image.resize(image, (160, 160))
            # image = tf.cast(image, tf.float32)
            # return (image - 127.5) / 128.0

            # The true parse function
            image = tf.image.resize(image, self.get_input_shape()[:2])
            image = tf.cast(image, tf.float32)
            # return image / 127.5 - 1
            return image / 255.

        return parse_function


class VGGFace2Database(Database):
    def __init__(self, input_shape=(84, 84, 3)):
        super(VGGFace2Database, self).__init__(
            settings.VGG_FACE2,
            settings.VGG_FACE2,
            random_seed=-1,
            input_shape=input_shape
        )
        self.input_shape = (160, 160, 3)

    def get_train_val_test_folders(self) -> Tuple[List[str], List[str], List[str]]:
        # TODO fix this
        train_address = os.path.join(self.database_address, 'train')
        dataset_folders = [
            os.path.join(train_address, class_name) for class_name in os.listdir(train_address)
        ]

        celeba_val_address = os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/celeba/identification_task/val/')
        val_folders = [
            os.path.join(celeba_val_address, class_name) for class_name in os.listdir(celeba_val_address)
        ]

        celeba_test_address = os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/celeba/identification_task/test/')
        test_folders = [
            os.path.join(celeba_test_address, class_name) for class_name in os.listdir(celeba_test_address)
        ]
        return dataset_folders, val_folders, test_folders

    def _get_parse_function(self) -> Callable:
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            # For face recognition ablation study uncomment this line
            # image = tf.image.resize(image, (84, 84))

            # for face net uncomment this line
            # image = tf.image.resize(image, (160, 160))
            #
            # image = tf.cast(image, tf.float32)
            # return (image - 127.5) / 128.0

            # The true parse function
            image = tf.image.resize(image, self.get_input_shape()[:2])
            image = tf.cast(image, tf.float32)
            return image / 127.5 - 1

        return parse_function
