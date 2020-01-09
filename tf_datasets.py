import os
from datetime import datetime
import shutil
import itertools
from abc import ABC, abstractmethod
import random

import tensorflow as tf
import numpy as np
import tqdm

import settings


class Database(ABC):
    def __init__(self, raw_database_address, database_address, random_seed=-1):
        if random_seed != -1:
            random.seed(random_seed)
            tf.random.set_seed(random_seed)
            np.random.seed(random_seed)
        
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
            return tf.data.Dataset.list_files(class_dir_address, shuffle=True).take(2 * k)
        return get_instances

    def _get_parse_function(self):
        def parse_function(example_address):
            return example_address
        return parse_function

    def load_dumped_features(self, name):
        """Returns two numpy matrices. First one is the name of all files which are dumped with the shape of (N, )
        and the second one is the features with the shape of (N, M), where N is the number of files and M is the
        length of the each feature vector. The i-th row in second matrix is the feature vector of i-th element in the
        first matrix.
        """
        dir_path = os.path.join(self.database_address, name)
        if not os.path.exists(dir_path):
            raise Exception('Requested features are not dumped.')

        files_names_address = os.path.join(dir_path, 'files_names.npy')
        features_address = os.path.join(dir_path, 'features.npy')
        all_files = np.load(files_names_address)
        features = np.load(features_address)

        return all_files, features

    def dump_vgg19_last_hidden_layer(self, partition):
        base_model = tf.keras.applications.VGG19(weights='imagenet')
        model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[24].output)

        self.dump_features(
            partition,
            'vgg19_last_hidden_layer',
            model,
            (224, 224),
            4096,
            tf.keras.applications.inception_v3.preprocess_input
        )

    def dump_features(self, dataset_partition, name, model, input_shape, feature_size, preprocess_fn):
        """Dumps the features of a partition of database: Train, val or test. The features are extracted
        by a model and then dumped into a .npy file. Also the filenames for which the features are dumped will be
        stored in another .npy file. The name of these .npy files are features.npy and files_names.npy and they are
        stored in a directory under database_address. The name of directory is the join of dataset_partition and
        name argument.

        Inputs:
        dataset_partition: Should be train, val or test.
        name: The name of features to dump. For example VGG19_layer_4
        model: The network or model which is used to dump features. The output of this model is the feature which
        will be stored.
        input_shape: The input shape of each image file.
        feature_size: The length or depth of features.
        preprocess_fn: The preprocessing function which is applied to the raw image before passing it through the model.
        Use None to ignore it.

        Returns: None

        This function does not return anything but stores two numpy files. In order to load them use
        load_dumped_features function. """

        if dataset_partition == 'train':
            files_dir = self.train_folders
        elif dataset_partition == 'val':
            files_dir = self.val_folders
        elif dataset_partition == 'test':
            files_dir = self.test_folders
        else:
            raise Exception('Pratition is not train val or test!')

        assert(dataset_partition in ('train', 'val', 'test'))

        dir_path = os.path.join(self.database_address, f'{name}_{dataset_partition}')
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        all_files = list()

        for class_name in files_dir:
            all_files.extend([os.path.join(class_name, file_name) for file_name in os.listdir(class_name)])

        files_names_address = os.path.join(dir_path, 'files_names.npy')
        np.save(files_names_address, all_files)

        features_address = os.path.join(dir_path, 'features.npy')

        n = len(all_files)
        m = feature_size
        features = np.zeros(shape=(n, m))

        begin_time = datetime.now()

        for index, sampled_file in enumerate(all_files):
            if index % 1000 == 0:
                print(f'{index}/{len(all_files)} images dumped')

            img = tf.keras.preprocessing.image.load_img(sampled_file, target_size=(input_shape[:2]))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            if preprocess_fn is not None:
                img = preprocess_fn(img)

            features[index, :] = model.predict(img).reshape(-1)

        np.save(features_address, features)
        end_time = datetime.now()
        print('Features dumped')
        print(f'Time to dump features: {str(end_time - begin_time)}')


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

    def get_folders_with_greater_than_equal_k_files(self, folders, k):
        to_be_removed = list()
        for folder in folders:
            if len(os.listdir(folder)) < k:
                to_be_removed.append(folder)

        for folder in to_be_removed:
            folders.remove(folder)

        return folders

    def get_dataset_from_tasks_directly(self, tasks, n, k, meta_batch_size, one_hot_labels=True):
        dataset = tf.data.Dataset.from_tensor_slices(tasks)
        dataset = dataset.unbatch()
        dataset = dataset.map(self._get_parse_function(), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        steps_per_epoch = len(tasks) // meta_batch_size
        labels_dataset = self.make_labels_dataset(n, k, meta_batch_size, steps_per_epoch, one_hot_labels)
        dataset = tf.data.Dataset.zip((dataset, labels_dataset))

        dataset = dataset.batch(k, drop_remainder=False)
        dataset = dataset.batch(n, drop_remainder=True)
        dataset = dataset.batch(2, drop_remainder=True)
        dataset = dataset.batch(meta_batch_size, drop_remainder=True)

        # for item in dataset:
        #     print('dataset item')
        #     print(item)
        #     exit()

        setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset

    def get_sp_meta_learning_dataset(
        self,
        folders,
        n,
        k,
        meta_batch_size,
        features_name='vgg19_last_hidden_layer_train',
        one_hot_labels=True,
        reshuffle_each_iteration=True,

    ):
        from utils import SP
        from datetime import datetime
        # load the feature space and sampled files first.
        # features = np.load(os.path.join(
        #     settings.PROJECT_ROOT_ADDRESS,
        #     'models/umtra-iterative-projection/vgg19_last_hidden_layer_CelebADatabase/features.npy'
        # ))
        # sampled_files = np.load(os.path.join(
        #     settings.PROJECT_ROOT_ADDRESS,
        #     'models/umtra-iterative-projection/vgg19_last_hidden_layer_CelebADatabase/sampled_files.npy'
        # ))

        sampled_files, features = self.load_dumped_features(features_name)

        feature_dict = {
            bytes(file_name, encoding='utf-8'): feature for file_name, feature in zip(sampled_files, features)
        }

        def get_instances(k):
            def choose_files_with_sp(files, num):
                """return a list which is a subset of files and it has to have num elements"""
                feature_of_files = np.zeros(shape=(4096, len(files)))
                np.random.shuffle(files)

                for i, file in enumerate(files):
                    feature_of_files[:, i] = feature_dict[file]

                #  # mean squared error
                #  data_point = feature_of_files[:, 0]
                #  others = feature_of_files[:, 1:]
                #  res = np.sum(np.square(others - data_point.reshape((4096, 1))), axis=0)
                #  closest = np.argmax(res)

                indices = SP(feature_of_files, num)
                # indices = np.array([0, 1])

                return files[indices]

            def f(class_dir_address):
                def get_files(dir_address):
                    dir_address = dir_address.numpy()[:-1]
                    files = np.array([os.path.join(dir_address, file_name) for file_name in os.listdir(dir_address)])
                    # the first k are train and the remaining are validation
                    # print(files[:2 * k])
                    chosen_files = choose_files_with_sp(files, 2 * k)

                    return np.array(chosen_files)

                return tf.data.Dataset.from_tensor_slices(tf.py_function(get_files, inp=[class_dir_address], Tout=tf.string))
            return f
            # return self._get_instances(k)

        folders = self.get_folders_with_greater_than_equal_k_files(folders, 2 * k)

        classes = [class_name + '/*' for class_name in folders]
        steps_per_epoch = len(classes) // n // meta_batch_size

        labels_dataset = self.make_labels_dataset(n, k, meta_batch_size, steps_per_epoch, one_hot_labels)

        dataset = tf.data.Dataset.from_tensor_slices(classes)
        dataset = dataset.shuffle(buffer_size=len(folders), reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = dataset.interleave(
            get_instances(k),
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

        # dataset = dataset.prefetch(4)

        setattr(dataset, 'steps_per_epoch', steps_per_epoch)

        return dataset

    def get_supervised_meta_learning_dataset(
            self,
            folders,
            n,
            k,
            meta_batch_size,
            one_hot_labels=True,
            reshuffle_each_iteration=True,
    ):
        folders = self.get_folders_with_greater_than_equal_k_files(folders, 2 * k)

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

    def get_abstract_learning_dataset(
            self,
            folders,
            n,
            k,
            meta_batch_size,
            reshuffle_each_iteration=True
    ):
        folders = self.get_folders_with_greater_than_equal_k_files(folders, 2 * k)
        classes = [class_name + '/*' for class_name in folders]
        steps_per_epoch = len(classes) // n // meta_batch_size

        labels_dataset = tf.data.Dataset.from_tensor_slices(tf.concat(([1.], tf.zeros((n - 1), dtype=tf.float32)), axis=0))

        labels_dataset = labels_dataset.interleave(
            lambda x: tf.data.Dataset.from_tensors(x).repeat(2 * k),
            cycle_length=n,
            block_length=k
        )
        labels_dataset = labels_dataset.repeat(meta_batch_size)
        labels_dataset = labels_dataset.repeat(steps_per_epoch)

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

    def get_random_dataset(self, folders, n, meta_batch_size, one_hot_labels=True, reshuffle_each_iteration=True):
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


class CelebADatabase(Database):
    def get_input_shape(self):
        return 84, 84, 3

    def get_train_val_test_partition(self):
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

    def get_attributes_task_dataset(self, partition, k, meta_batch_size):
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

            positive_lines = tf.random.shuffle(positive_lines)
            negative_lines = tf.random.shuffle(negative_lines)
            positive_lines = positive_lines[:2 * k]
            negative_lines = negative_lines[:2 * k]

            return tf.concat((positive_lines[:k], negative_lines[:k], positive_lines[k:], negative_lines[k:]), axis=0)

        all_tasks = os.listdir(tasks_base_address)
        dataset = tf.data.Dataset.from_tensor_slices(all_tasks)
        dataset = dataset.map(get_images_from_task_file)
        dataset = dataset.unbatch()

        steps_per_epoch = len(all_tasks) // meta_batch_size
        labels_dataset = self.make_labels_dataset(2, k, meta_batch_size, steps_per_epoch, True)

        dataset = dataset.map(self._get_parse_function(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = tf.data.Dataset.zip((dataset, labels_dataset))

        dataset = dataset.batch(k, drop_remainder=False)
        dataset = dataset.batch(2, drop_remainder=True)
        dataset = dataset.batch(2, drop_remainder=True)
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
                assert(os.path.exists(sample_full_address))
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

    def __init__(self, random_seed=-1, config=None):
        super(CelebADatabase, self).__init__(
            settings.CELEBA_RAW_DATA_ADDRESS,
            os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/celeba/identification_task'),
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


if __name__ == '__main__':
    database = MiniImagenetDatabase()
    database.dump_vgg19_last_hidden_layer(partition='train')
