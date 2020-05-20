import os
from datetime import datetime
from abc import ABC, abstractmethod
import random
from typing import Tuple, Callable, Dict

import tensorflow as tf
import numpy as np


class Database(ABC):
    def __init__(
            self,
            raw_database_address: str,
            database_address: str,
            random_seed: int = -1,
            input_shape: Tuple = (84, 84, 3)
    ):
        """Random seed just sets the random seed for train, val and test folders selection. So if anything random
        happens there, it will be the same with the same random seed. It will be ignored for all other parts
        of the code. Also notice that randomness in that function should be just based on python random."""
        self.raw_database_address = raw_database_address
        self.database_address = database_address
        self.input_shape = input_shape

        if random_seed != -1:
            random.seed(random_seed)

        self.train_folders, self.val_folders, self.test_folders = self.get_train_val_test_folders()

        self.train_folders = self.convert_to_dict(self.train_folders)
        self.val_folders = self.convert_to_dict(self.val_folders)
        self.test_folders = self.convert_to_dict(self.test_folders)

        if random_seed != -1:
            random.seed(None)

    def convert_to_dict(self, folders):
        if type(folders) == list:
            classes = dict()
            for folder in folders:
                instances = [os.path.join(folder, file_name) for file_name in os.listdir(folder)]
                classes[folder] = instances

            folders = classes
        return folders

    def get_input_shape(self) -> Tuple[int, int, int]:
        return self.input_shape

    def get_all_instances(self, partition_name='all', with_classes=False):
        """Return all instances of a partition of dataset
        Partition can be 'train', 'val', 'test' or 'all'
        """
        instances = list()
        if partition_name == 'all':
            partitions = (self.train_folders, self.val_folders, self.test_folders)
        elif partition_name == 'train':
            partitions = (self.train_folders, )
        elif partition_name == 'test':
            partitions = (self.test_folders,)
        elif partition_name == 'val':
            partitions = (self.val_folders, )
        else:
            raise Exception('The argument partition_name should be all, val, test or train.')

        instance_to_class = dict()
        class_ids = dict()
        class_id = 0
        for partition in partitions:
            for class_name, items in partition.items():
                if class_name not in class_ids:
                    class_ids[class_name] = class_id
                    class_id += 1

                for item in items:
                    instances.append(item)
                    instance_to_class[item] = class_name

        if with_classes:
            return instances, instance_to_class, class_ids

        return instances

    @abstractmethod
    def get_train_val_test_folders(self) -> Tuple:
        """Returns train, val and test folders as three lists or three dictionaries.
        Note that the python random seed might have been
        set here based on the class __init__ function."""
        pass

    @abstractmethod
    def _get_parse_function(self) -> Callable:
        pass

    def load_dumped_features(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
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

    def dump_vgg19_last_hidden_layer(self, partition: str) -> None:
        base_model = tf.keras.applications.VGG19(weights='imagenet')
        model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[24].output)

        self.dump_features(
            partition,
            'vgg19_last_hidden_layer',
            model,
            (224, 224),
            4096,
            tf.keras.applications.vgg19.preprocess_input
        )

    def dump_features(
            self,
            dataset_partition: str,
            name: str,
            model: tf.keras.models.Model,
            input_shape: Tuple,
            feature_size: int,
            preprocess_fn: Callable
    ) -> None:
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

        assert (dataset_partition in ('train', 'val', 'test'))

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

    def get_confusion_matrix(self, name: str, partition: str = 'train') -> Tuple[np.ndarray, Dict[str, int]]:
        """Returns the confusion matrix when applying knn on features. The name of dumped features are the input
        to the function. Returns also a mapping of class name to id of the class. The id corresponds with ith row and
        ith column of the confusion matrix"""

        if partition == 'train':
            folders = self.train_folders
        elif partition == 'val':
            folders = self.val_folders
        elif partition == 'test':
            folders = self.test_folders
        else:
            raise Exception('Partition should be one of the train, val or test.')

        class_ids = {class_name[class_name.rindex('/') + 1:]: i for i, class_name in enumerate(folders)}
        dir_path = os.path.join(self.database_address, name)
        confusion_matrix_path = os.path.join(dir_path, 'confusion_matrix.npy')
        if os.path.exists(confusion_matrix_path):
            confusion_matrix = np.load(confusion_matrix_path)
        else:
            from sklearn.neighbors import KNeighborsClassifier
            knn_model = KNeighborsClassifier()

            file_names, features = self.load_dumped_features(name)
            num_instances = len(file_names)
            ys = []

            for file_name in file_names:
                class_path = os.path.dirname(file_name)
                class_name = class_path[class_path.rindex('/') + 1:]
                ys.append(class_ids[class_name])

            sampled_instances = np.random.choice(np.arange(len(file_names)), num_instances, replace=False)
            knn_model.fit(features[sampled_instances, :], np.array(ys)[sampled_instances])

            confusion_matrix = np.zeros(shape=(len(folders), len(folders)))
            for i, y in enumerate(ys):
                if i % 1000 == 0:
                    print(f'classifiying {i} out of {len(ys)} is done.')
                predicted_class = knn_model.predict(features[i, ...].reshape(1, -1))
                confusion_matrix[y, predicted_class] += 1

            print(confusion_matrix)
            np.save(confusion_matrix_path, confusion_matrix)

        return confusion_matrix, class_ids


class MultipleDatabase(object):
    pass
