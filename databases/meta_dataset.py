import os
from typing import Tuple, Callable
import json

import tensorflow as tf
from scipy.io import loadmat

import settings
from utils import convert_grayscale_images_to_rgb

from .data_bases import Database
from .parse_mixins import JPGParseMixin


class CUBDatabase(JPGParseMixin, Database):
    def __init__(self, input_shape=(84, 84, 3)):
        super(CUBDatabase, self).__init__(
            raw_database_address=settings.CUB_RAW_DATASEST_ADDRESS,
            database_address='',
            random_seed=-1,
            input_shape=input_shape
        )

    def fix_2d_instances(self, train_folders, val_folders, test_folders):
        cub_info_file = os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/fixed_cubs_bad_examples.txt')
        if not os.path.exists(cub_info_file):
            instances = []
            for item in train_folders:
                instances.extend([os.path.join(item, file_name) for file_name in os.listdir(item)])
            for item in val_folders:
                instances.extend([os.path.join(item, file_name) for file_name in os.listdir(item)])
            for item in test_folders:
                instances.extend([os.path.join(item, file_name) for file_name in os.listdir(item)])

            num_fixed, fixed_instances = convert_grayscale_images_to_rgb(instances)

            with open(cub_info_file, 'w') as f:
                f.write(f'Changed {num_fixed} 2d data points to 3d.\n')
                f.write('\n'.join(fixed_instances))

    def get_train_val_test_folders(self) -> Tuple:
        """Returns train, val and test folders as three lists or three dictionaries.
        Note that the python random seed might have been
        set here based on the class __init__ function."""
        images_folder = os.path.join(self.raw_database_address, 'CUB_200_2011', 'images')
        splits = json.load(open(os.path.join(
            settings.PROJECT_ROOT_ADDRESS,
            'databases',
            'meta_dataset_meta',
            'splits',
            'cub_splits.json'
            )
        ))

        train_folders = [os.path.join(images_folder, item) for item in splits['train']]
        val_folders = [os.path.join(images_folder, item) for item in splits['valid']]
        test_folders = [os.path.join(images_folder, item) for item in splits['test']]

        self.fix_2d_instances(train_folders, val_folders, test_folders)

        return train_folders, val_folders, test_folders


class AirplaneDatabase(JPGParseMixin, Database):
    def __init__(self, input_shape=(84, 84, 3)):
        super(AirplaneDatabase, self).__init__(
            raw_database_address=settings.AIRCRAFT_RAW_DATASET_ADDRESS,
            database_address='',
            random_seed=-1,
            input_shape=input_shape
        )

    def get_train_val_test_folders(self) -> Tuple:
        """Returns train, val and test folders as three lists or three dictionaries.
        Note that the python random seed might have been
        set here based on the class __init__ function."""
        images_folder = os.path.join(self.raw_database_address, 'data', 'images')
        classes = dict()
        for partition in ('train', 'val', 'test'):
            with open(os.path.join(self.raw_database_address, 'data', f'images_variant_{partition}.txt')) as f:
                for line in f:
                    img, variant = line[:7], line[8:-1]
                    if variant not in classes:
                        classes[variant] = list()
                    classes[variant].append(os.path.join(images_folder, f'{img}.jpg'))

        splits = json.load(open(os.path.join(
            settings.PROJECT_ROOT_ADDRESS,
            'databases',
            'meta_dataset_meta',
            'splits',
            'airplane.json'
            )
        ))
        train_folders = {}
        val_folders = {}
        test_folders = {}

        for item in splits['train']:
            train_folders[item] = classes[item]
        for item in splits['valid']:
            val_folders[item] = classes[item]
        for item in splits['test']:
            test_folders[item] = classes[item]

        return train_folders, val_folders, test_folders


class DTDDatabase(JPGParseMixin, Database):
    def __init__(self, input_shape=(84, 84, 3)):
        super(DTDDatabase, self).__init__(
            raw_database_address=settings.DTD_RAW_DATASET_ADDRESS,
            database_address='',
            random_seed=-1,
            input_shape=input_shape
        )

    def get_train_val_test_folders(self) -> Tuple:
        """Returns train, val and test folders as three lists or three dictionaries.
        Note that the python random seed might have been
        set here based on the class __init__ function."""
        weird_dir_file = os.path.join(self.raw_database_address, 'dtd', 'images', 'waffled', '.directory')
        if os.path.exists(weird_dir_file):
            os.remove(weird_dir_file)

        splits = json.load(open(os.path.join(
            settings.PROJECT_ROOT_ADDRESS,
            'databases',
            'meta_dataset_meta',
            'splits',
            'dtd.json'
            )
        ))
        images_folder = os.path.join(self.raw_database_address, 'dtd', 'images')
        train_folders = [os.path.join(images_folder, item) for item in splits['train']]
        val_folders = [os.path.join(images_folder, item) for item in splits['valid']]
        test_folders = [os.path.join(images_folder, item) for item in splits['test']]

        return train_folders, val_folders, test_folders


class VGGFlowerDatabase(JPGParseMixin, Database):
    def __init__(self, input_shape=(84, 84, 3)):
        super(VGGFlowerDatabase, self).__init__(
            raw_database_address=settings.VGG_FLOWER_RAW_DATASET_ADDRESS,
            database_address='',
            random_seed=-1,
            input_shape=input_shape
        )

    def get_train_val_test_folders(self) -> Tuple:
        """Returns train, val and test folders as three lists or three dictionaries.
        Note that the python random seed might have been
        set here based on the class __init__ function."""
        splits = json.load(open(os.path.join(
            settings.PROJECT_ROOT_ADDRESS,
            'databases',
            'meta_dataset_meta',
            'splits',
            'vgg_flowers.json'
            ))
        )
        images_folder = os.path.join(self.raw_database_address, 'jpg')
        instances = [os.path.join(images_folder, folder_name) for folder_name in os.listdir(images_folder)]
        instances.sort()

        image_labels = loadmat(os.path.join(
            settings.PROJECT_ROOT_ADDRESS, 'databases', 'meta_dataset_meta', 'vggflowers', 'imagelabels.mat'
        ))['labels']
        image_labels = list(image_labels.reshape((-1, )))

        classes = dict()
        for instance, image_label in zip(instances, image_labels):
            if image_label not in classes:
                classes[image_label] = list()
            classes[image_label].append(instance)

        train_folders = {}
        val_folders = {}
        test_folders = {}

        for item in splits['train']:
            item = int(item[:3])
            train_folders[str(item)] = classes[item]
        for item in splits['valid']:
            item = int(item[:3])
            val_folders[str(item)] = classes[item]
        for item in splits['test']:
            item = int(item[:3])
            test_folders[str(item)] = classes[item]

        return train_folders, val_folders, test_folders


class TrafficSignDatabase(Database):
    def __init__(self, input_shape=(84, 84, 3)):
        super(TrafficSignDatabase, self).__init__(
            raw_database_address=settings.TRAFFIC_SIGN_RAW_DATASET_ADDRESS,
            database_address='',
            random_seed=-1,
            input_shape=input_shape
        )

    def get_train_val_test_folders(self) -> Tuple:
        """Returns train, val and test folders as three lists or three dictionaries.
        Note that the python random seed might have been
        set here based on the class __init__ function."""
        # splits = json.load(open(os.path.join(
        #     settings.PROJECT_ROOT_ADDRESS,
        #     'databases',
        #     'meta_dataset_meta',
        #     'splits',
        #     'traffic_sign.json'
        #     ))
        # )
        classes_folder = os.path.join(self.raw_database_address, 'GTSRB', 'Final_Training', 'Images')
        test_classes = [os.path.join(classes_folder, class_folder) for class_folder in os.listdir(classes_folder)]

        return [], [], test_classes

    def _get_parse_function(self) -> Callable:
        # TODO handle this image format
        def parse_function(example_address):
            image = tf.image.decode_ppg(tf.io.read_file(example_address))
            image = tf.image.resize(image, self.get_input_shape()[:2])
            image = tf.cast(image, tf.float32)

            return image / 255.

        return parse_function


class MSCOCODatabase(JPGParseMixin, Database):
    def __init__(self, input_shape=(84, 84, 3)):
        super(MSCOCODatabase, self).__init__(
            raw_database_address=settings.MSCOCO_RAW_DATASET_ADDRESS,
            database_address='',
            random_seed=-1,
            input_shape=input_shape
        )

    def fix_instances(self, img_instances):
        fixed_file_address = os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data', 'fixed_mscoco_bad_samples')
        if not os.path.exists(fixed_file_address):
            num_fixed, fixed_instances = convert_grayscale_images_to_rgb(img_instances)

            with open(fixed_file_address, 'w') as f:
                f.write(f'Changed {num_fixed} 2d data points to 3d.\n')
                f.write('\n'.join(fixed_instances))

    def get_train_val_test_folders(self) -> Tuple:
        # TODO add cropping to images
        """Returns train, val and test folders as three lists or three dictionaries.
        Note that the python random seed might have been
        set here based on the class __init__ function."""
        images_folder = os.path.join(self.raw_database_address, 'train2017')
        annotations_folder = os.path.join(self.raw_database_address, 'annotations_trainval2017', 'annotations')
        instances = json.load(open(os.path.join(annotations_folder, 'instances_train2017.json')))

        test_classes = dict()
        img_instances = list()
        instances_seen = set()

        for instance in instances['annotations']:
            if instance['image_id'] in instances_seen:
                continue
            instances_seen.add(instance['image_id'])
            category_id = str(instance['category_id'])
            instance_id = instance['image_id']
            if category_id not in test_classes:
                test_classes[category_id] = list()
            test_classes[category_id].append(os.path.join(images_folder, f'{instance_id:012d}.jpg'))
            img_instances.append(os.path.join(images_folder, f'{instance_id:012d}.jpg'))

        self.fix_instances(img_instances)

        return [], [], test_classes


class FungiDatabase(JPGParseMixin, Database):
    def __init__(self, input_shape=(84, 84, 3)):
        super(FungiDatabase, self).__init__(
            raw_database_address=settings.FUNGI_RAW_DATASET_ADDRESS,
            database_address='',
            random_seed=-1,
            input_shape=input_shape
        )

    def get_train_val_test_folders(self) -> Tuple:
        """Returns train, val and test folders as three lists or three dictionaries.
        Note that the python random seed might have been
        set here based on the class __init__ function."""
        images_folder = os.path.join(self.raw_database_address, 'fungi_train_val')
        splits = json.load(open(os.path.join(
            settings.PROJECT_ROOT_ADDRESS,
            'databases',
            'meta_dataset_meta',
            'splits',
            'fungi.json'
        )
        ))
        train_annotations = json.load(open(os.path.join(
            self.raw_database_address, 'train_val_annotations', 'train.json'))
        )
        val_annotations = json.load(open(os.path.join(self.raw_database_address, 'train_val_annotations', 'val.json')))
        images_list = train_annotations['images'] + val_annotations['images']
        id_to_filename = dict()
        for image in images_list:
            id_to_filename[image['id']] = image['file_name']
        annotations = train_annotations['annotations'] + val_annotations['annotations']

        classes = dict()
        for annotation in annotations:
            class_id = f'{annotation["category_id"]:04d}'
            image_file_name = id_to_filename[annotation['image_id']]
            if class_id not in classes:
                classes[class_id] = list()
            classes[class_id].append(os.path.join(images_folder, image_file_name))

        splits['train'] = set([item[:4] for item in splits['train']])
        splits['valid'] = set([item[:4] for item in splits['valid']])
        splits['test'] = set([item[:4] for item in splits['test']])

        train_folders = {class_id: classes[class_id] for class_id in classes if class_id in splits['train']}
        val_folders = {class_id: classes[class_id] for class_id in classes if class_id in splits['valid']}
        test_folders = {class_id: classes[class_id] for class_id in classes if class_id in splits['test']}

        return train_folders, val_folders, test_folders
