import os
from typing import Tuple, List

import tensorflow as tf
import numpy as np
import pandas as pd

import settings

from .data_bases import Database
from .parse_mixins import JPGParseMixin, PNGParseMixin


class EuroSatDatabase(JPGParseMixin, Database):
    def __init__(self, input_shape=(84, 84, 3)):
        super(EuroSatDatabase, self).__init__(
            settings.EUROSAT_RAW_DATASET_ADDRESS,
            settings.EUROSAT_RAW_DATASET_ADDRESS,
            random_seed=-1,
            input_shape=input_shape
        )

    def get_train_val_test_folders(self) -> Tuple[List[str], List[str], List[str]]:
        base = os.path.join(self.database_address, '2750')
        folders = [os.path.join(base, folder_name) for folder_name in os.listdir(base)]

        return folders, folders, folders


class PlantDiseaseDatabase(JPGParseMixin, Database):
    def __init__(self, input_shape=(84, 84, 3)):
        super(PlantDiseaseDatabase, self).__init__(
            settings.PLANT_DISEASE_DATASET_ADDRESS,
            settings.PLANT_DISEASE_DATASET_ADDRESS,
            random_seed=-1,
            input_shape=input_shape
        )

    def get_train_val_test_folders(self) -> Tuple[List[str], List[str], List[str]]:
        train_base = os.path.join(self.database_address, 'dataset', 'train')
        test_base = os.path.join(self.database_address, 'dataset', 'test')
        train_folders = [os.path.join(train_base, folder_name) for folder_name in os.listdir(train_base)]
        test_folders = [os.path.join(test_base, folder_name) for folder_name in os.listdir(test_base)]

        return train_folders, test_folders, test_folders


class ISICDatabase(JPGParseMixin, Database):
    def __init__(self, input_shape=(84, 84, 3)):
        super(ISICDatabase, self).__init__(
            settings.ISIC_RAW_DATASET_ADDRESS,
            settings.ISIC_RAW_DATASET_ADDRESS,
            random_seed=-1,
            input_shape=input_shape
        )

    def get_train_val_test_folders(self):
        gt_file = os.path.join(
            self.database_address,
            'ISIC2018_Task3_Training_GroundTruth',
            'ISIC2018_Task3_Training_GroundTruth.csv'
        )
        content = pd.read_csv(gt_file)
        class_names = list(content.columns[1:])

        images = list(content.iloc[:, 0])

        labels = np.array(content.iloc[:, 1:])
        labels = np.argmax(labels, axis=1)

        classes = dict()
        for class_name in class_names:
            classes[class_name] = list()

        for image, label in zip(images, labels):
            classes[class_names[label]].append(
                os.path.join(self.database_address, 'ISIC2018_Task3_Training_Input', image + '.jpg')
            )

        return classes, classes, classes


class ChestXRay8Database(PNGParseMixin, Database):
    def __init__(self, input_shape=(84, 84, 3)):
        super(ChestXRay8Database, self).__init__(
            settings.CHESTX_RAY8_RAW_DATASET_ADDRESS,
            settings.CHESTX_RAY8_RAW_DATASET_ADDRESS,
            random_seed=-1,
            input_shape=input_shape
        )

    def get_train_val_test_folders(self):
        # At first we have to index the addresses with image names since ground truth does not have the path but just
        # the image
        images_paths = dict()

        for folder_name in os.listdir(os.path.join(self.database_address, 'data')):
            if os.path.isdir(os.path.join(self.database_address, 'data', folder_name)):
                base_address = os.path.join(self.database_address, 'data', folder_name)
                for item in os.listdir(os.path.join(base_address, 'images')):
                    images_paths[item] = os.path.join(base_address, 'images', item)

        gt_file = os.path.join(self.database_address, 'data', 'Data_Entry_2017.csv')
        class_names = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"
        ]

        content = pd.read_csv(gt_file)
        images = list(content.iloc[:, 0])

        labels = np.asarray(content.iloc[:, 1])

        classes = dict()
        for class_name in class_names:
            classes[class_name] = list()

        for image, label in zip(images, labels):
            label = label.split("|")
            if (
                    len(label) == 1 and
                    label[0] != "No Finding" and
                    label[0] != "Pneumonia" and
                    label[0] in class_names
            ):
                classes[label[0]].append(images_paths[image])

        return classes, classes, classes