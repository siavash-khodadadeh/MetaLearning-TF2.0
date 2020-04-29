import os
import random
from typing import Tuple, List

import settings

from .data_bases import Database
from .parse_mixins import JPGParseMixin


class OmniglotDatabase(JPGParseMixin, Database):
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
            input_shape=(28, 28, 1)
        )

    def get_train_val_test_folders(self) -> Tuple:
        alphabets = list()
        for item in ('images_background', 'images_evaluation'):
            base_address = os.path.join(self.raw_database_address, item)
            alphabets.extend([os.path.join(base_address, alphabet) for alphabet in os.listdir(base_address)])

        characters = list()

        for alphabet in alphabets:
            characters.extend([os.path.join(alphabet, character_folder) for character_folder in os.listdir(alphabet)])
        characters.sort()

        num_train_classes = self.num_train_classes
        num_val_classes = self.num_val_classes

        random.shuffle(characters)
        train_chars = characters[:num_train_classes]
        val_chars = characters[num_train_classes:num_train_classes + num_val_classes]
        test_chars = characters[num_train_classes + num_val_classes:]

        train_classes = {char: [os.path.join(char, instance) for instance in os.listdir(char)] for char in train_chars}
        val_classes = {char: [os.path.join(char, instance) for instance in os.listdir(char)] for char in val_chars}
        test_classes = {char: [os.path.join(char, instance) for instance in os.listdir(char)] for char in test_chars}

        return train_classes, val_classes, test_classes


class MiniImagenetDatabase(JPGParseMixin, Database):
    def __init__(self, input_shape=(84, 84, 3)):
        super(MiniImagenetDatabase, self).__init__(
            settings.MINI_IMAGENET_RAW_DATA_ADDRESS,
            os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/mini-imagenet'),
            random_seed=-1,
            input_shape=input_shape
        )

    def get_train_val_test_folders(self) -> Tuple[List[str], List[str], List[str]]:
        dataset_folders = list()
        for dataset_type in ('train', 'val', 'test'):
            dataset_base_address = os.path.join(self.raw_database_address, dataset_type)
            folders = [
                os.path.join(dataset_base_address, class_name) for class_name in os.listdir(dataset_base_address)
            ]
            dataset_folders.append(folders)
        return dataset_folders[0], dataset_folders[1], dataset_folders[2]

