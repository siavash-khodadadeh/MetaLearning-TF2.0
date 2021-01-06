import os
from datetime import datetime
from abc import ABC, abstractmethod
import random
from typing import Tuple, Callable, Dict
import settings
import tensorflow as tf
import numpy as np
from .data_bases import Database
from .parse_mixins import JPGParseMixin

class MySubClass(JPGParseMixin,Database):
    def __init__(
            self,
            random_seed,
            num_train_classes,
            num_val_classes,
    ):
        self.num_train_classes = num_train_classes
        self.num_val_classes = num_val_classes
        super(MySubClass, self).__init__(
            settings.OMNIGLOT_RAW_DATA_ADDRESS,
            os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/'),
            random_seed=random_seed,
            input_shape=(28, 28, 4)
        )



    def get_train_val_test_folders(self) -> Tuple:
        damageTypes = list()
        myDir = "/data/yali/sam/Project/MetaLearning-TF2.0-master/data/Family/"
        for item in os.listdir(myDir):
            damageTypes.append(item)

        #print(damageTypes)
        damageImg = list()

        for damage in damageTypes:
            damageImg.append(os.path.join(myDir+damage))
        damageImg.sort()

        num_train_classes = self.num_train_classes
        num_val_classes = self.num_val_classes

        random.shuffle(damageImg)
        train_chars = damageImg[:num_train_classes]
        val_chars = damageImg[num_train_classes:num_train_classes + num_val_classes]
        test_chars = damageImg[num_train_classes + num_val_classes:]
        #print(train_chars)


        train_classes = {char: [os.path.join(char, instance) for instance in os.listdir(char)] for char in train_chars}
        val_classes = {char: [os.path.join(char, instance) for instance in os.listdir(char)] for char in val_chars}
        test_classes = {char: [os.path.join(char, instance) for instance in os.listdir(char)] for char in test_chars}
        return train_classes, val_classes, test_classes

    def _get_parse_function(self) -> Callable:
        def parse_function(example_address):
            image = tf.io.read_file(example_address)
            image = tf.io.decode_jpeg(image)
            image = tf.image.resize(image, (84, 84))
            image = image[:, :, :3]
            image = tf.image.rgb_to_grayscale(image)
            image = tf.cast(image, tf.float32)
            #tf.print(image.shape)
            return image / 255.
        return parse_function
