from abc import abstractmethod

import tensorflow as tf


class BaseParser(object):
    def __init__(self, shape):
        self.shape = shape

    @abstractmethod
    def get_parse_fn(self):
        pass


class OmniglotParser(BaseParser):
    def get_parse_fn(self):
        @tf.function
        def parse(example_address):
            image = tf.image.decode_png(tf.io.read_file(example_address))
            image = tf.reshape(tf.image.resize(image, self.shape[:2]), self.shape)
            image = tf.cast(image, tf.float32)

            return image / 255.

        return parse

