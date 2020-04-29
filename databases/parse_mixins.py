from typing import Callable

import tensorflow as tf


class JPGParseMixin(object):
    def _get_parse_function(self) -> Callable:
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, self.get_input_shape()[:2])
            image = tf.cast(image, tf.float32)

            return image / 255.

        return parse_function


class PNGParseMixin(object):
    def _get_parse_function(self) -> Callable:
        def parse_function(example_address):
            image = tf.image.decode_png(tf.io.read_file(example_address), channels=3)
            image = tf.image.resize(image, self.get_input_shape()[:2])
            image = tf.cast(image, tf.float32)
            return image / 255.

        return parse_function


class SameParseMixin(object):
    def _get_parse_function(self) -> Callable:
        def parse_function(example_address):
            return example_address

        return parse_function
