import tensorflow as tf
import time

from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input


class SimpleModel(tf.keras.Model):
    def __init__(self, weights=None, *args, **kwargs):
        super(SimpleModel, self).__init__(name='simple_model')

        self.conv1 = tf.keras.layers.Conv2D(64, 3, name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, name='conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(64, 3, name='conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(64, 3, name='conv4')
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn4')
        self.flatten = Flatten()
        self.dense = Dense(kwargs['num_classes'], activation=None, name='dense')

    def conv_block(self, conv, bn, features, training):
        conv_out = conv(features)
        batch_normalized_out = bn(conv_out, training=training)
        return tf.keras.activations.relu(batch_normalized_out)

    def call(self, inputs, training=False):
        image = inputs
        c1 = self.conv_block(self.conv1, self.bn1, image, training=training)
        c2 = self.conv_block(self.conv2, self.bn2, c1, training=training)
        c3 = self.conv_block(self.conv3, self.bn3, c2, training=training)
        c4 = self.conv_block(self.conv4, self.bn4, c3, training=training)
        c4 = tf.reduce_mean(c4, [1, 2])
        f = self.flatten(c4)
        out = self.dense(f)

        return out
