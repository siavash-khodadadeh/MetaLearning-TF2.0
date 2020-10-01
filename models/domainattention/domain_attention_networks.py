import tensorflow as tf
import numpy as np


class MiniImagenetModelForDomainAttention(tf.keras.Model):
    def __init__(self, num_classes, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'MiniImagenetModel'

        super(MiniImagenetModelForDomainAttention, self).__init__(*args, **kwargs)
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.conv1 = tf.keras.layers.Conv2D(32, 3, name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn1')
        # self.bn1 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, name='conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn2')
        # self.bn2 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(32, 3, name='conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn3')
        # self.bn3 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(32, 3, name='conv4')
        self.bn4 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn4')
        # self.bn4 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn4')
        self.flatten = tf.keras.layers.Flatten(name='flatten')

        self.dense = tf.keras.layers.Dense(num_classes, activation=None, name='dense')

    def conv_block(self, features, conv, bn=None, training=False):
        conv_out = conv(features)
        batch_normalized_out = bn(conv_out, training=training)
        batch_normalized_out = self.max_pool(batch_normalized_out)
        return tf.keras.activations.relu(batch_normalized_out)

    def get_conv1_features(self, inputs, training=False):
        image = inputs
        c1 = self.conv_block(image, self.conv1, self.bn1, training=training)
        return c1

    def get_conv2_features(self, c1, training=False):
        c2 = self.conv_block(c1, self.conv2, self.bn2, training=training)
        return c2

    def get_conv3_features(self, c2, training=False):
        c3 = self.conv_block(c2, self.conv3, self.bn3, training=training)
        return c3

    def get_conv4_features(self, c3, training=False):
        c4 = self.conv_block(c3, self.conv4, self.bn4, training=training)
        return c4

    def forward_flatten(self, features):
        return self.flatten(features)

    def forward_dense(self, features):
        return self.dense(features)

    def get_features(self, inputs, training=False):
        c1 = self.get_conv1_features(inputs, training=training)
        c2 = self.get_conv2_features(c1, training=training)
        c3 = self.get_conv3_features(c2, training=training)
        c4 = self.get_conv4_features(c3, training=training)
        c4 = tf.reshape(c4, [-1, np.prod([int(dim) for dim in c4.get_shape()[1:]])])
        f = self.flatten(c4)
        return f

    def call(self, inputs, training=False):
        f = self.get_features(inputs, training=training)
        out = self.dense(f)

        return out
