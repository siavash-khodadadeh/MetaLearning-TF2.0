import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Flatten, Dense


class SimpleModelProto(tf.keras.Model):
    name = 'SimpleModelProto'

    def __init__(self):
        super(SimpleModelProto, self).__init__(name='simple_model')

        self.conv1 = tf.keras.layers.Conv2D(64, 3, name='conv1', strides=(2, 2), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, name='conv2', strides=(2, 2), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(64, 3, name='conv3', strides=(2, 2), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(64, 3, name='conv4',  strides=(2, 2), padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn4')
        self.flatten = Flatten(name='flatten')

    def conv_block(self, features, conv, bn=None, training=False):
        conv_out = conv(features)
        batch_normalized_out = bn(conv_out, training=training)
        return tf.keras.activations.relu(batch_normalized_out)

    def call(self, inputs, training=False):
        image = inputs
        c1 = self.conv_block(image, self.conv1, self.bn1, training=training)
        c2 = self.conv_block(c1, self.conv2, self.bn2, training=training)
        c3 = self.conv_block(c2, self.conv3, self.bn3, training=training)
        c4 = self.conv_block(c3, self.conv4, self.bn4, training=training)
        f = self.flatten(c4)

        return f


class MiniImagenetModelProto(tf.keras.Model):
    name = 'MiniImagenetModel'
    def __init__(self, *args, **kwargs):

        super(MiniImagenetModelProto, self).__init__(*args, **kwargs)
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

    def conv_block(self, features, conv, bn=None, training=False):
        conv_out = conv(features)
        batch_normalized_out = bn(conv_out, training=training)
        batch_normalized_out = self.max_pool(batch_normalized_out)
        return tf.keras.activations.relu(batch_normalized_out)

    def get_features(self, inputs, training=False):
        import numpy as np
        image = inputs
        c1 = self.conv_block(image, self.conv1, self.bn1, training=training)
        c2 = self.conv_block(c1, self.conv2, self.bn2, training=training)
        c3 = self.conv_block(c2, self.conv3, self.bn3, training=training)
        c4 = self.conv_block(c3, self.conv4, self.bn4, training=training)
        c4 = tf.reshape(c4, [-1, np.prod([int(dim) for dim in c4.get_shape()[1:]])])
        f = self.flatten(c4)
        return f

    def call(self, inputs, training=False):
        out = self.get_features(inputs, training=training)

        return out


class VGGSmallModel(tf.keras.models.Model):
    name = 'VGGSmallModel'

    def __init__(self):
        super(VGGSmallModel, self).__init__(name='vgg_small_model')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.conv1 = tf.keras.layers.Conv2D(64, 3, name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(128, 3, name='conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(256, 3, name='conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(256, 3, name='conv4')
        self.bn4 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn4')
        self.conv5 = tf.keras.layers.Conv2D(512, 3, name='conv5')
        self.bn5 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn5')
        self.conv6 = tf.keras.layers.Conv2D(512, 3, name='conv6')
        self.bn6 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn6')
        self.flatten = Flatten(name='flatten')

    def conv_block(self, features, conv, bn=None, training=False):
        conv_out = conv(features)
        batch_normalized_out = bn(conv_out, training=training)
        batch_normalized_out = self.max_pool(batch_normalized_out)
        return tf.keras.activations.relu(batch_normalized_out)

    def call(self, inputs, training=False):
        image = inputs
        output = self.conv_block(image, self.conv1, self.bn1, training=training)
        output = self.conv_block(output, self.conv2, self.bn2, training=training)
        output = self.conv_block(output, self.conv3, self.bn3, training=training)
        output = self.conv_block(output, self.conv4, self.bn4, training=training)
        output = self.conv_block(output, self.conv5, self.bn5, training=training)
        output = self.conv_block(output, self.conv6, self.bn6, training=training)
        output = self.flatten(output)

        return output
