import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, BatchNormalization


class SimpleModel(tf.keras.Model):
    name = 'SimpleModel'

    def __init__(self, num_classes):
        super(SimpleModel, self).__init__(name='simple_model')

        self.conv1 = tf.keras.layers.Conv2D(64, 3, name='conv1', strides=(2, 2), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, name='conv2', strides=(2, 2), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(64, 3, name='conv3', strides=(2, 2), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(64, 3, name='conv4',  strides=(2, 2), padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn4')
        self.flatten = Flatten(name='flatten')
        self.dense = Dense(num_classes, activation=None, name='dense')

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
        c4 = tf.reduce_mean(c4, [1, 2])
        f = self.flatten(c4)
        out = self.dense(f)

        return out


class MiniImagenetModel(tf.keras.Model):
    name = 'MiniImagenetModel'

    def __init__(self, num_classes):
        super(MiniImagenetModel, self).__init__(name='mini_imagenet_model')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.conv1 = tf.keras.layers.Conv2D(32, 3, name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn1')
        # self.bn1 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, name='conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn2')
        # self.bn2 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(32, 3, name='conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn3')
        # self.bn3 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(32, 3, name='conv4')
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn4')
        # self.bn4 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn4')
        self.flatten = Flatten(name='flatten')

        self.dense = Dense(num_classes, activation=None, name='dense')

    def conv_block(self, features, conv, bn=None, training=False):
        conv_out = conv(features)
        batch_normalized_out = bn(conv_out, training=training)
        batch_normalized_out = self.max_pool(batch_normalized_out)
        return tf.keras.activations.relu(batch_normalized_out)

    def call(self, inputs, training=False):
        import numpy as np
        image = inputs
        c1 = self.conv_block(image, self.conv1, self.bn1, training=training)
        c2 = self.conv_block(c1, self.conv2, self.bn2, training=training)
        c3 = self.conv_block(c2, self.conv3, self.bn3, training=training)
        c4 = self.conv_block(c3, self.conv4, self.bn4, training=training)
        c4 = tf.reshape(c4, [-1, np.prod([int(dim) for dim in c4.get_shape()[1:]])])
        f = self.flatten(c4)
        out = self.dense(f)

        return out


class VGGSmallModel(tf.keras.models.Model):
    name = 'VGGSmallModel'

    def __init__(self, num_classes):
        super(VGGSmallModel, self).__init__(name='vgg_small_model')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.conv1 = tf.keras.layers.Conv2D(64, 3, name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(128, 3, name='conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(256, 3, name='conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(256, 3, name='conv4')
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn4')
        self.conv5 = tf.keras.layers.Conv2D(512, 3, name='conv5')
        self.bn5 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn5')
        self.conv6 = tf.keras.layers.Conv2D(512, 3, name='conv6')
        self.bn6 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn6')
        self.flatten = Flatten(name='flatten')

        self.dense1 = tf.keras.layers.Dense(32, activation=None, name='dense1')
        self.bn_dense = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn_dense')
        self.dense = Dense(num_classes, activation=None, name='dense')

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
        output = self.dense1(output)
        output = self.bn_dense(output)
        output = tf.keras.activations.relu(output)
        output = self.dense(output)
        return output


class VGG19Model(tf.keras.models.Model):
    name = 'VGG19Model'

    def __init__(self, num_classes):
        super(VGG19Model, self).__init__(name='vgg19_model')
        self.block1_conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',  padding='same', name='block1_conv1')
        self.block1_conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')
        self.block1_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

        self.block2_conv1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')
        self.block2_conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')
        self.block2_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

        self.block3_conv1 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')
        self.block3_conv2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')
        self.block3_conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')
        self.block3_conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')
        self.block3_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')

        self.block4_conv1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')
        self.block4_conv2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')
        self.block4_conv3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')
        self.block4_conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')
        self.block4_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')

        self.block5_conv1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')
        self.block5_conv2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')
        self.block5_conv3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')
        self.block5_conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')
        self.block5_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')

        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.fc1 = tf.keras.layers.Dense(4096, activation='relu', name='fc1')
        self.fc2 = tf.keras.layers.Dense(4096, activation='relu', name='fc2')
        self.fc3 = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')

    def call(self, inputs, training=False):
        image = inputs
        output = self.block1_conv1(image)
        output = self.block1_conv2(output)
        output = self.block1_pool(output)
        output = self.block2_conv1(output)
        output = self.block2_conv2(output)
        output = self.block2_pool(output)
        output = self.block3_conv1(output)
        output = self.block3_conv2(output)
        output = self.block3_conv3(output)
        output = self.block3_conv4(output)
        output = self.block3_pool(output)
        output = self.block4_conv1(output)
        output = self.block4_conv2(output)
        output = self.block4_conv3(output)
        output = self.block4_conv4(output)
        output = self.block4_pool(output)
        output = self.block5_conv1(output)
        output = self.block5_conv2(output)
        output = self.block5_conv3(output)
        output = self.block5_conv4(output)
        output = self.block5_pool(output)
        output = self.flatten(output)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output
