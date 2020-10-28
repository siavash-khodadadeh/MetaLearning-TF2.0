import tensorflow as tf


class SelectiveDropoutLayer(tf.keras.layers.Dropout):
    pass


class SimpleModelDropout(tf.keras.Model):
    name = 'SimpleModelDropout'

    def __init__(self, num_classes):
        super(SimpleModelDropout, self).__init__(name='simple_model_dropout')
        self.conv1 = tf.keras.layers.Conv2D(64, 3, name='conv1', strides=(2, 2), padding='same')
        self.do1 = SelectiveDropoutLayer(rate=0.1, name='do1')
        self.bn1 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, name='conv2', strides=(2, 2), padding='same')
        self.do2 = SelectiveDropoutLayer(rate=0.1, name='do2')
        self.bn2 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(64, 3, name='conv3', strides=(2, 2), padding='same')
        self.do3 = SelectiveDropoutLayer(rate=0.1, name='do3')
        self.bn3 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(64, 3, name='conv4',  strides=(2, 2), padding='same')
        self.do4 = SelectiveDropoutLayer(rate=0.1, name='do4')
        self.bn4 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn4')
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.dense = tf.keras.layers.Dense(num_classes, activation=None, name='dense')

    def conv_block(self, features, conv, bn, do, training=False):
        conv_out = conv(features)
        batch_normalized_out = bn(conv_out, training=training)
        do_out = do(batch_normalized_out, training=training)
        return tf.keras.activations.relu(do_out)

    def call(self, inputs, training=False):
        image = inputs
        c1 = self.conv_block(image, self.conv1, self.bn1, self.do1, training=training)
        c2 = self.conv_block(c1, self.conv2, self.bn2, self.do2, training=training)
        c3 = self.conv_block(c2, self.conv3, self.bn3, self.do3, training=training)
        c4 = self.conv_block(c3, self.conv4, self.bn4, self.do4, training=training)
        c4 = tf.reduce_mean(c4, [1, 2])
        f = self.flatten(c4)
        out = self.dense(f)

        return out
