import tensorflow as tf
import numpy as np
from networks.maml_umtra_networks import SimpleModel, MiniImagenetModel


class VariationalAutoEncoderFeature(tf.keras.models.Model):
    def __init__(self, input_shape, latent_dim, n_classes):
        super(VariationalAutoEncoderFeature, self).__init__(name='vae')
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.encoder = tf.keras.Sequential(
            (
                tf.keras.layers.InputLayer(input_shape=input_shape),
                # conv block 1
                tf.keras.layers.Conv2D(32, 3, activation=None, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                # conv block 2
                tf.keras.layers.Conv2D(32, 3, activation=None, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                # conv block 3
                tf.keras.layers.Conv2D(32, 3, activation=None, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                # conv block 4
                tf.keras.layers.Conv2D(32, 3, activation=None, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                # Flatten
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim, activation=None)
            )
        )

        self.encoder_dense = tf.keras.layers.Dense(latent_dim + latent_dim, activation=None)
        self.classification_dense = tf.keras.layers.Dense(n_classes, activation=tf.keras.activations.softmax)

        self.decoder = tf.keras.Sequential(
            (
                tf.keras.layers.InputLayer(input_shape=(latent_dim, )),
                tf.keras.layers.Dense(3 * 3 * 32, activation='relu'),
                tf.keras.layers.Reshape((3, 3, 32)),
                # deconv block 1
                tf.keras.layers.Conv2DTranspose(32, 3, activation=None, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                # deconv block 2
                tf.keras.layers.Conv2DTranspose(32, 3, activation=None, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.UpSampling2D(3),
                # deconv block 3
                tf.keras.layers.Conv2DTranspose(32, 3, activation=None, use_bias=False, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.UpSampling2D(2),
                # deconv block 4
                tf.keras.layers.Conv2DTranspose(3, 3, activation=None, use_bias=False, padding='same'),
                tf.keras.layers.UpSampling2D(2),
            )
        )

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        encoder_output = self.encoder_dense(self.encoder(x))
        mean, logvar = tf.split(encoder_output, num_or_size_splits=2, axis=1)
        return mean, logvar

    def classify(self, x):
        encoder_output = self.encoder_dense(self.encoder(x))
        y_hat = self.classification_dense(encoder_output)
        return y_hat

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return mean + eps * tf.exp(logvar * 0.5)

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

    def compute_vae_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def compute_classification_loss(self, x, y):
        logits = self.classify(x)
        cross_ent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        classification_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits), tf.argmax(y)), tf.float32))
        return cross_ent, classification_accuracy


class SimpleModelFeature(SimpleModel):
    def call(self, inputs, training=False):
        image = inputs
        c1 = self.conv_block(image, self.conv1, self.bn1, training=training)
        c2 = self.conv_block(c1, self.conv2, self.bn2, training=training)
        c3 = self.conv_block(c2, self.conv3, self.bn3, training=training)
        c4 = self.conv_block(c3, self.conv4, self.bn4, training=training)
        f = self.flatten(c4)
        out = self.dense(f)

        return out

    def get_sequential_model(self):
        x = tf.keras.layers.Input(shape=(28, 28, 1))
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))


class MiniImagenetFeature(MiniImagenetModel):
    def __init__(self, num_classes):
        super(MiniImagenetFeature, self).__init__(num_classes)
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.conv1 = tf.keras.layers.Conv2D(32, 3, name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, name='conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(32, 3, name='conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(32, 3, name='conv4')
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn4')

        self.flatten = Flatten(name='flatten')

        self.dense = Dense(num_classes, activation=None, name='dense')

    def call(self, inputs, training=False):
        image = inputs
        c1 = self.conv_block(image, self.conv1, self.bn1, training=training)
        c2 = self.conv_block(c1, self.conv2, self.bn2, training=training)
        c3 = self.conv_block(c2, self.conv3, self.bn3, training=training)
        c4 = self.conv_block(c3, self.conv4, self.bn4, training=training)
        f = self.flatten(c4)
        out = self.dense(f)

        return out

    def get_sequential_model(self):
        x = tf.keras.layers.Input(shape=(28, 28, 1))
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))