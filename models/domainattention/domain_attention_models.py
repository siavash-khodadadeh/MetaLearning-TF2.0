import os
import tensorflow as tf
import numpy as np

from models.domainattention.domain_attention_networks import MiniImagenetModelForDomainAttention
from networks.maml_umtra_networks import MiniImagenetModel


class DomainAttentionModel(tf.keras.models.Model):
    def __init__(
        self,
        num_classes,
        train_dbs,
        db_encoder_epochs,
        db_encoder_lr,
        root,
        image_shape=(84, 84, 3),
        channel_wise_attention=True,
        element_wise_attention=False,
        dense_layer_sizes=None,
        *args,
        **kwargs
    ):
        super(DomainAttentionModel, self).__init__(*args, **kwargs)
        self.root = root
        self.train_dbs = train_dbs
        self.image_shape = image_shape
        self.db_encoder_epochs = db_encoder_epochs
        self.db_encoder_lr = db_encoder_lr
        self.feature_networks = []
        self.element_wise_attention = element_wise_attention
        self.channel_wise_attention = channel_wise_attention
        if dense_layer_sizes is not None:
            self.dense_layer_sizes = dense_layer_sizes
        else:
            self.dense_layer_sizes = list()
        self.perform_pre_training()

        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool')
        self.conv1 = tf.keras.layers.Conv2D(32, 3, name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, name='conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(32, 3, name='conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(32, 3, name='conv4')
        self.bn4 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn4')
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        if self.channel_wise_attention:
            self.channels_attention = list()
            for i in range(len(self.feature_networks)):
                self.channels_attention.append(
                    tf.keras.layers.Dense(
                        32,
                        activation='sigmoid',
                        name=f'channel_attention_{i}'
                    )
                )

        elif self.element_wise_attention:
            self.attention_network_dense = tf.keras.layers.Dense(
                len(self.feature_networks) * self.feature_size,
                activation='sigmoid',
                name='attention_network_dense'
            )
        else:
            self.attention_network_dense = tf.keras.layers.Dense(
                len(self.feature_networks),
                activation='softmax',
                name='attention_network_dense'
            )

        self.dense_layers = []
        i = 1
        for n in self.dense_layer_sizes:
            self.dense_layers.append(tf.keras.layers.Dense(n, activation='relu', name='classification_dense' + str(i)))
            i += 1
        self.dense_layers.append(tf.keras.layers.Dense(num_classes, activation=None, name='classification_dense' + str(i)))

    def conv_block(self, features, conv, bn=None, training=False):
        conv_out = conv(features)
        batch_normalized_out = bn(conv_out, training=training)
        batch_normalized_out = self.max_pool(batch_normalized_out)
        return tf.keras.activations.relu(batch_normalized_out)

    def call(self, inputs, training=None, mask=None):
        weights = self.conv_block(inputs, self.conv1, self.bn1, training=training)
        weights = self.conv_block(weights, self.conv2, self.bn2, training=training)
        weights = self.conv_block(weights, self.conv3, self.bn3, training=training)
        weights = self.conv_block(weights, self.conv4, self.bn4, training=training)
        weights = self.flatten(weights)
        feature_vectors = list()

        if self.channel_wise_attention:
            for i, feature_network in enumerate(self.feature_networks):
                c1 = feature_network.get_conv1_features(inputs, training=False)
                c2 = feature_network.get_conv2_features(c1, training=False)
                c3 = feature_network.get_conv3_features(c2, training=False)
                c4 = feature_network.get_conv4_features(c3, training=False)

                attention_features = self.channels_attention[i](weights)
                feature_vector = c4 * tf.reshape(attention_features, (attention_features.shape[0], 1, 1, 32))
                feature_vector = feature_network.forward_flatten(feature_vector)
                feature_vectors.append(feature_vector)

            feature_vectors = tf.stack(feature_vectors, axis=1)
            x = self.flatten(feature_vectors)

        else:
            weights = self.attention_network_dense(weights)

            for i, feature_network in enumerate(self.feature_networks):
                c1 = feature_network.get_conv1_features(inputs, training=False)
                c2 = feature_network.get_conv2_features(c1, training=False)
                c3 = feature_network.get_conv3_features(c2, training=False)
                c4 = feature_network.get_conv4_features(c3, training=False)
                feature_vector = feature_network.forward_flatten(c4)
                feature_vectors.append(feature_vector)

            feature_vectors = tf.stack(feature_vectors, axis=1)

            if self.element_wise_attention:
                x = tf.reshape(weights, feature_vectors.shape) * feature_vectors
                x = self.flatten(x)
                # x = tf.reshape(x, (x.shape[1], -1))
            else:
                x = tf.expand_dims(tf.transpose(weights), axis=2) * feature_vectors
                x = tf.reduce_sum(x, axis=0)

        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_attention_network(self):
        network = MiniImagenetModel(num_classes=len(self.feature_networks))
        input_layer = tf.keras.layers.Input(shape=self.image_shape, name='at_input')

        output = input_layer
        for layer in network.layers:
            output = layer(output)

        network = tf.keras.models.Model(input_layer, outputs=output, name='AttentionModel')
        return network

    def get_db_encoder(self, db_name, db_dataset, num_classes, db_index):
        network = MiniImagenetModelForDomainAttention(num_classes=num_classes, name=db_name + f'_{db_index}')
        network.predict(tf.zeros(shape=(1, *self.image_shape)))
        return network

    def get_parse_function(self):
        @tf.function
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, self.image_shape[:2])
            if tf.shape(image)[2] == 1:
                image = tf.squeeze(image, axis=2)
                image = tf.stack((image, image, image), axis=2)
            image = tf.cast(image, tf.float32)
            return image / 255.

        return parse_function

    def get_db_process_path(self, db, instance_to_class, class_ids):
        num_classes = len(db.train_folders)

        def process_path(file_path):
            def extract_label_from_file_path(fp):
                fp = str(fp.numpy(), 'utf-8')
                label = instance_to_class[fp]
                label = class_ids[label]
                label = tf.one_hot(label, depth=num_classes)
                return label

            label = tf.py_function(extract_label_from_file_path, inp=[file_path], Tout=tf.float32)
            image = self.get_parse_function()(file_path)
            return image, label

        return process_path

    def get_db_dataset(self, db):
        instances, instance_to_class, class_ids = db.get_all_instances(partition_name='train', with_classes=True)
        dataset = tf.data.Dataset.from_tensor_slices(instances)
        dataset = dataset.map(self.get_db_process_path(db, instance_to_class, class_ids))
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=len(instances))

        dataset = dataset.batch(64)

        return dataset

    def get_feature_networks(self):
        feature_networks = []
        db_index = 0
        for db in self.train_dbs:
            db_name = str(db.__class__)[str(db.__class__).rfind('.') + 1:-2]
            num_classes = len(db.train_folders)
            db_dataset = self.get_db_dataset(db)
            db_encoder = self.get_db_encoder(db_name, db_dataset, num_classes, db_index)

            db_saved_models = os.path.join(self.root, 'databases_info', db_name, 'saved_models/')
            latest_checkpoint = tf.train.latest_checkpoint(db_saved_models)
            initial_epoch = 0
            if latest_checkpoint is not None:
                initial_epoch = int(latest_checkpoint[latest_checkpoint.rfind('_') + 1:])
                db_encoder.load_weights(latest_checkpoint)

            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.root, 'databases_info', db_name, 'logs'),
                profile_batch=0
            )

            db_encoder.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.CategoricalAccuracy()],
                optimizer=tf.keras.optimizers.Adam(self.db_encoder_lr)
            )

            db_encoder.fit(
                db_dataset,
                epochs=self.db_encoder_epochs[db_index],
                callbacks=[tensorboard_callback],
                initial_epoch=initial_epoch
            )

            db_encoder.save_weights(
                filepath=os.path.join(db_saved_models, f'{db_name}_{self.db_encoder_epochs[db_index]}')
            )

            # db_encoder = tf.keras.models.Model(
            #     inputs=db_encoder.inputs,
            #     outputs=[db_encoder.get_layer('flatten').output]
            # )
            feature_networks.append(db_encoder)
            db_index += 1

        return feature_networks

    def perform_pre_training(self):
        feature_networks = self.get_feature_networks()
        for feature_network in feature_networks:
            feature_network.trainable = False
            self.feature_networks.append(feature_network)
        if self.element_wise_attention:
            input = tf.keras.layers.Input(shape=self.image_shape, name='at_input')
            feature_vector = feature_network.get_features(input, training=False)
            self.feature_size = feature_vector.shape[-1]
