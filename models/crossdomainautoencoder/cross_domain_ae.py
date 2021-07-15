import os
import sys

import tensorflow as tf
import numpy as np

from models.base_data_loader import BaseDataLoader
from networks.maml_umtra_networks import MiniImagenetModel
from utils import combine_first_two_axes


class CrossDomainAE(object):
    def __init__(self, database, batch_size, domains):
        self.database = database
        self.batch_size = batch_size
        self.encoder, self.decoder, self.autoencoder = self.init_autoencoder()
        self.domains = domains

        self.root = os.path.dirname(sys.argv[0])
        with tf.device('GPU:0'):
            self.domain_models = []
            for model_name in self.domains:
                checkpoint_path = os.path.join(self.root, f'saved_models/{model_name}_model.ckpt-60000')
                model = MiniImagenetModel(num_classes=5)
                model(tf.zeros(shape=(1, 84, 84, 3)))
                model.load_weights(checkpoint_path)
                model.trainable = False
                self.domain_models.append(model)

    def init_autoencoder(self):
        encoded_dim = 288

        encoder_inputs = tf.keras.Input(shape=(encoded_dim * 6, ))

        encoded = tf.keras.layers.Dense(encoded_dim, activation=None)(encoder_inputs)
        relu_encoded = tf.keras.layers.ReLU()(encoded)
        decoded = tf.keras.layers.Dense(encoded_dim * 6, activation=None)(relu_encoded)

        autoencoder = tf.keras.Model(inputs=encoder_inputs, outputs=decoded)

        encoder = tf.keras.Model(inputs=encoder_inputs, outputs=encoded)

        encoded_input = tf.keras.Input(shape=(encoded_dim, ))  # decoder inputs
        decoder_final_layer = autoencoder.layers[-1]
        decoder = tf.keras.Model(inputs=encoded_input, outputs=decoder_final_layer(encoded_input))
        return encoder, decoder, autoencoder

    def get_feature_parser(self):
        def f(x):
            with tf.device('GPU:0'):
                features = list()
                for domain_model in self.domain_models:
                    features.append(domain_model.get_features(x, apply_final_activation=False))
                return tf.concat(features, axis=1)

        return f

    def get_db_dataset(self, db):
        f = self.get_feature_parser()
        instances, instance_to_class, class_ids = db.get_all_instances(partition_name='train', with_classes=True)
        dataset = tf.data.Dataset.from_tensor_slices(instances)
        dataset = dataset.map(db._get_parse_function())
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(f)
        dataset = dataset.zip((dataset, dataset))

        return dataset

    def train(self, epochs=5):
        print('Training Domains: ')
        print(self.domains)
        print('Target domain: ')
        print(self.database.__class__)
        dataset = self.get_db_dataset(self.database)

        self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', loss_weights=[100])

        os.makedirs(os.path.join(self.root, 'cross_domain'), exist_ok=True)
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.root, 'cross_domain', f'model_{self.database.__class__}.h5'),
            save_weights_only=True
        )
        self.autoencoder.fit(dataset, epochs=epochs, shuffle=True, callbacks=[save_callback])

    def evaluate(self, iterations, num_tasks, k_test, k_val_test, inner_learning_rate, seed=-1):
        self.autoencoder.load_weights(os.path.join(self.root, 'cross_domain', f'model_{self.database.__class__}.h5'))

        data_loader = BaseDataLoader(
            database=self.database,
            val_database=self.database,
            test_database=self.database,
            n=5,
            k_ml=1,
            k_val_ml=1,
            k_val=1,
            k_val_val=15,
            k_test=k_test,
            k_val_test=k_val_test,
            meta_batch_size=1,
            num_tasks_val=100,
            val_seed=42
        )

        test_dataset = data_loader.get_test_dataset(num_tasks=num_tasks, seed=seed)
        print(self.database.__class__)
        print(f'K_test: {k_test}')
        print(f'K_test_val: {k_val_test}')

        accs = list()
        losses = list()

        counter = 0
        f = self.get_feature_parser()
        optimizer = tf.keras.optimizers.SGD(learning_rate=inner_learning_rate)

        for (train_ds, val_ds), (train_labels, val_labels) in test_dataset:
            train_ds = combine_first_two_axes(tf.squeeze(train_ds, axis=0))
            val_ds = combine_first_two_axes(tf.squeeze(val_ds, axis=0))
            train_labels = combine_first_two_axes(train_labels)
            val_labels = combine_first_two_axes(val_labels)

            remainder_num = num_tasks // 20
            if remainder_num == 0:
                remainder_num = 1
            if counter % remainder_num == 0:
                print(f'{counter} / {num_tasks} are evaluated.')

            counter += 1

            train_feats = f(train_ds)
            val_feats = f(val_ds)

            domain_train_feats = self.encoder(train_feats)
            rec_train_feats = self.decoder(tf.keras.activations.relu(domain_train_feats))
            print('here')
            # domain_train_feats = train_feats

            # tf.print(tf.reduce_max(domain_train_feats))
            # tf.print(tf.reduce_min(domain_train_feats))

            dense = tf.keras.layers.Dense(5, activation=None)
            for inner_step in range(iterations):
                with tf.GradientTape() as tape:
                    logits = dense(domain_train_feats)
                    loss = tf.reduce_mean(
                        tf.losses.categorical_crossentropy(train_labels, logits, from_logits=True)
                    )
                gradients = tape.gradient(loss, dense.trainable_variables)
                optimizer.apply_gradients(zip(gradients, dense.trainable_variables))

            domain_val_feats = self.encoder(val_feats)
            # domain_val_feats = val_feats
            val_logits = dense(domain_val_feats)
            val_loss = tf.reduce_mean(
                tf.losses.categorical_crossentropy(val_labels, val_logits, from_logits=True)
            )

            predicted_class_labels = tf.argmax(val_logits, axis=-1)
            real_val_labels = tf.argmax(val_labels, axis=-1)

            val_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_class_labels, real_val_labels), tf.float32))

            losses.append(val_loss)
            accs.append(val_acc)

        final_acc_mean = np.mean(accs)
        final_acc_std = np.std(accs)

        print(f'loss mean: {np.mean(losses)}')
        print(f'loss std: {np.std(losses)}')
        print(f'accuracy mean: {final_acc_mean}')
        print(f'accuracy std: {final_acc_std}')
        # Free the seed :D
        if seed != -1:
            np.random.seed(None)

        confidence_interval = 1.96 * final_acc_std / np.sqrt(num_tasks)

        print(
            f'final acc: {final_acc_mean} +- {confidence_interval}'
        )
        print(
            f'final acc: {final_acc_mean * 100:0.2f} +- {confidence_interval * 100:0.2f}'
        )
        return np.mean(accs)
