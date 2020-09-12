import os

from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import face_recognition

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import MiniImagenetModel
from databases import CelebADatabase
from utils import combine_first_two_axes, get_folders_with_greater_than_equal_k_files
import tensorflow_addons as tfa


class VGGSmallModel(tf.keras.models.Model):
    name = 'VGGSmallModel'

    def __init__(self, num_classes):
        super(VGGSmallModel, self).__init__(name='vgg_small_model')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.conv1 = tf.keras.layers.Conv2D(32, 3, name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, name='conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(32, 3, name='conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(32, 3, name='conv4')
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn4')
        self.conv5 = tf.keras.layers.Conv2D(32, 3, name='conv5')
        self.bn5 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn5')
        self.conv6 = tf.keras.layers.Conv2D(32, 3, name='conv6')
        self.bn6 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn6')
        self.flatten = tf.keras.layers.Flatten(name='flatten')

        self.dense = tf.keras.layers.Dense(num_classes, activation=None, name='dense')
        self.l2 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

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
        output = self.dense(output)
        # u_safe = tf.where(tf.abs(output) > 0.001, output, tf.ones_like(output))
        # output = self.l2(u_safe)
        output = self.l2(output)
        return output


class FaceRecognition(ModelAgnosticMetaLearningModel):
    def initialize_network(self):
        model = self.network_cls(num_classes=128)
        model(tf.zeros(shape=(self.n * self.k_ml, *self.database.input_shape)))
        return model

    def get_tf_dataset(self, folders):
        folders = get_folders_with_greater_than_equal_k_files(folders, 2)
        label_counter = 0
        items = []
        labels = []

        for folder in folders:
            folder_items = [os.path.join(folder, file_name) for file_name in os.listdir(folder)]
            for item in folder_items:
                items.append(item)
                labels.append(label_counter)
            label_counter += 1

        dataset = tf.data.Dataset.from_tensor_slices(items)
        dataset = dataset.map(self.database._get_parse_function())
        labels_dataset = dataset.from_tensor_slices(labels)
        dataset = tf.data.Dataset.zip((dataset, labels_dataset))
        dataset = dataset.shuffle(buffer_size=2048)
        dataset = dataset.batch(128)
        dataset = dataset.prefetch(1)

        # dataset = dataset.batch(2)
        # dataset = dataset.map(get_triplet)
        # dataset = dataset.map(_parse_func)
        # # dataset = dataset.unbatch()
        # dataset = dataset.batch(1)

        return dataset

    def get_root(self):
        return os.path.dirname(__file__)

    def train(self, iterations=5):
        tr_dataset = self.get_tf_dataset(self.database.train_folders)
        val_dataset = self.get_tf_dataset(self.database.val_folders)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            # loss=tfa.losses.ContrastiveLoss()
            loss=tfa.losses.TripletSemiHardLoss()
        )

        saver_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkpoint_dir, 'model.ckpt-{epoch:02d}-{val_loss:.2f}'),
            save_best_only=True,
            save_weights_only=True
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(self.train_log_dir)
        self.model.fit(
            tr_dataset,
            validation_data=val_dataset,
            epochs=iterations,
            callbacks=[saver_callback, tensorboard_callback]
        )

    def euclidean_distance(self, a, b):
        # a.shape = N x D
        # b.shape = M x D
        N, D = tf.shape(a)[0], tf.shape(a)[1]
        M = tf.shape(b)[0]
        a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
        b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
        return tf.reduce_mean(tf.square(a - b), axis=2)

    def load_model(self, iterations=None, acc=None):
        iteration_count = 0
        if iterations is not None:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model.ckpt-{iterations}-{acc}')
            iteration_count = iterations
        else:
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)

        if checkpoint_path is not None:
            try:
                self.model.load_weights(checkpoint_path)
                print(f'==================\nResuming Training\n======={iteration_count}=======\n==================')
            except Exception as e:
                print('Could not load the previous checkpoint!')

        else:
            print('No previous checkpoint found!')

        return iteration_count

    def evaluate(self, iterations, iterations_to_load_from=None, seed=-1, num_tasks=1000):
        # self.load_model(iterations=51, acc='0.78')
        # self.load_model()
        # model = load_model('facenet_keras.h5')
        model = load_model('/home/siavash/PycharmProjects/facenet/model/facenet_keras.h5')
        print(model.inputs)
        print(model.outputs)
        model.trainable = False
        self.model = model
        self.model.summary()
        # dense = tf.keras.layers.Dense(self.n, activation='softmax')(model.output)
        # self.model = tf.keras.models.Model(inputs=model.inputs, outputs=dense)
        self.test_dataset = self.get_test_dataset(seed=seed, num_tasks=num_tasks)

        task_number = 0
        accs = list()
        for (train_ds, val_ds), (train_labels, val_labels) in self.test_dataset:
            task_number += 1
            if task_number % (num_tasks // 20) == 0:
                print(f'{task_number} finished.')
            train_labels = combine_first_two_axes(train_labels)
            val_labels = combine_first_two_axes(val_labels)

            train_labels = tf.math.argmax(train_labels, axis=1)
            val_labels = tf.math.argmax(val_labels, axis=1)

            train_ds = tf.squeeze(train_ds, axis=0)
            train_ds = combine_first_two_axes(train_ds)
            encodings = self.model.predict(train_ds)

            from datetime import datetime
            begin = datetime.now()
            val_ds = tf.squeeze(val_ds, axis=0)
            val_ds = combine_first_two_axes(val_ds)
            val_encodings = self.model.predict(val_ds)
            dists = self.euclidean_distance(val_encodings, encodings)
            # predicted_labels = train_labels[tf.argmin(dists, axis=1)]
            predicted_labels = train_labels.numpy()[tf.argmin(dists, axis=1).numpy()]
            end = datetime.now()
            print(end - begin)

            task_final_accuracy = np.mean(predicted_labels == val_labels)
            accs.append(task_final_accuracy)

        print(f'accuracy mean: {np.mean(accs)}')
        print(f'accuracy std: {np.std(accs)}')
        print(
            f'final acc: {np.mean(accs)} +- {1.96 * np.std(accs) / np.sqrt(num_tasks)}'
        )
        return np.mean(accs)

    def evaluate_with_original_face_recognition(self, iterations, iterations_to_load_from=None, seed=-1, num_tasks=1000):
        self.test_dataset = self.get_test_dataset(seed=seed, num_tasks=num_tasks)

        accs = list()
        counter = 0

        for (train_ds, val_ds), (train_labels, val_labels) in self.test_dataset:
            if counter % 50 == 0:
                print(f'{counter} / {num_tasks} are evaluated.')
            counter += 1
            train_labels = combine_first_two_axes(train_labels)
            val_labels = combine_first_two_axes(val_labels)

            train_labels = tf.math.argmax(train_labels, axis=1)
            val_labels = tf.math.argmax(val_labels, axis=1)

            train_ds = tf.squeeze(train_ds, axis=0)
            train_ds = combine_first_two_axes(train_ds)
            val_ds = tf.squeeze(val_ds, axis=0)
            val_ds = combine_first_two_axes(val_ds)

            encodings = []
            labels = []
            for image, label in zip(train_ds, train_labels):
                image *= 255
                image = image.numpy().astype(np.uint8)
                encoding = face_recognition.face_encodings(image)
                if len(encoding) > 0:
                    encoding = encoding[0]
                    encodings.append(encoding)
                else:
                    print('bad')
                    encodings.append([0] * 128)

                labels.append(label)

            true_count = 0
            all_count = 0
            for image, label in zip(val_ds, val_labels):
                image *= 255
                image = image.numpy().astype(np.uint8)
                encoding = face_recognition.face_encodings(image)
                if len(encoding) > 0:
                    encoding = encoding[0]
                    face_distances = face_recognition.face_distance(encodings, encoding)

                    best_match_index = np.argmin(face_distances)
                    if labels[best_match_index] == label:
                        true_count += 1

                all_count += 1

            task_final_accuracy = true_count / all_count
            accs.append(task_final_accuracy)

        print(f'accuracy mean: {np.mean(accs)}')
        print(f'accuracy std: {np.std(accs)}')
        print(
            f'final acc: {np.mean(accs)} +- {1.96 * np.std(accs) / np.sqrt(num_tasks)}'
        )
        return np.mean(accs)


def run_celeba():
    celeba_database = CelebADatabase(input_shape=(224, 224, 3))
    # celeba_database = MiniImagenetDatabase(input_shape=(224, 224, 3))
    # for facenet
    # celeba_database = CelebADatabase(input_shape=(160, 160, 3))
    # celeba_database = LFWDatabase(input_shape=(224, 224, 3))
    maml = FaceRecognition(
        database=celeba_database,
        network_cls=MiniImagenetModel,
        # network_cls=VGGSmallModel,
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val=1,
        k_val_val=15,
        k_test=1,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.01,
        num_steps_validation=5,
        save_after_iterations=20,
        meta_learning_rate=0.001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='celeba'
    )

    # maml.train(iterations=100)
    maml.evaluate(5, seed=42)
    # maml.evaluate_with_original_face_recognition(5, seed=42)


if __name__ == '__main__':
    # tf.config.set_visible_devices([], 'GPU')
    # run_omniglot()
    # run_mini_imagenet()
    run_celeba()
