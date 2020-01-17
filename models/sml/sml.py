import os
import pickle

import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks import SimpleModel, MiniImagenetModel, SimpleModelFeature, MiniImagenetFeature
from tf_datasets import OmniglotDatabase, MiniImagenetDatabase, CelebADatabase


class SML(ModelAgnosticMetaLearningModel):
    def __init__(
            self,
            database,
            network_cls,
            n,
            k,
            meta_batch_size,
            num_steps_ml,
            lr_inner_ml,
            num_steps_validation,
            save_after_epochs,
            meta_learning_rate,
            report_validation_frequency,
            log_train_images_after_iteration,  # Set to -1 if you do not want to log train images.
            n_clusters,
            feature_model,
            feature_size,
            input_shape,
            preprocess_function=None,
            least_number_of_tasks_val_test=-1,  # Make sure the val and test dataset pick at least this many tasks.
            clip_gradients=False,
            experiment_name=None,
    ):
        super(SML, self).__init__(
            database=database,
            network_cls=network_cls,
            n=n,
            k=k,
            meta_batch_size=meta_batch_size,
            num_steps_ml=num_steps_ml,
            lr_inner_ml=lr_inner_ml,
            num_steps_validation=num_steps_validation,
            save_after_epochs=save_after_epochs,
            meta_learning_rate=meta_learning_rate,
            report_validation_frequency=report_validation_frequency,
            log_train_images_after_iteration=log_train_images_after_iteration,
            least_number_of_tasks_val_test=least_number_of_tasks_val_test,
            clip_gradients=clip_gradients,
            experiment_name=experiment_name
        )
        self.features_model = feature_model
        self.n_clusters = n_clusters
        self.feature_size = feature_size
        self.input_shape = input_shape
        self.preprocess_fn = preprocess_function

    def get_root(self):
        return os.path.dirname(__file__)

    def get_features(self, dir_name=None):
        files_names_address = os.path.join(dir_name, 'file_names.npy')
        features_address = os.path.join(dir_name, 'features.npy')

        if dir_name is not None and os.path.exists(dir_name):
            return np.load(features_address), np.load(files_names_address)

        all_files = list()

        for class_name in self.database.train_folders:
            all_files.extend([os.path.join(class_name, file_name) for file_name in os.listdir(class_name)])

        n = len(all_files)
        m = self.feature_size
        features = np.zeros(shape=(n, m))

        for index, sampled_file in enumerate(all_files):
            if index % 1000 == 0:
                print(f'{index}/{len(all_files)} images loaded.')

            img = tf.keras.preprocessing.image.load_img(sampled_file, target_size=self.input_shape)
            img = tf.keras.preprocessing.image.img_to_array(img)
            if self.input_shape[2] == 1:
                img = np.expand_dims(img[:, :, 0], axis=-1)

            img = np.expand_dims(img, axis=0)
            if self.preprocess_fn is not None:
                img = self.preprocess_fn(img)

            features[index, :] = self.features_model.predict(img).reshape(-1)

        if dir_name is not None:
            os.makedirs(dir_name)
            np.save(files_names_address, all_files)
            np.save(features_address, features)
        return features, all_files

    def get_train_dataset(self):
        clusters_files_dir = os.path.join(self.get_root(), f'cache/{self.experiment_name}', 'clusters')
        if not os.path.exists(clusters_files_dir):
            features, all_files = self.get_features(
                dir_name=os.path.join(self.get_root(), f'cache/{self.experiment_name}')
            )

            k_means_path = os.path.join(self.get_root(), f'cache/{self.experiment_name}', 'k_means.pkl')
            if os.path.exists(k_means_path):
                k_means = pickle.load(open(k_means_path, 'rb'))
                cluster_ids = k_means.predict(features)
            else:
                k_means = KMeans(n_clusters=self.n_clusters)
                cluster_ids = k_means.fit_predict(features)
                with open(k_means_path, 'wb') as f:
                    pickle.dump(k_means, f)

            clusters = dict()
            for i, file_address in enumerate(all_files):
                cluster_index = cluster_ids[i]
                if cluster_index not in clusters.keys():
                    clusters[cluster_index] = list()

                clusters[cluster_index].append(file_address)

            # drop clusters with small numbers
            final_clusters = dict()
            counter = 0
            for i in range(self.n_clusters):
                if i in clusters:
                    if len(clusters[i]) >= 2 * self.k:
                        final_clusters[counter] = clusters[i]
                        counter += 1

            os.makedirs(clusters_files_dir)
            for cluster_id, cluster_instances in final_clusters.items():
                with open(os.path.join(clusters_files_dir, str(cluster_id) + '.txt'), 'w') as cluster_file:
                    for cluster_instance in cluster_instances:
                        cluster_file.write(cluster_instance)
                        cluster_file.write('\n')

        return self.database.get_meta_learning_dataset_from_clusters(
            clusters_dir=clusters_files_dir,
            n=self.n,
            k=self.k,
            meta_batch_size=self.meta_batch_size
        )

    def get_config_info(self):
        config_str = f'sml_' \
               f'model-{self.network_cls.name}_' \
               f'mbs-{self.meta_batch_size}_' \
               f'n-{self.n}_' \
               f'k-{self.k}_' \
               f'stp-{self.num_steps_ml}'
        if self.experiment_name != '':
            config_str += '_' + self.experiment_name

        return config_str


def sample_data_points(classes_dirs, n_samples):
    instances = list()
    labels = list()

    for class_dir in classes_dirs:
        class_instances = [os.path.join(class_dir, file_name) for file_name in os.listdir(class_dir)]
        instances.extend(class_instances)
        class_dir = class_dir[class_dir.rfind('/') + 1:]
        labels.extend([class_dir] * len(class_instances))

    instances = np.array(instances)
    labels = np.array(labels)

    indices = list(range(len(instances)))
    selected_indices = np.random.choice(indices, n_samples, replace=False)
    not_selected_indices = np.array([index for index in indices if index not in selected_indices])

    return instances[selected_indices], labels[selected_indices], instances[not_selected_indices]


def make_features_dataset_mini_imagenet(data, labels, non_labeled_data):
    label_values = np.unique(labels)
    depth = len(label_values)
    label_to_index = {label_value: i for i, label_value in enumerate(label_values)}
    labels = [label_to_index[label] for label in labels]

    x_dataset = tf.data.Dataset.from_tensor_slices(data)
    y_dataset = tf.data.Dataset.from_tensor_slices(labels)

    def _parse_image_label_pair(image_file, label):
        image = tf.image.decode_jpeg(tf.io.read_file(image_file))
        image = tf.image.resize(image, (84, 84))
        image = tf.cast(image, tf.float32)

        label = tf.one_hot(label, depth=depth)
        return image / 255., label

    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
    dataset = dataset.map(_parse_image_label_pair)

    dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.batch(32, drop_remainder=True)

    return dataset, depth


def make_features_dataset_omniglot(data, labels, non_labeled_data):
    label_values = np.unique(labels)
    depth = len(label_values)
    label_to_index = {label_value: i for i, label_value in enumerate(label_values)}
    labels = [label_to_index[label] for label in labels]

    x_dataset = tf.data.Dataset.from_tensor_slices(data)
    y_dataset = tf.data.Dataset.from_tensor_slices(labels)

    def _parse_image_label_pair(image_file, label):
        image = tf.image.decode_jpeg(tf.io.read_file(image_file))
        image = tf.image.resize(image, (28, 28))
        image = tf.cast(image, tf.float32)

        label = tf.one_hot(label, depth=depth)
        return 1 - (image / 255.), label

    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
    dataset = dataset.map(_parse_image_label_pair)

    dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.batch(32, drop_remainder=True)

    return dataset, depth


def train_the_feature_model(sequential_model, dataset, n_classes, input_shape):
    features = sequential_model.layers[-2].output
    y_hat = tf.keras.layers.Dense(n_classes, activation='softmax')(features)

    model = tf.keras.models.Model(inputs=sequential_model.input, outputs=y_hat)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # TODO does this reset dataset at each iteration and shuffle it randomly
    model.fit(dataset, epochs=90)
    model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    return model


def run_omniglot():
    omniglot_database = OmniglotDatabase(
        random_seed=30,
        num_train_classes=1200,
        num_val_classes=100,
    )
    n_data_points = 10000

    data_points, classes, non_labeled_data_points = sample_data_points(omniglot_database.train_folders, n_data_points)
    features_dataset, n_classes = make_features_dataset_omniglot(data_points, classes, non_labeled_data_points)
    feature_model = SimpleModelFeature(num_classes=5).get_sequential_model()
    feature_model = train_the_feature_model(feature_model, features_dataset, n_classes, omniglot_database.input_shape)

    sml = SML(
        database=omniglot_database,
        network_cls=SimpleModel,
        n=5,
        k=1,
        meta_batch_size=32,
        num_steps_ml=5,
        lr_inner_ml=0.01,
        num_steps_validation=5,
        save_after_epochs=5,
        meta_learning_rate=0.001,
        n_clusters=1200,
        feature_model=feature_model,
        feature_size=256,
        input_shape=(28, 28, 1),
        log_train_images_after_iteration=10,
        report_validation_frequency=10,
        experiment_name='omniglot_simple_model_feature_10000'
    )

    # sml.train(epochs=101)
    # sml.evaluate(iterations=50)


def run_mini_imagenet():
    mini_imagenet_database = MiniImagenetDatabase(random_seed=-1)
    # n_data_points = 10000
    # data_points, classes, non_labeled_data_points = sample_data_points(
    #     mini_imagenet_database.train_folders,
    #     n_data_points
    # )
    # features_dataset, n_classes = make_features_dataset_mini_imagenet(data_points, classes, non_labeled_data_points)
    # feature_model = MiniImagenetFeature(num_classes=5).get_sequential_model()
    # feature_model = train_the_feature_model(
    #     feature_model,
    #     features_dataset,
    #     n_classes,
    #     mini_imagenet_database.input_shape
    # )
    feature_model = None

    sml = SML(
        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.01,
        num_steps_validation=5,
        save_after_epochs=500,
        meta_learning_rate=0.001,
        n_clusters=500,
        feature_model=feature_model,
        feature_size=288,
        input_shape=(84, 84, 3),
        log_train_images_after_iteration=50,
        least_number_of_tasks_val_test=50,
        report_validation_frequency=50,
        experiment_name='mini_imagenet_model_feature_10000_clusters_500'
    )

    sml.train(epochs=4001)
    # sml.evaluate(iterations=50)


def run_celeba():
    celeba_dataset = CelebADatabase(random_seed=-1)
    sml = SML(
        database=celeba_dataset,
        network_cls=MiniImagenetModel,
        n=2,
        meta_batch_size=8,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_epochs=5,
        meta_learning_rate=0.001,
        log_train_images_after_iteration=10,
        least_number_of_tasks_val_test=50,
        report_validation_frequency=100,
        experiment_name='euclidean'
    )

    sml.train(epochs=21)
    sml.evaluate(iterations=50)


if __name__ == '__main__':
    # run_omniglot()
    run_mini_imagenet()
    # run_celeba()

