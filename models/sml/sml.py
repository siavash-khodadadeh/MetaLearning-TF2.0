import os
import pickle

import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import SimpleModel, MiniImagenetModel
from networks.sml_feature_networks import SimpleModelFeature
from databases import OmniglotDatabase, MiniImagenetDatabase


class SML(ModelAgnosticMetaLearningModel):
    def __init__(
            self,
            database,
            network_cls,
            n,
            k,
            k_val_ml,
            k_val_val,
            k_val_test,
            k_test,
            meta_batch_size,
            num_steps_ml,
            lr_inner_ml,
            num_steps_validation,
            save_after_iterations,
            meta_learning_rate,
            report_validation_frequency,
            log_train_images_after_iteration,  # Set to -1 if you do not want to log train images.
            n_clusters,
            feature_model,
            feature_size,
            input_shape,
            preprocess_function=None,
            number_of_tasks_val=-1,  # Make sure the val and test dataset pick at least this many tasks.
            number_of_tasks_test=-1,
            clip_gradients=False,
            experiment_name=None,
            val_seed=-1,
            val_test_batch_norm_momentum=0.0,
            target_database=None
    ):
        super(SML, self).__init__(
            database=database,
            network_cls=network_cls,
            n=n,
            k=k,
            k_val_ml=k_val_ml,
            k_val_val=k_val_val,
            k_val_test=k_val_test,
            k_test=k_test,
            meta_batch_size=meta_batch_size,
            num_steps_ml=num_steps_ml,
            lr_inner_ml=lr_inner_ml,
            num_steps_validation=num_steps_validation,
            save_after_iterations=save_after_iterations,
            meta_learning_rate=meta_learning_rate,
            report_validation_frequency=report_validation_frequency,
            log_train_images_after_iteration=log_train_images_after_iteration,
            number_of_tasks_val=number_of_tasks_val,
            number_of_tasks_test=number_of_tasks_test,
            clip_gradients=clip_gradients,
            experiment_name=experiment_name,
            val_seed=val_seed,
            val_test_batch_norm_momentum=val_test_batch_norm_momentum,
            target_database=target_database
        )
        self.features_model = feature_model
        self.n_clusters = n_clusters
        self.feature_size = feature_size
        self.input_shape = input_shape
        self.preprocess_fn = preprocess_function

    def get_network_name(self):
        return self.model.name

    def get_config_str(self):
        config_str = super(SML, self).get_config_str()
        config_str += f'_clusters_{self.n_clusters}'
        return config_str

    def get_features(self, dir_name=None):
        files_names_address = os.path.join(dir_name, 'file_names.npy')
        features_address = os.path.join(dir_name, 'features.npy')

        if dir_name is not None and os.path.exists(dir_name):
            return np.load(features_address), np.load(files_names_address)

        all_files = list()

        for class_name in self.database.train_folders:
            # all_files.extend([os.path.join(class_name, file_name) for file_name in os.listdir(class_name)])
            all_files.extend(self.database.train_folders[class_name])

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
            # features[index, :] = self.features_model.encode(img).reshape(-1)

        if dir_name is not None:
            os.makedirs(dir_name)
            np.save(files_names_address, all_files)
            np.save(features_address, features)
        return features, all_files

    def get_meta_learning_dataset_from_clusters(
            self,
            clusters_dir,
            n,
            k,
            k_val,
            meta_batch_size,
            one_hot_labels=True,
            reshuffle_each_iteration=True,
    ):
        def _get_instances(cluster_id_file_address):
            cluster_instances = tf.strings.split(tf.io.read_file(cluster_id_file_address), '\n')[:-1]

            def get_instances(ci):
                instances = np.random.choice(ci, size=k + k_val, replace=False)
                return instances[:k], instances[k:k + k_val]

            return tf.py_function(get_instances, inp=[cluster_instances], Tout=[tf.string, tf.string])

        def parse_function(tr_imgs_addresses, val_imgs_addresses):
            tr_imgs = tf.map_fn(self.get_parse_function(), tr_imgs_addresses, dtype=tf.float32)
            val_imgs = tf.map_fn(self.get_parse_function(), val_imgs_addresses, dtype=tf.float32)

            return tf.stack(tr_imgs), tf.stack(val_imgs)

        classes = sorted([os.path.join(clusters_dir, file_address) for file_address in os.listdir(clusters_dir)])
        final_classes = list()

        for class_name in classes:
            if len(open(class_name).readlines()) >= k + k_val:
                final_classes.append(class_name)

        classes = final_classes
        steps_per_epoch = len(classes) // n // meta_batch_size

        dataset = tf.data.Dataset.from_tensor_slices(classes)
        dataset = dataset.shuffle(buffer_size=len(classes), reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = dataset.map(_get_instances,  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(n, drop_remainder=True)

        labels_dataset = self.make_labels_dataset(n, k, k_val, one_hot_labels)
        dataset = tf.data.Dataset.zip((dataset, labels_dataset))
        dataset = dataset.batch(meta_batch_size, drop_remainder=True)

        setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset

    def get_train_dataset(self):
        clusters_files_dir = os.path.join(
            self.get_root(), f'cache/{self.experiment_name}', f'clusters_{self.n_clusters}'
        )
        if not os.path.exists(clusters_files_dir):
            features, all_files = self.get_features(
                dir_name=os.path.join(self.get_root(), f'cache/{self.experiment_name}')
            )
            print('Feature vectors are loaded.')

            k_means_path = os.path.join(
                self.get_root(),
                f'cache/{self.experiment_name}',
                f'k_means_{self.n_clusters}.pkl'
            )
            if os.path.exists(k_means_path):
                k_means = pickle.load(open(k_means_path, 'rb'))
                cluster_ids = k_means.predict(features)
            else:
                pca = PCA(n_components=256, whiten=True)
                features = pca.fit_transform(features)
                row_sums = np.linalg.norm(features, axis=1)
                features = features / row_sums[:, np.newaxis]
                print('PCA Finished.')
                # k_means = KMeans(n_clusters=self.n_clusters)
                k_means = KMeans(
                    n_clusters=self.n_clusters,
                    n_init=1,  # It does not take forever in case of celeba
                    max_iter=3000,
                    precompute_distances=True,
                    n_jobs=32,
                    verbose=2
                )
                cluster_ids = k_means.fit_predict(features)
                with open(k_means_path, 'wb') as f:
                    pickle.dump(k_means, f)
                print('K-means finished.')

            clusters = dict()
            for i, file_address in enumerate(all_files):
                cluster_index = cluster_ids[i]
                if cluster_index not in clusters.keys():
                    clusters[cluster_index] = list()

                clusters[cluster_index].append(file_address)

            os.makedirs(clusters_files_dir)
            for cluster_id, cluster_instances in clusters.items():
                with open(os.path.join(clusters_files_dir, str(cluster_id) + '.txt'), 'w') as cluster_file:
                    for cluster_instance in cluster_instances:
                        cluster_file.write(cluster_instance)
                        cluster_file.write('\n')

        tr_dataset = self.get_meta_learning_dataset_from_clusters(
            clusters_dir=clusters_files_dir,
            n=self.n,
            k=self.k,
            k_val=self.k_val_ml,
            meta_batch_size=self.meta_batch_size
        )
        print(tr_dataset.steps_per_epoch)
        return tr_dataset


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

    np.random.seed(27)
    selected_indices = np.random.choice(indices, n_samples, replace=False)
    np.random.seed(None)

    not_selected_indices = np.array([index for index in indices if index not in selected_indices])

    # return instances[selected_indices], labels[selected_indices], instances[not_selected_indices]
    return instances[selected_indices], labels[selected_indices], None


def make_features_dataset_mini_imagenet(data, labels, non_labeled_data, shuffle_buffer_size=None, batch_size=32):
    # non_labeled_data = np.concatenate((data, non_labeled_data))
    label_values = np.unique(labels)
    depth = len(label_values)
    label_to_index = {label_value: i for i, label_value in enumerate(label_values)}
    labels = [label_to_index[label] for label in labels]

    x_dataset = tf.data.Dataset.from_tensor_slices(data)
    y_dataset = tf.data.Dataset.from_tensor_slices(labels)

    def _parse_image(image_file):
        image = tf.image.decode_jpeg(tf.io.read_file(image_file))
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32)
        return image / 255.

    def _parse_image_label_pair(image_file, label):
        image = _parse_image(image_file)
        label = tf.one_hot(label, depth=depth)
        return image, label

    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
    dataset = dataset.map(_parse_image_label_pair)

    # non_labeled_dataset = tf.data.Dataset.from_tensor_slices(non_labeled_data)
    # non_labeled_dataset = non_labeled_dataset.map(_parse_image)

    # dataset = tf.data.Dataset.zip((dataset, non_labeled_dataset))
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=False)

    # dataset = dataset.cache()

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


def train_the_feature_model2(sequential_model, dataset, n_classes, input_shape):
    features = sequential_model.layers[-2].output
    y_hat = tf.keras.layers.Dense(n_classes, activation='softmax')(features)

    model = tf.keras.models.Model(inputs=sequential_model.input, outputs=y_hat)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # TODO does this reset dataset at each iteration and shuffle it randomly
    model.fit(dataset, epochs=90)
    model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    return model


def train_the_feature_model_with_classification(fm, dataset, n_classes, input_shape):
    file_path = f'./feature_models/feature_model_100'
    inputs = tf.keras.layers.Input(shape=fm.encoder.input.shape[1:])
    outputs = fm.classification_dense(fm.encoder(inputs))
    network = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    network.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    if not os.path.exists(file_path + '.index'):
        network.fit(dataset, epochs=100)
        network.save_weights(filepath=file_path)
    else:
        # Do this to instantiate all the parameters of the network before loading.
        network.predict(tf.random.uniform(shape=[2, *inputs.shape[1:]]))
        network.load_weights(filepath=file_path)
        # network.evaluate(dataset)

    inputs = tf.keras.layers.Input(shape=fm.encoder.input.shape[1:])
    outputs = network.layers[-2](inputs)
    network = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    setattr(network, 'latent_dim', fm.latent_dim)
    return network


def train_the_feature_model(fm, dataset, n_classes, input_shape):
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_on_batch_classification(feature_model, x, y, x_unsupervised):
        with tf.GradientTape() as tape:
            classification_loss, classification_acc = feature_model.compute_classification_loss(x, y)
            grads = tape.gradient(classification_loss, [
                *feature_model.encoder.trainable_variables,
                *feature_model.classificatino_dense.trainable_variables
            ])

        return grads, classification_loss, classification_acc

    @tf.function
    def train_on_batch(feature_model, x, y, x_unsupervised):
        classification_coefficient = 3200
        with tf.GradientTape() as tape:
            # tf,concatenate((x, x_unsupervised), axis=0)
            vae_loss = feature_model.compute_vae_loss(x_unsupervised)
            classification_loss, classification_acc = feature_model.compute_classification_loss(x, y)
            classification_loss *= classification_coefficient
            grads = tape.gradient(vae_loss + classification_loss, feature_model.trainable_variables)
        return grads, vae_loss, classification_loss, classification_acc

    def train_network(feature_model, dataset):
        iteration_counter = 0
        for epoch in range(0, 501):
            print(f'================\nEpoch {epoch}:')
            # vae_losses = list()
            classification_losses = list()
            classification_accs = list()

            for item in tqdm(dataset):
                (x, y), x_unsupervised = item
                # grads, vae_loss, classification_loss, classification_acc = train_on_batch_classification(
                grads, classification_loss, classification_acc = train_on_batch_classification(
                    feature_model, x, y, x_unsupervised
                )
                classification_losses.append(classification_loss)
                classification_accs.append(classification_acc)
                # vae_losses.append(vae_loss)

                # optimizer.apply_gradients(zip(grads, feature_model.trainable_variables))
                optimizer.apply_gradients(zip(
                    grads,
                    [
                        *feature_model.encoder.trainable_variables,
                        *feature_model.classificatino_dense.trainable_variables
                    ]
                ))
                iteration_counter += 1

            # print(f'Reconstruction loss: {np.mean(reconstruction_losses)}')
            tf.print(f'Classification loss: {np.mean(classification_losses)}')
            tf.print(f'Classification acc: {np.mean(classification_accs)}')
            # print(f'KLD loss: {np.mean(kld_losses)}')
            # print(f'VAE loss: {np.mean(vae_losses)}')

            if epoch != 0 and epoch % 500 == 0:
                feature_model.save_weights(filepath=f'./feature_models/feature_model_{epoch}')

    epoch = 20
    for item in dataset:
        (x, y), x_unsupervised = item
        print(y)
        # grads, vae_loss, classification_loss, classification_acc = train_on_batch(fm, x, y, x_unsupervised)
        grads, classification_loss, classification_acc = train_on_batch_classification(fm, x, y, x_unsupervised)
        break

    # fm.load_weights(filepath=f'./feature_models/feature_model_{epoch}')
    train_network(fm, dataset)

    # for item in dataset.take(1):
    #     (x, y), x_unsupervised = item
    #     mean, var = fm.encode(x)
    #     z = fm.reparameterize(mean, var)
    #     x_logit = fm.decode(z, apply_sigmoid=True)
    #
    #     for image in (x[0, ...], x_logit[0, ...]):
    #         import matplotlib.pyplot as plt
    #         print(image.shape)
    #         plt.imshow(image)
    #         plt.show()

    return fm


def run_omniglot():
    omniglot_database = OmniglotDatabase(
        random_seed=30,
        num_train_classes=1200,
        num_val_classes=100,
    )
    n_data_points = 10000

    data_points, classes, non_labeled_data_points = sample_data_points(omniglot_database.train_folders, n_data_points)
    features_dataset, n_classes = make_features_dataset_omniglot(data_points, classes, non_labeled_data_points)
    print(n_classes)
    feature_model = SimpleModelFeature(num_classes=5).get_sequential_model()
    feature_model = train_the_feature_model(feature_model, features_dataset, n_classes, omniglot_database.input_shape)

    sml = SML(
        database=omniglot_database,
        network_cls=SimpleModel,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
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
        experiment_name='omniglot_vae_model_feature_10000'
    )

    sml.train(epochs=101)
    # sml.evaluate(iterations=50)


def run_mini_imagenet():
    mini_imagenet_database = MiniImagenetDatabase()
    n_data_points = 38400
    data_points, classes, non_labeled_data_points = sample_data_points(
        mini_imagenet_database.train_folders,
        n_data_points
    )
    features_dataset, n_classes = make_features_dataset_mini_imagenet(
        data_points,
        classes,
        non_labeled_data_points,
        # shuffle_buffer_size=n_data_points,
        # batch_size=32,
        batch_size=16,
        shuffle_buffer_size=1000,
    )
    feature_model = tf.keras.applications.VGG19(weights=None, classes=n_classes)
    feature_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
    )

    def save_call_back(epoch, logs):
        if epoch % 100 == 0:
            feature_model.save_weights(filepath=f'./feature_models/feature_model_{epoch}')

    # saver_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=save_call_back)
    # feature_model.fit(features_dataset, epochs=101, callbacks=[saver_callback])
    feature_model.load_weights(filepath=f'./feature_models/feature_model_100')
    # feature_model.evaluate(features_dataset)
    # exit()
    feature_model = tf.keras.models.Model(inputs=feature_model.input, outputs=feature_model.layers[24].output)

    # print(n_classes)
    # feature_model = VariationalAutoEncoderFeature(input_shape=(84, 84, 3), latent_dim=32, n_classes=n_classes)
    # feature_model = train_the_feature_model_with_classification(
    #     feature_model,
    #     features_dataset,
    #     n_classes,
    #     mini_imagenet_database.input_shape
    # )

    # feature_model = None

    # use imagenet
    # base_model = tf.keras.applications.VGG19(weights='imagenet')
    # feature_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[24].output)

    sml = SML(
        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=1,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=15000,
        meta_learning_rate=0.001,
        n_clusters=500,
        feature_model=feature_model,
        # feature_size=288,
        feature_size=4096,
        input_shape=(224, 224, 3),
        preprocess_function=tf.keras.applications.vgg19.preprocess_input,
        log_train_images_after_iteration=1000,
        least_number_of_tasks_val_test=100,
        report_validation_frequency=250,
        experiment_name='mini_imagenet_learn_miniimagent_features'
    )
    sml.train(iterations=60000)
    # sml.evaluate(iterations=50, seed=14)


