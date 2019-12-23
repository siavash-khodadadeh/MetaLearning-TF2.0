import os
from shutil import copyfile
from datetime import datetime

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD, PCA
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks import SimpleModel, MiniImagenetModel
from tf_datasets import OmniglotDatabase, MiniImagenetDatabase, CelebADatabase
import settings


class UMTRAIterativeProjection(ModelAgnosticMetaLearningModel):
    def __init__(
            self,
            database,
            network_cls,
            n,
            meta_batch_size,
            num_steps_ml,
            lr_inner_ml,
            num_steps_validation,
            save_after_epochs,
            meta_learning_rate,
            report_validation_frequency,
            log_train_images_after_iteration,  # Set to -1 if you do not want to log train images.
            least_number_of_tasks_val_test=-1,  # Make sure the val and test dataset pick at least this many tasks.
            clip_gradients=False,
    ):
        super(UMTRAIterativeProjection, self).__init__(
            database=database,
            network_cls=network_cls,
            n=n,
            k=5,
            meta_batch_size=meta_batch_size,
            num_steps_ml=num_steps_ml,
            lr_inner_ml=lr_inner_ml,
            num_steps_validation=num_steps_validation,
            save_after_epochs=save_after_epochs,
            meta_learning_rate=meta_learning_rate,
            report_validation_frequency=report_validation_frequency,
            log_train_images_after_iteration=log_train_images_after_iteration,
            least_number_of_tasks_val_test=least_number_of_tasks_val_test,
            clip_gradients=clip_gradients
        )

    def get_root(self):
        return os.path.dirname(__file__)

    def get_features_from_files(self):
        # return self.get_pca_features_from_files(sampled_files)
        # return self.get_imagenet_vgg16_features_from_files()
        # return self.get_imagenet_inception_v3()
        return self.get_vgg19_last_hidden_layer()

    def get_vgg19_last_hidden_layer(self):
        base_model = tf.keras.applications.VGG19(weights='imagenet')
        model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[24].output)
        database_name = self.database.__class__.__name__
        return self.get_features(
            'vgg19_last_hidden_layer_{}'.format(database_name),
            model,
            input_shape=(224, 224),
            feature_size=4096,
            preprocess_fn=tf.keras.applications.inception_v3.preprocess_input
        )

    def get_inception_v3_features(self):
        model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)
        return self.get_features(
            'inception_v3',
            model,
            input_shape=(224, 224),
            feature_size=7*7*512,
            preprocess_fn=tf.keras.applications.inception_v3.preprocess_input
        )

    def get_features(self, name, model, input_shape, feature_size, preprocess_fn):
        dir_path = os.path.join(self.get_root(), name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        sampled_files_address = os.path.join(dir_path, 'sampled_files.npy')
        if os.path.exists(sampled_files_address):
            sampled_files = np.load(sampled_files_address)
        else:
            all_files = list()
            for class_name in self.database.train_folders:
                all_files.extend([os.path.join(class_name, file_name) for file_name in os.listdir(class_name)])

            sampled_files = np.random.choice(all_files, len(all_files), replace=False)
            np.save(sampled_files_address, sampled_files)

        features_address = os.path.join(dir_path, 'features.npy')
        if os.path.exists(features_address):
            features = np.load(features_address)
        else:
            n = len(sampled_files)
            m = feature_size
            features = np.zeros(shape=(n, m))

            for index, sampled_file in enumerate(sampled_files):
                if index % 1000 == 0:
                    print(f'{index}/{len(sampled_files)} images read')

                img = tf.keras.preprocessing.image.load_img(sampled_file, target_size=(input_shape[:2]))
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                if preprocess_fn is not None:
                    img = preprocess_fn(img)

                features[index, :] = model.predict(img).reshape(-1)

            np.save(features_address, features)

        print('features loaded')
        return np.transpose(features), sampled_files, dir_path

    def get_pca_features_from_files(self, sampled_files):
        all_files = list()
        root_address = os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/mini-imagenet/train')
        for dirpath, dirname, filenames in os.walk(root_address):
            if dirpath != root_address:
                all_files.extend([os.path.join(dirpath, file_name) for file_name in filenames])

        sampled_files = np.random.choice(all_files, len(all_files))

        np.save('sampled_files', sampled_files)
        # sampled_files = np.load('sampled_files.npy')

        n = len(sampled_files)
        m = 200  # Feature size

        all_imgs = np.zeros(shape=(n, 84 * 84 * 3))
        for index, sampled_file in enumerate(sampled_files):
            if index % 1000 == 0:
                print('{} images read'.format(index))

            img = Image.open(sampled_file)
            img = np.array(img.getdata()).reshape((img.size[0], img.size[1], 3))
            img = img.reshape(-1)
            all_imgs[index, :] = img

        np.save(os.path.join(self.get_root(), 'pca_imgs.npy'), all_imgs)
        # all_imgs = np.load(os.path.join(self.get_root(), 'ipca_mgs.npy'))

        print('data loaded')
        normalizer = Normalizer()
        all_imgs = normalizer.fit_transform(all_imgs)

        print('running PCA')
        pca = PCA(n_components=m)
        all_imgs = pca.fit_transform(all_imgs)
        print('PCA done')

        return np.transpose(all_imgs)

    def prepare_data(self):
        return self.get_features_from_files()

        # return np.random.random((200, 1000))

    def SP(self, data, K):
        A = data
        At = data
        inds = np.zeros(K, )
        inds = inds.astype(int)
        iter = 0
        for k in range(0, K):
            iter = iter + 1
            # Compute just the first column from U and V
            svd = TruncatedSVD(n_components=1)
            svd.fit(np.transpose(At))
            # [U, S, V] = np.linalg.svd(At, full_matrices=False)
            # u1 = U[:, 0]
            # v = V[:, 1]
            u = svd.components_.reshape(-1)
            N = np.linalg.norm(At, axis=0)
            B = At / N
            B = np.transpose(B)
            Cr = np.abs(np.matmul(B, u))
            ind = np.argsort(Cr)[::-1]
            p = ind[0]
            inds[k] = p
            A3 = A[:, inds[0:k + 1]]
            At = A - np.matmul(np.matmul(A3, np.linalg.pinv(np.matmul(np.transpose(A3), A3))),
                               np.matmul(np.transpose(A3), A))
        # ind2 = np.zeros(K - 1, );
        # for iter in range(1, 5):
        #     for k in range(0, K):
        #         ind2 = np.delete(inds, k)
        #         A3 = A[:, ind2]
        #         At = A - np.matmul(np.matmul(A3, np.linalg.pinv(np.matmul(np.transpose(A3), A3))),
        #                            np.matmul(np.transpose(A3), A))
        #         [U, S, V] = np.linalg.svd(At, full_matrices=False)
        #         u = U[:, 1]
        #         v = V[:, 1]
        #         N = np.linalg.norm(At, axis=0)
        #         B = At / N
        #         B = np.transpose(B)
        #         Cr = np.abs(np.matmul(B, u))
        #         ind = np.argsort(Cr)[::-1]
        #         p = ind[0]
        #         inds[k] = p

        return inds

    def get_norm_p(self, A):
        # fr_norm = np.linalg.norm(A)
        # return fr_norm

        vectors_norm = np.linalg.norm(A, 2, axis=0)
        norm_nim = np.linalg.norm(vectors_norm, 0.2)
        return norm_nim

    def idea_3(self, A, inds, k):
        n_classes = 5
        data_points = list()

        norm_values = np.linalg.norm(A, 2, axis=0)
        A = A / norm_values

        B = np.copy(A)
        B[:, inds] = np.zeros((A.shape[0], len(inds)))

        for i in range(n_classes):
            dot_product_values = np.matmul(np.transpose(A[:, inds[i]]), B)
            ind_j = np.argmax(np.abs(dot_product_values))
            data_points.append(inds[i])
            data_points.append(ind_j)
            # print(inds[i])
            # print(ind_j)

        # the list of all elements: 1, 1, 2, 2, 3, 3, 4, 4, 5, 5
        return data_points


    def idea_2(self, A, inds, k):
        n_classes = 5
        data_points = list()

        for i in range(n_classes):
            fr_norm_min = 10e35
            ind_j = -1
            for j in range(n_classes, k):
                print('running i: {}, j: {}'.format(i, j))
                A3 = A[:, [inds[i], inds[j]]]

                begin_time = datetime.now()
                At = A - np.matmul(np.matmul(A3, np.linalg.pinv(np.matmul(np.transpose(A3), A3) + 0* np.eye(2))),
                                   np.matmul(np.transpose(A3), A))

                end_time = datetime.now()
                print(end_time - begin_time)

                begin_time = datetime.now()
                fr_norm = self.get_norm_p(At)
                end_time = datetime.now()
                print(end_time - begin_time)

                if fr_norm < fr_norm_min:
                    ind_j = inds[j]
                    fr_norm_min = fr_norm

            print('fr_norm: {}'.format(fr_norm_min))
            data_points.append(inds[i])
            data_points.append(ind_j)

            A3 = A[:, [inds[i], ind_j]]
            # A = A - np.matmul(np.matmul(A3, np.linalg.pinv(np.matmul(np.transpose(A3), A3) + 0.5 * np.eye(2))), np.matmul(np.transpose(A3), A))

        # the list of all elements: 1, 1, 2, 2, 3, 3, 4, 4, 5, 5
        return data_points


    def idea_1(self, A, inds, k):
        data_points = list()

        for data_point in range(5):
            fr_norm_min = 10e6
            ind_i = -1
            ind_j = -1

            for i in range(k):
                for j in range(i + 1, k):
                    print('running i: {}, j: {}'.format(i, j))
                    A3 = A[:, [inds[i], inds[j]]]

                    begin_time = datetime.now()
                    At = A - np.matmul(np.matmul(A3, np.linalg.pinv(np.matmul(np.transpose(A3), A3) + 0.01 * np.eye(2))),
                                       np.matmul(np.transpose(A3), A))

                    end_time = datetime.now()
                    print(end_time - begin_time)

                    begin_time = datetime.now()
                    fr_norm = self.get_norm_p(At)
                    end_time = datetime.now()
                    print(end_time - begin_time)

                    if fr_norm < fr_norm_min:
                        ind_i = inds[i]
                        ind_j = inds[j]
                        fr_norm_min = fr_norm

            data_points.append(ind_i)
            data_points.append(ind_j)
            A3 = A[:, [ind_i, ind_j]]
            A = A - np.matmul(np.matmul(A3, np.linalg.pinv(np.matmul(np.transpose(A3), A3) + 0.5 * np.eye(2))),
                               np.matmul(np.transpose(A3), A))

        # the list of all elements: 1, 1, 2, 2, 3, 3, 4, 4, 5, 5
        return data_points

    def get_test_dataset(self):
        self.database.get_attributes_task_dataset(partition='test', k=self.k, meta_batch_size=1)

    def get_val_dataset(self):
        val_dataset = self.database.get_supervised_meta_learning_dataset(
            self.database.val_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=1,
            reshuffle_each_iteration=True
        )
        steps_per_epoch = max(val_dataset.steps_per_epoch, self.least_number_of_tasks_val_test)
        val_dataset = val_dataset.repeat(-1)
        val_dataset = val_dataset.take(steps_per_epoch)
        setattr(val_dataset, 'steps_per_epoch', steps_per_epoch)
        return val_dataset


    def get_train_dataset(self):
        k = 5
        num_tasks = 20000

        data, sampled_files, dir_path = self.prepare_data()
        tasks_address = os.path.join(dir_path, f'tasks_{num_tasks}.npy')

        if os.path.exists(tasks_address):
            tasks = np.load(tasks_address)
        else:
            tasks = list()
            begin_time = datetime.now()
            print(begin_time)
            for i in range(num_tasks):
                if i % 100 == 0:
                    print(f'{i} tasks are generated out of {num_tasks}')

                cols = np.random.choice(data.shape[1], 2000, replace=False)
                task_data_points = data[:, cols]
                task_sampled_files = sampled_files[cols]

                inds = self.SP(task_data_points, k)
                # inds = np.random.randint(0, 2000, size=k)

                selected_data_points = self.idea_3(task_data_points, inds, k)

                tasks.append(task_sampled_files[selected_data_points])

            np.save(tasks_address, tasks)

            end_time = datetime.now()
            print(end_time)
            print(f'time to generate tasks: {end_time - begin_time}')

        dataset = self.database.get_dataset_from_tasks_directly(
            tasks,
            self.n,
            self.k,
            self.meta_batch_size
        )

        # dataset = self.database.get_umtra_dataset(
        #     self.database.train_folders,
        #     n=self.n,
        #     meta_batch_size=self.meta_batch_size,
        #     augmentation_function=self.augmentation_function
        # )

        return dataset

    def get_config_info(self):
        return f'umtra_' \
               f'model-{self.network_cls.name}_' \
               f'mbs-{self.meta_batch_size}_' \
               f'n-{self.n}_' \
               f'k-{self.k}_' \
               f'stp-{self.num_steps_ml}'


def run_omniglot():
    omniglot_database = OmniglotDatabase(
        random_seed=-1,
        num_train_classes=1200,
        num_val_classes=100,
    )

    umtra = UMTRAIterativeProjection(
        database=omniglot_database,
        network_cls=SimpleModel,
        n=5,
        meta_batch_size=32,
        num_steps_ml=5,
        lr_inner_ml=0.01,
        num_steps_validation=5,
        save_after_epochs=5,
        meta_learning_rate=0.001,
        log_train_images_after_iteration=10,
        report_validation_frequency=1,
    )

    umtra.train(epochs=10)
    umtra.evaluate(iterations=50)


def run_mini_imagenet():
    mini_imagenet_database = MiniImagenetDatabase(random_seed=-1)

    umtra = UMTRAIterativeProjection(
        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.01,
        num_steps_validation=5,
        save_after_epochs=5,
        meta_learning_rate=0.001,
        log_train_images_after_iteration=10,
        report_validation_frequency=1,
    )

    umtra.train(epochs=10)
    umtra.evaluate(iterations=50)


def run_celeba():
    celeba_dataset = CelebADatabase(random_seed=-1)
    umtra = UMTRAIterativeProjection(
        database=celeba_dataset,
        network_cls=MiniImagenetModel,
        n=5,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.01,
        num_steps_validation=5,
        save_after_epochs=5,
        meta_learning_rate=0.001,
        log_train_images_after_iteration=10,
        report_validation_frequency=1,
    )

    # umtra.train(epochs=100)
    umtra.evaluate(iterations=50)


if __name__ == '__main__':
    # run_omniglot()
    # run_mini_imagenet()
    run_celeba()

