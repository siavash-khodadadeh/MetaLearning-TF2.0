import os
import tensorflow as tf
import numpy as np

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import MiniImagenetModel
from databases import MiniImagenetDatabase, PlantDiseaseDatabase, ISICDatabase

from typing import List
from utils import keep_keys_with_greater_than_equal_k_items


class SeparateDomainMAML(ModelAgnosticMetaLearningModel):
    def get_train_dataset(self):
        trn_database = MiniImagenetDatabase()
        val_database = PlantDiseaseDatabase()
        dataset = self.get_separated_supervised_meta_learning_dataset(
            trn_database.train_folders,
            val_database.train_folders,
            n=self.n,
            k=self.k_ml,
            k_validation=self.k_val_ml,
            meta_batch_size=self.meta_batch_size
        )
        return dataset

    def get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, (84, 84))
            image = tf.cast(image, tf.float32)
            return image / 255.
        return parse_function

    def get_separated_supervised_meta_learning_dataset(
            self,
            trn_folders: List[str],
            val_folders: List[str],
            n: int,
            k: int,
            k_validation: int,
            meta_batch_size: int,
            one_hot_labels: bool = True,
            reshuffle_each_iteration: bool = True,
            seed: int = -1,
            dtype=tf.float32, #  The input dtype
    ) -> tf.data.Dataset:
        """Folders can be a dictionary and also real name of folders.
        If it is a dictionary then each item is the class name and the corresponding values are the file addressses
        of images of that class."""
        if seed != -1:
            np.random.seed(seed)

        def _get_trn_instances(class_dir_address):
            def get_instances(class_dir_address):
                class_dir_address = class_dir_address.numpy().decode('utf-8')
                instance_names = trn_folders[class_dir_address]
                return instance_names
            return tf.py_function(get_instances, inp=[class_dir_address], Tout=[tf.string])

        def _get_val_instances(class_dir_address):
            def get_instances(class_dir_address):
                class_dir_address = class_dir_address.numpy().decode('utf-8')
                instance_names = val_folders[class_dir_address]
                return instance_names
            return tf.py_function(get_instances, inp=[class_dir_address], Tout=[tf.string])

        if seed != -1:
            parallel_iterations = 1
        else:
            parallel_iterations = None

        def parse_function(imgs_addresses):
            imgs = tf.map_fn(
                self.get_parse_function(),
                imgs_addresses,
                dtype=dtype,
                parallel_iterations=parallel_iterations
            )
            return tf.stack(imgs)

        def convert_folders_to_list(folders):
            if type(folders) == list:
                classes = dict()
                for folder in folders:
                    instances = [os.path.join(folder, file_name) for file_name in os.listdir(folder)]
                    classes[folder] = instances
                folders = classes
            return folders

        trn_folders = convert_folders_to_list(trn_folders)
        val_folders = convert_folders_to_list(val_folders)

        keep_keys_with_greater_than_equal_k_items(trn_folders, k)
        keep_keys_with_greater_than_equal_k_items(val_folders, k_validation)

        trn_dataset = tf.data.Dataset.from_tensor_slices(sorted(list(trn_folders.keys())))
        val_dataset = tf.data.Dataset.from_tensor_slices(sorted(list(val_folders.keys())))

        if seed != -1:
            trn_dataset = trn_dataset.shuffle(
                buffer_size=len(trn_folders.keys()),
                reshuffle_each_iteration=reshuffle_each_iteration,
                seed=seed
            )
            val_dataset = val_dataset.shuffle(
                buffer_size=len(val_folders.keys()),
                reshuffle_each_iteration=reshuffle_each_iteration,
                seed=seed
            )
            # When using a seed the map should be done in the same order so no parallel execution
            trn_dataset = trn_dataset.map(_get_trn_instances, num_parallel_calls=1)
            val_dataset = val_dataset.map(_get_val_instances, num_parallel_calls=1)
        else:
            trn_dataset = trn_dataset.shuffle(
                buffer_size=len(trn_folders.keys()),
                reshuffle_each_iteration=reshuffle_each_iteration
            )
            val_dataset = val_dataset.shuffle(
                buffer_size=len(val_folders.keys()),
                reshuffle_each_iteration=reshuffle_each_iteration
            )
            trn_dataset = trn_dataset.map(_get_trn_instances, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            val_dataset = val_dataset.map(_get_val_instances, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        trn_dataset = trn_dataset.batch(k, drop_remainder=True)
        val_dataset = val_dataset.repeat().batch(k_validation)

        trn_dataset = trn_dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = tf.data.Dataset.zip((trn_dataset, val_dataset))
        dataset = dataset.batch(n, drop_remainder=True)

        labels_dataset = self.make_labels_dataset(n, k, k_validation, one_hot_labels=one_hot_labels)

        dataset = tf.data.Dataset.zip((dataset, labels_dataset))
        dataset = dataset.batch(meta_batch_size)

        steps_per_epoch = len(trn_folders.keys()) // (n * meta_batch_size)
        setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset


def run_separate_domain():
    test_database = ISICDatabase()
    maml = SeparateDomainMAML(
        database=test_database,
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
        report_validation_frequency=1000,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=True,
        experiment_name='separate_domain_isic',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
    )

    maml.train(iterations=60000)
    maml.evaluate(100, seed=14)


if __name__ == '__main__':
    run_separate_domain()
