import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from databases import MiniImagenetDatabase, OmniglotDatabase, AirplaneDatabase, CUBDatabase, DTDDatabase, \
    FungiDatabase, VGGFlowerDatabase, TrafficSignDatabase, MSCOCODatabase, PlantDiseaseDatabase, ISICDatabase, \
    EuroSatDatabase, ChestXRay8Database


def get_dataset_from_database(database, shape=(84, 84, 3)):
    def instance_parse_function(example):
        if isinstance(database, OmniglotDatabase):
            example = tf.squeeze(example, axis=2)
            example = tf.stack((example, example, example), axis=2)

        example = tf.image.resize(example, shape[:2])
        return example

    instances = database.get_all_instances()
    instances = np.random.choice(instances, 20, replace=False)

    dataset = tf.data.Dataset.from_tensor_slices(instances)
    dataset = dataset.map(database._get_parse_function())
    dataset = dataset.map(instance_parse_function)

    return dataset


def visualize_database(database, save_folder):
    dataset = get_dataset_from_database(database)

    fig, axes = plt.subplots(4, 5)
    row_counter = 0
    col_counter = 0
    for item in dataset:
        axes[row_counter, col_counter].imshow(item)
        col_counter += 1
        if col_counter == 5:
            col_counter = 0
            row_counter += 1

    title_name = str(database.__class__)[str(database.__class__).rfind('.') + 1:-2]
    fig.suptitle(title_name, fontsize=12, y=1)
    plt.savefig(fname=os.path.join(save_folder, title_name))
    plt.show()


def visualize_all_domains_together(databases, root_folder_to_save):
    fig, axes = plt.subplots(4, 12)
    fig.set_figwidth(12)
    fig.set_figheight(4)

    for i, database in enumerate(databases):
        dataset = get_dataset_from_database(database, (84, 84, 3)).take(4)
        for j, item in enumerate(dataset):
            axes[j, i].set_xticklabels([])
            axes[j, i].set_xticks([])
            axes[j, i].set_yticklabels([])
            axes[j, i].set_yticks([])
            axes[j, i].imshow(item, aspect='equal')

        title_name = str(database.__class__)[str(database.__class__).rfind('.') + 1:-2][:-8]
        axes[0, i].xaxis.set_label_position('top')
        axes[0, i].set_xlabel(title_name)

    # fig.suptitle('', fontsize=12, y=1)
    plt.savefig(fname=os.path.join(root_folder_to_save, 'all_domains.pdf'))
    plt.show()


if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')
    root_folder_to_save = os.path.expanduser('~/datasets_visualization/')
    if not os.path.exists(root_folder_to_save):
        os.mkdir(root_folder_to_save)

    databases = (
        MiniImagenetDatabase(),
        OmniglotDatabase(random_seed=42, num_train_classes=1200, num_val_classes=100),
        AirplaneDatabase(),
        CUBDatabase(),
        DTDDatabase(),
        FungiDatabase(),
        VGGFlowerDatabase(),
        # TrafficSignDatabase(),
        MSCOCODatabase(),
        PlantDiseaseDatabase(),
        EuroSatDatabase(),
        ISICDatabase(),
        ChestXRay8Database(),
    )

    # for database in databases:
    #     visualize_database(database, root_folder_to_save)

    visualize_all_domains_together(databases, root_folder_to_save)
