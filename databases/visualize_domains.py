import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from databases import MiniImagenetDatabase, OmniglotDatabase, AirplaneDatabase, CUBDatabase, DTDDatabase, \
    FungiDatabase, VGGFlowerDatabase, TrafficSignDatabase, MSCOCODatabase, PlantDiseaseDatabase, ISICDatabase, \
    EuroSatDatabase, ChestXRay8Database


def visualize_database(database, save_folder):
    def instance_parse_function(example):
        if isinstance(database, OmniglotDatabase):
            example = tf.squeeze(example, axis=2)
            example = tf.stack((example, example, example), axis=2)
        return example

    instances = database.get_all_instances()
    instances = np.random.choice(instances, 20, replace=False)

    dataset = tf.data.Dataset.from_tensor_slices(instances)
    dataset = dataset.map(database._get_parse_function())
    dataset = dataset.map(instance_parse_function)

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


if __name__ == '__main__':
    root_folder_to_save = os.path.expanduser('~/datasets_visualization/')
    if not os.path.exists(root_folder_to_save):
        os.mkdir(root_folder_to_save)

    for database in (
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
    ):
        visualize_database(database, root_folder_to_save)
