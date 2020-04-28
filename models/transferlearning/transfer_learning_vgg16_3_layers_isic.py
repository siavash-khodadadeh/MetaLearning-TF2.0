from models.transferlearning.transfer_learning_vgg16 import TransferLearningVGG16
from databases import ISICDatabase


def run_transfer_learning():
    isic_database = ISICDatabase()
    transfer_learning = TransferLearningVGG16(
        database=isic_database,
        n=5,
        k_val_test=15,
        k_test=5,
        lr_inner_ml=0.001,
        number_of_tasks_test=100,
        val_test_batch_norm_momentum=0.0,
        random_layer_initialization_seed=42,
        num_trainable_layers=3,
    )
    transfer_learning.evaluate(10, seed=42, use_val_batch_statistics=True)


if __name__ == '__main__':
    run_transfer_learning()
