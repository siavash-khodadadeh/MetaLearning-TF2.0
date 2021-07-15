import tensorflow as tf

from databases import CelebADatabase, CUBDatabase
from models.maml_umtra.maml_umtra import MAMLUMTRA
from networks.maml_umtra_networks import MiniImagenetModel


# class MAMLUMTRANMI(MAMLUMTRA):
#     def outer_loss(self, labels, logits, inner_losses=None):
#         losses = list()
#         import itertools
#
#         for permutation in itertools.permutations([0, 1, 2, 3, 4]):
#             new_labels = tf.gather(labels, permutation, axis=1)
#             loss = tf.reduce_mean(tf.losses.categorical_crossentropy(new_labels, logits, from_logits=True))
#             losses.append(loss)
#
#         return tf.reduce_min(losses)


if __name__ == '__main__':
    # import tensorflow as tf
    # tf.config.experimental_run_functions_eagerly(True)

    cub_database = CUBDatabase()

    maml_umtra = MAMLUMTRA(
        database=cub_database,
        network_cls=MiniImagenetModel,
        n=5,
        k_ml=1,
        k_val_ml=1,
        k_val=1,
        k_val_val=15,
        k_test=5,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=15000,
        meta_learning_rate=0.001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='cub',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    shape = (84, 84, 3)
    # maml_umtra.visualize_umtra_task(shape, num_tasks_to_visualize=2)

    maml_umtra.train(iterations=60000)
    maml_umtra.evaluate(iterations=50, num_tasks=1000, seed=42, use_val_batch_statistics=False)
