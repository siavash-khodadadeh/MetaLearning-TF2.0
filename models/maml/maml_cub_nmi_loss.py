import tensorflow as tf

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import MiniImagenetModel
from databases import CUBDatabase, Omniglot84x84Database, AirplaneDatabase, FungiDatabase, DTDDatabase, \
    VGGFlowerDatabase, MiniImagenetDatabase


class ModelAgnosticMetaLearningNMI(ModelAgnosticMetaLearningModel):
    pass
    def outer_loss(self, labels, logits, inner_losses=None):
        # tf.print('logits: ', end='')
        # tf.print(tf.argmax(logits, axis=1), summarize=-1)
        # tf.print('labels:', end='')
        # tf.print(tf.argmax(labels, axis=1), summarize=-1)
        #
        # return super(ModelAgnosticMetaLearningNMI, self).outer_loss(labels, logits, inner_losses)
        losses = list()
        import itertools

        for permutation in itertools.permutations([0, 1, 2, 3, 4]):
            new_labels = tf.gather(labels, permutation, axis=1)
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(new_labels, logits, from_logits=True))
            losses.append(loss)

        return tf.reduce_min(losses)
        # if inner_losses is not None:
        # loss = inner_losses[0]
        # loss = tf.reduce_mean(
        #     tf.losses.categorical_crossentropy(labels, logits, from_logits=True)
        # )

        num_x = labels.shape[0]
        p_y_on_x = labels
        p_c_on_x = tf.nn.softmax(logits, axis=1)

        p_y = tf.reduce_sum(p_y_on_x, axis=0, keepdims=True) / num_x  # 1-by-num_y
        h_y = -tf.reduce_sum(p_y * tf.math.log(p_y))
        p_c = tf.reduce_sum(p_c_on_x, axis=0) / num_x  # 1-by-num_c
        h_c = -tf.reduce_sum(p_c * tf.math.log(p_c))
        p_x_on_y = p_y_on_x / num_x / p_y  # num_x-by-num_y
        p_c_on_y = tf.matmul(p_c_on_x, p_x_on_y, transpose_a=True)  # num_c-by-num_y
        h_c_on_y = -tf.reduce_sum(tf.reduce_sum(p_c_on_y * tf.math.log(p_c_on_y), axis=0) * p_y)
        i_y_c = h_c - h_c_on_y
        nmi = 2 * i_y_c / (h_y + h_c)
        loss = -nmi * 10
        loss = -i_y_c * 10

        return loss


def run_cub():
    cub_database = CUBDatabase()

    maml = ModelAgnosticMetaLearningNMI(
        database=cub_database,
        # test_database=MiniImagenetDatabase(),
        network_cls=MiniImagenetModel,
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val=1,
        k_val_val=15,
        k_test=1,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=15000,
        meta_learning_rate=0.001,
        report_validation_frequency=1000,
        log_train_images_after_iteration=1000,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='cub_nmi_permutation_loss',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
    )

    maml.train(iterations=15000)
    maml.evaluate(50, num_tasks=1000, seed=42, use_val_batch_statistics=True)
    maml.evaluate(50, num_tasks=1000, seed=42, use_val_batch_statistics=False)


if __name__ == '__main__':
    # tf.config.experimental_run_functions_eagerly(True)

    run_cub()
