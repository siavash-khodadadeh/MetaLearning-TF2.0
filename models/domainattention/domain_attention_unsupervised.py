import tensorflow as tf

from databases import MiniImagenetDatabase, AirplaneDatabase, CUBDatabase, OmniglotDatabase, DTDDatabase, \
    FungiDatabase, VGGFlowerDatabase, EuroSatDatabase
from models.crossdomain.cdml import CombinedCrossDomainMetaLearning
from models.domainattention.domain_attention_models import DomainAttentionModel


class DomainAttentionUnsupervised(CombinedCrossDomainMetaLearning):
    def __init__(self, train_databases, image_shape=(84, 84, 3), *args, **kwargs):
        self.train_databases = train_databases
        self.epochs_each_domain = [450, 50, 50, 50]
        self.image_shape = image_shape
        super(DomainAttentionUnsupervised, self).__init__(*args, **kwargs)

    def outer_loss(self, labels, logits, inner_losses=None):
        # if inner_losses is not None:
        # loss = inner_losses[0]
        # loss = tf.reduce_mean(
        #     tf.losses.categorical_crossentropy(labels, logits, from_logits=True)
        # )

        losses = list()
        import itertools

        for permutation in itertools.permutations([0, 1, 2, 3, 4]):
            new_labels = tf.gather(labels, permutation, axis=1)
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(new_labels, logits, from_logits=True))
            losses.append(loss)

        return tf.reduce_min(losses)

        # num_x = labels.shape[0]
        # p_y_on_x = labels
        # p_c_on_x = tf.nn.softmax(logits, axis=1)
        #
        # p_y = tf.reduce_sum(p_y_on_x, axis=0, keepdims=True) / num_x  # 1-by-num_y
        # h_y = -tf.reduce_sum(p_y * tf.math.log(p_y + 1e-9))
        # p_c = tf.reduce_sum(p_c_on_x, axis=0) / num_x  # 1-by-num_c
        # h_c = -tf.reduce_sum(p_c * tf.math.log(p_c + 1e-9))
        # p_x_on_y = p_y_on_x / num_x / p_y  # num_x-by-num_y
        # p_c_on_y = tf.matmul(p_c_on_x, p_x_on_y, transpose_a=True)  # num_c-by-num_y
        # h_c_on_y = -tf.reduce_sum(tf.reduce_sum(p_c_on_y * tf.math.log(p_c_on_y + 1e-9), axis=0) * p_y)
        # i_y_c = h_c - h_c_on_y
        # nmi = 2 * i_y_c / (h_y + h_c + 1e-9)
        # loss = -nmi * 10
        #
        # return loss

    def get_network_name(self):
        return 'DomainAttentionModel'

    def initialize_network(self):
        da = DomainAttentionModel(
            train_dbs=self.train_databases,
            num_classes=self.n,
            root=self._root,
            db_encoder_epochs=self.epochs_each_domain,
            db_encoder_lr=0.001,
            image_shape=self.image_shape
        )
        da(tf.zeros(shape=(1, *self.image_shape)))

        return da

    # def get_cross_domain_meta_learning_dataset(
    #         self,
    #         databases,
    #         n: int,
    #         k_ml: int,
    #         k_validation: int,
    #         meta_batch_size: int,
    #         one_hot_labels: bool = True,
    #         reshuffle_each_iteration: bool = True,
    #         seed: int = -1,
    #         dtype=tf.float32,  # The input dtype
    # ) -> tf.data.Dataset:
    #     database = databases[0]
    #     dataset = self.data_loader.get_unsupervised_dataset(
    #         database.train_folders,
    #         n,
    #         # k_ml,
    #         # k_validation,
    #         meta_batch_size=meta_batch_size,
    #         one_hot_labels=one_hot_labels,
    #         reshuffle_each_iteration=reshuffle_each_iteration,
    #         seed=seed,
    #     )
    #     steps_per_epoch = tf.data.experimental.cardinality(dataset)
    #     return dataset

    def get_only_outer_loop_update_layers(self):
        only_outer_loop_update_layers = set()
        for layer_name in (
            'conv1',
            'conv2',
            'conv3',
            'conv4',
            'bn1',
            'bn2',
            'bn3',
            'bn4',
            'channel_attention_0',
            'channel_attention_1',
            'channel_attention_2',
            'channel_attention_3',
            # 'classification_dense1'
        ):
            # only_outer_loop_update_layers.add(self.model.get_layer(layer_name))
            only_outer_loop_update_layers.add(layer_name)

        return only_outer_loop_update_layers


def run_domain_attention():
    train_domain_databases = [
        MiniImagenetDatabase(),
        OmniglotDatabase(random_seed=47, num_train_classes=1200, num_val_classes=100),
        DTDDatabase(),
        VGGFlowerDatabase()
    ]
    meta_train_domain_databases = [
        CUBDatabase(),
        # FungiDatabase(),
        # CUBDatabase(),
    ]

    test_database = CUBDatabase()

    da = DomainAttentionUnsupervised(
        train_databases=train_domain_databases,
        meta_train_databases=meta_train_domain_databases,
        database=test_database,
        test_database=test_database,
        network_cls=None,
        image_shape=(84, 84, 3),
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val=1,
        k_val_val=15,
        k_test=1,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=1,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=5000,
        meta_learning_rate=0.001,
        report_validation_frequency=1000,
        log_train_images_after_iteration=1000,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='domain_attention_freeze_attention_instance_supervised_cub_permutation_last_layer',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
    )

    da.train(iterations=15000)
    da.evaluate(iterations=5, num_tasks=1000, seed=42)


if __name__ == '__main__':
    # tf.config.experimental_run_functions_eagerly(True)

    # TODO Run experiments for VAE
    # TODO We do not need three fully connected layers
    # TODO We have to combine features in a better way
    # TODO Check whether attention is implemented correctly
    run_domain_attention()
