import tensorflow as tf

import settings
from models.anil.anil import ANIL
from utils import combine_first_two_axes


class ANILUnsupervised(ANIL):
    def get_train_dataset(self):
        dataset = self.data_loader.get_unsupervised_dataset(
            self.database.train_folders,
            self.n,
            # k_ml,
            # k_validation,
            meta_batch_size=self.meta_batch_size,
            one_hot_labels=True,
            reshuffle_each_iteration=True,
            seed=-1,
        )
        steps_per_epoch = tf.data.experimental.cardinality(dataset)
        return dataset

    def get_losses_of_tasks_batch_eval(self, iterations, training):
        def euclidean_distance(a, b):
            # a.shape = N x D
            # b.shape = M x D
            N, D = tf.shape(a)[0], tf.shape(a)[1]
            M = tf.shape(b)[0]
            a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
            b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
            return tf.losses.cosine_similarity(a, b)

            # return tf.reduce_mean(tf.square(a - b), axis=2)

        def f(inputs):
            train_ds, val_ds, train_labels, val_labels = inputs
            train_ds = combine_first_two_axes(train_ds)
            val_ds = combine_first_two_axes(val_ds)

            self._initialize_eval_model()
            for i in range(iterations):
                self._train_model_for_eval(train_ds, train_labels)

            train_ds = self.eval_model.get_features(train_ds, training, apply_final_activation=False)
            # train_ds = self.eval_model.predict(train_ds, training)

            train_ds = tf.reshape(train_ds, (self.n, self.k_test, -1))
            train_ds = tf.reduce_mean(train_ds, axis=1)

            val_ds = self.eval_model.get_features(val_ds, training, apply_final_activation=False)
            # val_ds = self.eval_model.predict(val_ds, training)

            dists = euclidean_distance(train_ds, val_ds)
            log_p_y = tf.transpose(tf.nn.log_softmax(-dists))
            predicted_class_labels = tf.argmax(log_p_y, axis=-1)

            real_val_labels = tf.argmax(val_labels, axis=-1)

            val_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_class_labels, real_val_labels), tf.float32))
            val_loss = 0


            if settings.DEBUG:
                tf.print()
                tf.print(val_loss)
                tf.print(val_acc)
                tf.print()
            return val_acc, val_loss

        return f

    def outer_loss(self, labels, logits, inner_losses=None, method='train'):
        # if inner_losses is not None:
        # loss = inner_losses[0]
        # loss = tf.reduce_mean(
        #     tf.losses.categorical_crossentropy(labels, logits, from_logits=True)
        # )

        if method == 'train':
            logits = tf.linalg.normalize(logits, ord='euclidean', axis=1)[0]
            loss = -tf.abs(tf.linalg.det(logits))
        elif method == 'eval':
            loss = tf.reduce_mean(
                tf.losses.categorical_crossentropy(labels, logits, from_logits=True)
            )

        # losses = list()
        # import itertools
        # #
        # for permutation in itertools.permutations([0, 1, 2, 3, 4]):
        #     new_labels = tf.gather(labels, permutation, axis=1)
        #     loss = tf.reduce_mean(tf.losses.categorical_crossentropy(new_labels, logits, from_logits=True))
        #     losses.append(loss)
        #
        # return tf.reduce_min(losses)

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

        return loss
