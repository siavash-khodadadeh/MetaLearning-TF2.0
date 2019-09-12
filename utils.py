import tensorflow as tf


def combine_first_two_axes(tensor):
    shape = tensor.shape
    return tf.reshape(tensor, (shape[0] * shape[1], *shape[2:]))


def average_gradients(tower_grads):
    average_grads = list()

    for grad in zip(*tower_grads):
        grads = tf.stack(grad)
        grads = tf.reduce_mean(grads, axis=0)
        average_grads.append(grads)

    return average_grads