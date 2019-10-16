import tensorflow as tf


def combine_first_two_axes(tensor):
    shape = tensor.shape
    return tf.reshape(tensor, (shape[0] * shape[1], *shape[2:]))


def average_gradients(tower_grads):
    average_grads = list()

    for grads in tower_grads:
        grad = tf.reduce_mean(grads, axis=0)
        average_grads.append(grad)

    return average_grads
