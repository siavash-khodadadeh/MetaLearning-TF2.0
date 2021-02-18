import tensorflow as tf
import numpy as np
from tqdm import tqdm


def squared_dist(A):
    expanded_a = tf.expand_dims(A, 1)
    expanded_b = tf.expand_dims(A, 0)
    distances = tf.reduce_sum(tf.math.squared_difference(expanded_a, expanded_b), 2)
    return distances

def main(threshold):
    N = 5
    LATENT_DIM = 512
    META_BATCH_SIZE = 4
    NUM_ITERATIONS = 60000
    NUM_TRIALS = NUM_ITERATIONS * META_BATCH_SIZE

    counter = 0
    tri_mask = np.ones(N ** 2, dtype=np.bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False

    for _ in tqdm(range(NUM_TRIALS)):
        vectors = tf.random.normal((N, LATENT_DIM))
        dist = squared_dist(vectors)
        elements = tf.boolean_mask(dist, tri_mask)
        if tf.reduce_min(elements) < threshold:
            counter += 1
    return counter, NUM_TRIALS


if __name__ == '__main__':
    for stddev in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2):
        a = tf.random.normal((5, 512))
        b = a + tf.random.normal(shape=a.shape, mean=0, stddev=stddev)
        print('-----------------------------------')
        print(f'stddev: {stddev}')
        threshold = tf.reduce_sum(tf.square((b - a) * (b - a)))
        print(f'threshold: {threshold}')
        counter, num_trials = main(threshold)
        print(f'{counter} / {num_trials}')
        print('-----------------------------------')

