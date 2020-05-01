import os

import tensorflow as tf
import numpy as np
from sklearn.decomposition import TruncatedSVD


def combine_first_two_axes(tensor):
    shape = tensor.shape
    return tf.reshape(tensor, (shape[0] * shape[1], *shape[2:]))


def average_gradients(tower_grads, losses):
    average_grads = list()

    for grads, loss in zip(tower_grads, losses):
        grad = tf.math.reduce_mean(grads, axis=0)
        average_grads.append(grad)

    return average_grads


def convert_grayscale_images_to_rgb(instances):
    """Gets a list of full path to images and replaces the ones which are grayscale with the same image but in RGB
    format."""
    counter = 0
    fixed_instances = list()
    for instance in instances:
        image = tf.image.decode_jpeg(tf.io.read_file(instance))

        if image.shape[2] != 3:
            print(f'Overwriting 2d instance with 3d data: {instance}')
            fixed_instances.append(instance)
            image = tf.squeeze(image, axis=2)
            image = tf.stack((image, image, image), axis=2)
            image_data = tf.image.encode_jpeg(image)
            tf.io.write_file(instance, image_data)
            counter += 1

    return counter, fixed_instances

def keep_keys_with_greater_than_equal_k_items(folders_dict, k):
    """Gets a dictionary and just keeps the keys which have greater than equal k items."""
    to_be_removed = list()
    for folder in folders_dict.keys():
        if len(folders_dict[folder]) < k:
            to_be_removed.append(folder)

    for folder in to_be_removed:
        del folders_dict[folder]


def get_folders_with_greater_than_equal_k_files(folders, k):
    to_be_removed = list()
    for folder in folders:
        if len(os.listdir(folder)) < k:
            to_be_removed.append(folder)

    for folder in to_be_removed:
        folders.remove(folder)

    return folders


def SP(data, K):
    A = data

    indices = np.random.choice(range(data.shape[1]), K, replace=False)

    indices = indices.astype(int)
    iter = 0

    for iter in range(0, K):
        k = iter % K
        inds = np.delete(np.copy(indices), k)
        A3 = A[:, inds]
        At = A - np.random.uniform(low=0.5, high=1) * np.matmul(np.matmul(A3, np.linalg.pinv(np.matmul(np.transpose(A3), A3))),
                           np.matmul(np.transpose(A3), A))

        # Compute just the first column from U and V
        svd = TruncatedSVD(n_components=1)
        svd.fit(np.transpose(At))
        # [U, S, V] = np.linalg.svd(At, full_matrices=False)
        # u1 = U[:, 0]
        # v = V[:, 1]
        u = svd.components_.reshape(-1)

        N = np.linalg.norm(At, axis=0)
        B = At / N
        B = np.transpose(B)

        Cr = np.abs(np.matmul(B, u))
        # ind = np.argsort(Cr)[::-1]
        # p = ind[0]
        p = np.argsort(Cr)[-1]
        indices[k] = p

    # ind2 = np.zeros(K - 1, );
    # for iter in range(1, 5):
    #     for k in range(0, K):
    #         ind2 = np.delete(inds, k)
    #         A3 = A[:, ind2]
    #         At = A - np.matmul(np.matmul(A3, np.linalg.pinv(np.matmul(np.transpose(A3), A3))),
    #                            np.matmul(np.transpose(A3), A))
    #         [U, S, V] = np.linalg.svd(At, full_matrices=False)
    #         u = U[:, 1]
    #         v = V[:, 1]
    #         N = np.linalg.norm(At, axis=0)
    #         B = At / N
    #         B = np.transpose(B)
    #         Cr = np.abs(np.matmul(B, u))
    #         ind = np.argsort(Cr)[::-1]
    #         p = ind[0]
    #         inds[k] = p

    return indices


def SP_deterministic(data, K):
    A = data
    At = data
    inds = np.zeros(K, )
    inds = inds.astype(int)
    iter = 0
    for k in range(0, K):
        iter = iter + 1
        # Compute just the first column from U and V
        svd = TruncatedSVD(n_components=1)
        svd.fit(np.transpose(At))
        # [U, S, V] = np.linalg.svd(At, full_matrices=False)
        # u1 = U[:, 0]
        # v = V[:, 1]
        u = svd.components_.reshape(-1)
        N = np.linalg.norm(At, axis=0)
        B = At / N
        B = np.transpose(B)
        Cr = np.abs(np.matmul(B, u))
        ind = np.argsort(Cr)[::-1]
        p = ind[0]
        inds[k] = p
        A3 = A[:, inds[0:k + 1]]
        At = A - np.matmul(np.matmul(A3, np.linalg.pinv(np.matmul(np.transpose(A3), A3))),
                           np.matmul(np.transpose(A3), A))
    # ind2 = np.zeros(K - 1, )
    # for iter in range(1, 5):
    #     for k in range(0, K):
    #         ind2 = np.delete(inds, k)
    #         A3 = A[:, ind2]
    #         At = A - np.matmul(np.matmul(A3, np.linalg.pinv(np.matmul(np.transpose(A3), A3))),
    #                            np.matmul(np.transpose(A3), A))
    #         [U, S, V] = np.linalg.svd(At, full_matrices=False)
    #         u = U[:, 1]
    #         v = V[:, 1]
    #         N = np.linalg.norm(At, axis=0)
    #         B = At / N
    #         B = np.transpose(B)
    #         Cr = np.abs(np.matmul(B, u))
    #         ind = np.argsort(Cr)[::-1]
    #         p = ind[0]
    #         inds[k] = p

    return inds


def SSP_with_random_validation_set(features, labels, K, delta=20):
    label_values = np.unique(labels)
    num_classes = len(label_values)

    label_matrix = np.zeros((len(label_values), len(labels)))
    for i, label in enumerate(labels):
        label_matrix[label, i] = delta

    A = np.concatenate((features, label_matrix), axis=0)
    At = np.copy(A)

    inds = np.zeros(num_classes * K, )
    inds = inds.astype(int)
    iter = 0

    counter = 0
    chosen_indices = list()

    for k in range(0, K // 2):
        iter = iter + 1
        # Compute just the first column from U and V
        svd = TruncatedSVD(n_components=1)
        svd.fit(np.transpose(At))
        # [U, S, V] = np.linalg.svd(At, full_matrices=False)
        # u1 = U[:, 0]
        # v = V[:, 1]
        u = svd.components_.reshape(-1)
        new_At = At[:4096, :]
        N = np.linalg.norm(new_At, axis=0)
        B = new_At / N
        B = np.transpose(B)
        Cr = np.abs(np.matmul(B, u[:4096]))

        for label_value in label_values:
            x = np.multiply(Cr, A[features.shape[0] + label_value, ...])
            ind = np.argsort(x)
            inds[label_value * K // 2 + counter] = np.random.choice((ind[-1], ind[-2], ind[-3], ind[-4]), 1, p=(0.5, 0.3, 0.1, 0.1))
            chosen_indices.append(inds[label_value * K // 2 + counter])
            validation_choices = np.array(np.where(x != 0)).reshape((-1, ))
            inds[label_value * K // 2 + counter + 2 * num_classes] = np.random.choice(validation_choices, 1)

        counter += 1
        # return inds

        if k != K // 2 - 1:
            A3 = A[:, chosen_indices]
            At = A - np.matmul(np.matmul(A3, np.linalg.pinv(np.matmul(np.transpose(A3), A3))),
                               np.matmul(np.transpose(A3), A))

    # print(inds)
    return inds


def SSP(features, labels, K, delta=10):
    label_values = np.unique(labels)
    num_classes = len(label_values)

    label_matrix = np.zeros((len(label_values), len(labels)))
    for i, label in enumerate(labels):
        label_matrix[label, i] = delta

    A = np.concatenate((features, label_matrix), axis=0)
    At = np.copy(A)

    inds = np.zeros(num_classes * K, )
    inds = inds.astype(int)
    iter = 0

    counter = 0
    for k in range(0, K):
        iter = iter + 1
        # Compute just the first column from U and V
        svd = TruncatedSVD(n_components=1)
        svd.fit(np.transpose(At))
        # [U, S, V] = np.linalg.svd(At, full_matrices=False)
        # u1 = U[:, 0]
        # v = V[:, 1]
        u = svd.components_.reshape(-1)
        N = np.linalg.norm(At, axis=0)
        B = At / N
        B = np.transpose(B)
        Cr = np.abs(np.matmul(B, u))

        for label_value in label_values:
            x = np.multiply(Cr, A[features.shape[0] + label_value, ...])
            ind = np.argsort(x)[::-1]
            inds[counter] = np.random.choice((ind[0], ind[1], ind[2], ind[3]), 1, p=(0.5, 0.3, 0.1, 0.1))
            counter += 1

        A3 = A[:, inds[0:counter + 1]]
        At = A - np.matmul(np.matmul(A3, np.linalg.pinv(np.matmul(np.transpose(A3), A3))),
                           np.matmul(np.transpose(A3), A))

    return inds


if __name__ == '__main__':
    features = np.random.rand(4096, 12000)
    labels = [0] * 2000 + [1] * 4000 + [2] * 2600 + [3] * 2000 + [4] * 1400

    while True:

        indices = SSP_with_random_validation_set(features, labels, 4)
        print(indices)
        if indices[0] == indices[2] or indices[1] == indices[3] or indices[2] == indices[4]:
            break
    print(indices)


    # data = np.random.rand(40, 73)
    # A = data
    #
    # indices = SP(data, 5)
    # A3 = A[:, indices]
    # At = A - np.matmul(np.matmul(A3, np.linalg.pinv(np.matmul(np.transpose(A3), A3))),
    #                    np.matmul(np.transpose(A3), A))
    #
    # norm = np.linalg.norm(At)
    # print(norm)
    #
    # for test_case in range(1000):
    #     rand_numbers = np.random.randint(0, 73, size=5)
    #     A3 = A[:, rand_numbers]
    #     At = A - np.matmul(np.matmul(A3, np.linalg.pinv(np.matmul(np.transpose(A3), A3))),
    #                        np.matmul(np.transpose(A3), A))
    #     current_norm = np.linalg.norm(At)
    #
    #     print(current_norm)
    #     assert(current_norm >= norm)
    #
    # print(norm)
    # indices = SP_deterministic(data, 5)
    # A3 = A[:, indices]
    # At = A - np.matmul(np.matmul(A3, np.linalg.pinv(np.matmul(np.transpose(A3), A3))),
    #                    np.matmul(np.transpose(A3), A))
    #
    # print(np.linalg.norm(At))

