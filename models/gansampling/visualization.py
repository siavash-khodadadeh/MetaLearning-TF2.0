import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.lines as mlines

from utils import combine_first_two_axes

gan_generator = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']
k = 1
k_val_ml = 1


def tf_image_translate(images, tx, ty, interpolation='NEAREST'):
    transforms = [1, 0, -tx, 0, 1, -ty, 0, 0]
    return tfa.image.transform(images, transforms, interpolation)


def generate_all_vectors_by_interpolation(latent_dim, tf_seed=3, n=5):
    if tf_seed is not None:
        tf.random.set_seed(tf_seed)
    class_vectors = tf.random.normal((n, latent_dim))
    class_vectors = class_vectors / tf.reshape(tf.norm(class_vectors, axis=1), (class_vectors.shape[0], 1))
    vectors = list()

    vectors.append(class_vectors)
    for i in range(k + k_val_ml - 1):

        new_vectors = class_vectors
        noise = tf.random.normal(shape=class_vectors.shape, mean=0, stddev=1)
        noise = noise / tf.reshape(tf.norm(noise, axis=1), (noise.shape[0], 1))

        new_vectors = new_vectors + (noise - new_vectors) * 0.4

        vectors.append(new_vectors)

    return vectors, noise



def get_task(resolution=(25, 25, 3), tf_seed=3, n=5):
    # return (train_ds, val_ds), (train_labels, val_labels)
    latent_dim = 512  # 100 for omniglot
    generated_image_shape = resolution  # (28, 28, 1) for omniglot

    vectors, noise_vectors = generate_all_vectors_by_interpolation(latent_dim, tf_seed, n=n)

    vectors = tf.reshape(tf.stack(vectors, axis=0), (-1, latent_dim))

    images = gan_generator(vectors)['default']
    images = tf.image.resize(images, generated_image_shape[:2])

    train_ds = images[:n * k]
    train_indices = [i // k + i % k * n for i in range(n * k)]
    train_ds = tf.gather(train_ds, train_indices, axis=0)
    train_ds = tf.reshape(train_ds, (n, k, *generated_image_shape))

    val_ds = images[n * k:]
    val_indices = [i // k_val_ml + i % k_val_ml * n for i in range(n * k_val_ml)]
    val_ds = tf.gather(val_ds, val_indices, axis=0)
    val_ds = tf.reshape(val_ds, (n, k_val_ml, *generated_image_shape))

    val_ds = combine_first_two_axes(val_ds)
    val_ds = tf.reshape(val_ds, (n, k_val_ml, *generated_image_shape))

    noise_images = gan_generator(noise_vectors)['default']
    noise_images = tf.image.resize(noise_images, generated_image_shape[:2])

    train_labels = np.repeat(np.arange(n), k)
    val_labels = np.repeat(np.arange(n), k_val_ml)
    train_labels = tf.one_hot(train_labels, depth=n)
    val_labels = tf.one_hot(val_labels, depth=n)

    yield (train_ds, val_ds), (train_labels, val_labels), vectors, noise_vectors, noise_images


def imscatter(x, y, images, texts, textcolor, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()

    x, y = np.atleast_1d(x, y)
    counter = 0
    artists = []
    for x0, y0 in zip(x, y):
        im = OffsetImage(images[counter], zoom=zoom)
        plt.text(x0 - 2, y0 + 10, texts[counter], color=textcolor)
        counter += 1
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def draw_line(ax, point1, point2, im1, im2, im3, color='green'):
    l = mlines.Line2D(
        (point1[0], point2[0]), (point1[1], point2[1]),
        linewidth=1,
        linestyle='dashed',
        color=color
    )
    new_point = point1 + (point2 - point1) * 0.4
    plt.scatter(new_point[0], new_point[1], color='red')

    imscatter(
        [point1[0], point2[0], new_point[0]],
        [point1[1], point2[1], new_point[1]],
        [im1, im3, im2],
        ['(tr)', '(no)', '(val)'],
        textcolor=color
    )

    ax.add_line(l)


# Plot 1
n = 3
for item in get_task(n=n):
    (generated_image, val_generated_image), (_, _), vectors, noise_vectors, noise_images = item
    # input_vector = tf.random.normal([self.n, 100], mean=0, stddev=1)
    # generated_image = self.gan_generator.predict(input_vector)
    generated_image = generated_image[:, 0, :, :, :]
    val_generated_image = val_generated_image[:, 0, :, :, :]

    fig, ax = plt.subplots(1, 1)
    # randomly choose 10 points

    seed = 9
    np.random.seed(seed)
    points = np.random.uniform(low=20, high=180, size=4 * n).reshape(-1, 2)
    plt.scatter(points[:, 0], points[:, 1])

    colors = ['green', 'red', 'blue', 'gray', 'black']

    for i in range(n):
        draw_line(
            ax,
            points[i * 2],
            points[i * 2 + 1],
            generated_image[i],
            val_generated_image[i],
            noise_images[i],
            color=colors[i]
        )

    ax.axis('off')
    plt.show()

fig.savefig(os.path.expanduser('~/Desktop/latent_space_visualization.png'))

# Plot 2
n = 5
seed = 3
for item in get_task(resolution=(256, 256, 3), tf_seed=seed, n=n):
    (generated_image, val_generated_image), (_, _), vectors, noise_vectors, noise_images = item
    generated_image = generated_image[:, 0, :, :, :]
    val_generated_image = val_generated_image[:, 0, :, :, :]
    fig, axes = plt.subplots(3, n)
    for i in range(n):
        image = generated_image[i, :, :, :]
        axes[0, i].imshow(image)
        # axes[0, i].axis('off')

    for i in range(n):
        noise_image = noise_images[i, :, :, :]
        axes[1, i].imshow(noise_image)  # cmap='gray' for omniglot
        # axes[1, i].axis('off')

    for i in range(n):
        val_image = val_generated_image[i, :, :, :]
        axes[2, i].imshow(val_image)  # cmap='gray' for omniglot
        # axes[2, i].axis('off')

    for i in range(3):
        for j in range(n):
            axes[i, j].set_xticklabels([])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticklabels([])
            axes[i, j].set_yticks([])

    axes[0, 0].set_ylabel('Train', rotation=90, size='large')
    axes[1, 0].set_ylabel('Noise', rotation=90, size='large')
    axes[2, 0].set_ylabel('Validation', rotation=90, size='large')

    plt.show()

fig.savefig(os.path.expanduser('~/Desktop/latent_space_visualization2.png'))

print('Figures are saved in ~/Desktop/ as latent_space_visualization and latent_space_visualization2')
