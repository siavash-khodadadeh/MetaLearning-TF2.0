import os
import numpy as np
import tensorflow as tf

from databases import MiniImagenetDatabase, ISICDatabase, ChestXRay8Database, PlantDiseaseDatabase
from databases.meta_dataset import DTDDatabase


def convert_str_to_img(item):
    img = tf.io.read_file(item)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(img, tf.float32)
    if img.shape[2] == 1:
        img = tf.squeeze(img, axis=-1)
        img = tf.stack((img, img, img), axis=-1)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    img = tf.image.resize(img, (224, 224))
    return img


def get_all_instances(database):
    res_strs = []
    if isinstance(database.train_folders, list):
        for folder in database.train_folders:
            for item in os.listdir(folder):
                res_strs.append(os.path.join(folder, item))

    else:
        for class_name, instances in database.train_folders.items():
            res_strs.extend(instances)

    res = []
    for item in res_strs:
        # item = convert_str_to_img(item)
        res.append(item)

    return res


vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=False)
vgg19.trainable = False
style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
content_layers = ['block5_conv2']

outputs = [vgg19.get_layer(style_layer_name).output for style_layer_name in style_layers]
model = tf.keras.models.Model(inputs=vgg19.inputs, outputs=outputs)
def convert_to_activaitons(imgs):
    imgs_activations = []
    for image in imgs:
        image = convert_str_to_img(image)[np.newaxis, ...]
        activations = model.predict(image)
        imgs_activations.append(activations)
    return imgs_activations


DTDDatabase()
d1_imgs = get_all_instances(MiniImagenetDatabase())
d2_imgs = get_all_instances(ISICDatabase())
d3_imgs = get_all_instances(ChestXRay8Database())
d4_imgs = get_all_instances(PlantDiseaseDatabase())

d1_imgs = np.random.choice(d1_imgs, 10, replace=False)
d2_imgs = np.random.choice(d2_imgs, 10, replace=False)
d3_imgs = np.random.choice(d3_imgs, 10, replace=False)
d4_imgs = np.random.choice(d4_imgs, 10, replace=False)

print(d1_imgs)
print(d2_imgs)
print(d3_imgs)
print(d4_imgs)

d1_imgs_activations = convert_to_activaitons(d1_imgs)
d2_imgs_activations = convert_to_activaitons(d2_imgs)
d3_imgs_activations = convert_to_activaitons(d3_imgs)
d4_imgs_activations = convert_to_activaitons(d4_imgs)



