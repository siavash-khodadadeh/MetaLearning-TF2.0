import tensorflow as tf
import numpy as np

'''An issue is that after combined with attention, half of base_model would
maintain two nodes (with and without attention), which is inefficient.'''

class MiniImagenetModel(tf.keras.Model):
    name = 'MiniImagenetModel'
    # can be replaced with other solver network

    def __init__(self, num_classes):
        super(MiniImagenetModel, self).__init__(name='mini_imagenet_model')
        self.conv1 = tf.keras.layers.Conv2D(64, 3, name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn1')
        self.max_pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool1')
        self.relu1 = tf.keras.layers.ReLU(name='relu1')

        self.conv2 = tf.keras.layers.Conv2D(64, 3, name='conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn2')
        self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool2')
        self.relu2 = tf.keras.layers.ReLU(name='relu2')

        self.conv3 = tf.keras.layers.Conv2D(64, 3, name='conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn3')
        self.max_pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool3')
        self.relu3 = tf.keras.layers.ReLU(name='relu3')

        self.conv4 = tf.keras.layers.Conv2D(64, 3, name='conv4')
        self.bn4 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn4')
        self.max_pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool4')
        self.relu4 = tf.keras.layers.ReLU(name='relu4')

        self.flatten = tf.keras.layers.Flatten(name='flatten')

        self.dense = tf.keras.layers.Dense(num_classes, activation=None, name='dense')

    def call(self, inputs, training=False):
        output = inputs

        output = self.conv1(output)
        output = self.bn1(output)
        output = self.max_pool1(output)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.max_pool2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.max_pool3(output)
        output = self.relu3(output)

        output = self.conv4(output)
        output = self.bn4(output)
        output = self.max_pool4(output)
        output = self.relu4(output)

        output = self.flatten(output)
        output = self.dense(output)

        return output

class AttentionModel(tf.keras.Model):
    name = 'AttentionModel'

    def __init__(self, num_features=64):
        super(AttentionModel, self).__init__(name='attention_model')
        self.conv1 = tf.keras.layers.Conv2D(16, 3, name='at_conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='at_bn1')
        self.max_pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='at_max_pool1')
        self.relu1 = tf.keras.layers.ReLU(name='at_relu1')

        self.conv2 = tf.keras.layers.Conv2D(16, 3, name='at_conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='at_bn2')
        self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='at_max_pool2')
        self.relu2 = tf.keras.layers.ReLU(name='at_relu2')

        self.conv3 = tf.keras.layers.Conv2D(16, 3, name='at_conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='at_bn3')
        self.max_pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='at_max_pool3')
        self.relu3 = tf.keras.layers.ReLU(name='at_relu3')

        self.dense = tf.keras.layers.Dense(num_features, activation='softmax', name='at_dense')
#         self.flatten = tf.keras.layers.Flatten(name='at_flatten')

    def call(self, inputs, training=False):
        output = inputs

        output = self.conv1(output)
        output = self.bn1(output)
        output = self.max_pool1(output)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.max_pool2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.max_pool3(output)
        output = self.relu3(output)

        output = self.dense(output)
#         output = self.flatten(output)
        return output

def decompose_attention_model(attention, input_shape=(84, 84, 3)):
    input = tf.keras.Input(shape=input_shape, name='at_input')

    output = input
    for l in attention.layers:
        output = l(output)

    attention = tf.keras.models.Model(input, outputs=output, name='AttentionModel')
    return attention

class Combine(tf.keras.layers.Layer):
    # apply attention to the solver

    def __init__(self, **kwargs):
        super(Combine, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(Combine, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        '''
        a: attention weights
        b: activations in the solver
        '''
        a, b = x
        a = tf.reduce_mean(tf.reshape(a, (-1, a.shape[-1])), 0)
#         while a.get_shape().ndims > 1:
#             a = tf.reduce_mean(a, axis=0)
        return b * a

    def compute_output_shape(self, input_shape):
        # consistent with the shape of solver activations
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return shape_b

def assemble_model(attention, solver, ind, inner_trainable=False):
    # attention is multiplied to the output of ind-th layer

    layers = solver.layers
    output = Combine(name='combine')([attention.output, solver.layers[ind].output])

    for i in range(ind + 1, len(layers)):
        output = layers[i](output)
        setattr(layers[i], 'inner_trainable', True)

    assembled_model = tf.keras.models.Model(
        inputs=[attention.input, solver.input],
        outputs=output,
        name='AssembledModel'
    )

    for layer in assembled_model.layers:
        if not hasattr(layer, 'inner_trainable'):
            setattr(layer, 'inner_trainable', inner_trainable)

    return assembled_model


if __name__ == '__main__':
    def get_assembled_model(num_classes, ind, architecture=MiniImagenetModel, input_shape=(84, 84, 3)):
        attention_model = AttentionModel()
        attention_model = decompose_attention_model(attention_model, input_shape)

        base_model = architecture(num_classes)
        base_input = tf.keras.Input(shape=input_shape, name='base_input')
        base_model(base_input)

        return attention_model, base_model, assemble_model(attention_model, base_model, ind)

    attention_model, base_model, assembled_model = get_assembled_model(
        num_classes = 5,
        ind = 7
    )

    attention_model.summary()
    base_model.summary()
    assembled_model.summary()

    print('\n--------------------')
    print('test output')
    input1 = tf.keras.backend.random_uniform((2, 84, 84, 3), minval=0, maxval=1)
    input2 = tf.keras.backend.random_uniform((4, 84, 84, 3), minval=0, maxval=1)
    print(assembled_model([input1, input2]).numpy())

    print('\n--------------------')
    print('trainable during inner loop')
    for layer in assembled_model.layers:
        print(layer.name, layer.inner_trainable)