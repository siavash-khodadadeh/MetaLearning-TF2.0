import tensorflow as tf

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import get_transfer_net


class TransferMetaLearningVGG16(ModelAgnosticMetaLearningModel):
    def __init__(self, num_trainable_layers, random_layer_initialization_seed=None, *args, **kwargs):
        self.num_trainable_layers = num_trainable_layers
        self.random_layer_initialization_seed = random_layer_initialization_seed

        super(TransferMetaLearningVGG16, self).__init__(*args, **kwargs)

    def get_network_name(self):
        return 'VGG16'

    def initialize_network(self):
        model = get_transfer_net(
            architecture='VGG16',
            num_trainable_layers=self.num_trainable_layers,
            num_classes=self.n,
            random_layer_initialization_seed=self.random_layer_initialization_seed
        )
        model(tf.zeros(shape=(1, 224, 224, 3)))
        return model

    def get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, (224, 224))
            image = tf.keras.applications.vgg16.preprocess_input(image)
            return image

        return parse_function
