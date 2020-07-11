import tensorflow as tf

from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import VGGSmallModel
from databases import VGGFace2Database


class MAMLVGGFace2(ModelAgnosticMetaLearningModel):
    def get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, (224, 224))
            image = tf.cast(image, tf.float32)
            return (image - 127.5) / 128.0

        return parse_function

    def initialize_network(self):
        model = self.network_cls(num_classes=self.n)
        model(tf.zeros(shape=(1, 224, 224, 3)))
        return model


def run_vggface2():
    vggface_database = VGGFace2Database()
    maml = MAMLVGGFace2(
        database=vggface_database,
        network_cls=VGGSmallModel,
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
        meta_learning_rate=0.0001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='vgg_face2_conv128_mlr_0.0001'
    )

    # maml.train(iterations=500000)
    maml.evaluate(50, num_tasks=1000, seed=42)


if __name__ == '__main__':
    run_vggface2()
