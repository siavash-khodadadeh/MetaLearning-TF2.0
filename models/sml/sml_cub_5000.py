import tensorflow as tf

from models.sml.sml import SML
from networks.maml_umtra_networks import SimpleModel, MiniImagenetModel
from databases import CUBDatabase


def run_cub():
    cub_database = CUBDatabase()
    base_model = tf.keras.applications.VGG19(weights='imagenet')
    feature_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[24].output)

    sml = SML(
        database=cub_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=3000,
        meta_learning_rate=0.001,
        n_clusters=5000,
        feature_model=feature_model,
        # feature_size=288,
        feature_size=4096,
        input_shape=(224, 224, 3),
        preprocess_function=tf.keras.applications.vgg19.preprocess_input,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=True,
        report_validation_frequency=250,
        experiment_name='cub_imagenet_features'
    )
    # sml.train(iterations=6000)
    sml.evaluate(iterations=50, seed=42, iterations_to_load_from=3000)


if __name__ == '__main__':
    run_cub()
