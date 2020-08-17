import tensorflow as tf
import tensorflow_hub as hub

from models.sml.sml import SML
from networks.maml_umtra_networks import MiniImagenetModel, VoxCelebModel
from databases import VoxCelebDatabase


def run_celeba():
    vox_celeb_database = VoxCelebDatabase()
    feature_model = hub.Module("https://tfhub.dev/google/speech_embedding/1")

    sml = SML(
        database=vox_celeb_database,
        network_cls=VoxCelebModel,
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val_val=15,
        k_val_test=15,
        k_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=15000,
        meta_learning_rate=0.001,
        n_clusters=20,
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
        experiment_name='voxceleb_embedding_features'
    )
    sml.train(iterations=60000)
    sml.evaluate(iterations=50, seed=42)


if __name__ == '__main__':
    run_celeba()
