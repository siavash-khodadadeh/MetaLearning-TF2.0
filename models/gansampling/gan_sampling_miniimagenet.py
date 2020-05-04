from models.gansampling.gan_sampling import GANSampling
from networks.maml_umtra_networks import MiniImagenetModel
from databases import MiniImagenetDatabase


def run_mini_imagenet():
    mini_imagenet_database = MiniImagenetDatabase()

    gan_sampling = GANSampling(
        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        k_val_ml=5,
        k_val_train=1,
        k_val_val=15,
        k_val_test=15,
        k_test=1,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=50,
        log_train_images_after_iteration=250,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=True,
        experiment_name='mini_imagenet_interpolation_std_1.2_shift_5',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    gan_sampling.train(iterations=60000)
    gan_sampling.evaluate(iterations=50, use_val_batch_statistics=True, seed=42, iterations_to_load_from=16000)


if __name__ == '__main__':
    run_mini_imagenet()
