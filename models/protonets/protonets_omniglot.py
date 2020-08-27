from databases import OmniglotDatabase
from models.protonets.proto_nets import PrototypicalNetworks
from networks.proto_networks import SimpleModelProto


def run_omniglot():
    omniglot_database = OmniglotDatabase(
        random_seed=47,
        num_train_classes=1200,
        num_val_classes=100,
    )

    proto_net = PrototypicalNetworks(
        database=omniglot_database,
        network_cls=SimpleModelProto,
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val=1,
        k_val_val=15,
        k_test=1,
        k_val_test=15,
        meta_batch_size=4,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,  # Set to -1 if you do not want to log train images.
        num_tasks_val=100,
        val_seed=-1,
        experiment_name=None
    )

    # proto_net.train(iterations=5000)
    proto_net.evaluate(-1, num_tasks=1000)


if __name__ == '__main__':
    run_omniglot()
