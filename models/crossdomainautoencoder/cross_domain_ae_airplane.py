from databases import AirplaneDatabase, FungiDatabase, EuroSatDatabase, ISICDatabase, CUBDatabase, DTDDatabase, \
    MiniImagenetDatabase, Omniglot84x84Database, VGGFlowerDatabase
from models.crossdomainautoencoder.cross_domain_ae2 import CrossDomainAE2
from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import MiniImagenetModel


def run_airplane():
    test_database = AirplaneDatabase()

    cdae = CrossDomainAE2(
        database=test_database,
        batch_size=512,
        # domains=('fungi', ),
        # domains=('airplane', 'fungi', 'cub', 'dtd', 'miniimagenet', 'omniglot', 'vggflowers'),
        domains=('fungi', 'cub', 'dtd', 'miniimagenet', 'omniglot', 'vggflowers'),
        # domains=('cub', 'miniimagenet', 'vggflowers'),
        # domains=('fungi', 'cub', 'dtd', 'miniimagenet', 'omniglot', 'vggflowers'),
    )

    experiment_name = 'all_domains_288'

    cdae.train(epochs=20, experiment_name=experiment_name)
    cdae.evaluate(
        10,
        num_tasks=1000,
        k_test=1,
        k_val_test=15,
        inner_learning_rate=0.001,
        experiment_name=experiment_name,
        seed=42
    )


if __name__ == '__main__':
    # import tensorflow as tf
    # tf.config.experimental_run_functions_eagerly(True)
    run_airplane()
