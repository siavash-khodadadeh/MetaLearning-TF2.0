from databases import AirplaneDatabase, FungiDatabase
from models.crossdomainautoencoder.cross_domain_ae import CrossDomainAE
from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import MiniImagenetModel


def run_airplane():
    test_database = FungiDatabase()

    cdae = CrossDomainAE(
        database=test_database,
        batch_size=512,
        domains=('airplane', 'cub', 'dtd', 'miniimagenet', 'omniglot', 'vggflowers'),
        # domains=('airplane', 'cub', 'dtd', 'miniimagenet', 'omniglot', 'vggflowers'),
    )

    # cdae.train(epochs=20)
    cdae.evaluate(10, num_tasks=1000, k_test=5, k_val_test=15, inner_learning_rate=0.001, seed=42)


if __name__ == '__main__':
    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(True)
    run_airplane()
