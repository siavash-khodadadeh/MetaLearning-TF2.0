from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import SimpleModel
from databases.myDataset import MySubClass
import sys


project_root_address = '/data/yali/sam/Project/MetaLearning-TF2.0-master/'
sys.path.insert(0, project_root_address)

def run_omniglot():
    my_database = MySubClass(
        random_seed=47,
        num_train_classes=4,
        num_val_classes=2,
    )

    maml = ModelAgnosticMetaLearningModel(
        database=my_database,
        network_cls=SimpleModel,
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val=1,
        k_val_val=15,
        k_test=1,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.4,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=50,
        log_train_images_after_iteration=200,
        num_tasks_val=100,
        clip_gradients=False,
        experiment_name='road_damage',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    maml.train(iterations=5000)
    maml.evaluate(iterations=50, num_tasks=100, use_val_batch_statistics=True, seed=42)


if __name__ == '__main__':
    run_omniglot()
