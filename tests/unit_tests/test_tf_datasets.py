import unittest


class TestTFDatasets(unittest.TestCase):
    def test_remove_folders_with_smaller_than_k_files(self):
        OmniglotDatabase(random_seed=-1, num_train_classes=1200, num_val_classes=100)

        with self.assertRaises(Exception) as cm:
            self.get_dataset(
                random_seed=1,
                num_train_classes=1200,
                num_val_classes=100,
                n=7,
                k=12,
                mbs=3
            )
        self.assertTrue(cm.exception.__str__().startswith('The number of '))