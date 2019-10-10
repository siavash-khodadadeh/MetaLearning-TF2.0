from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, database, network_cls):
        self.database = database
        self.network_cls = network_cls
        self.train_dataset = self.get_train_dataset()
        self.val_dataset = self.get_val_dataset()
        self.test_dataset = self.get_test_dataset()

    @abstractmethod
    def train(self, epochs):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def get_train_dataset(self):
        pass

    @abstractmethod
    def get_val_dataset(self):
        pass

    @abstractmethod
    def get_test_dataset(self):
        pass

    @abstractmethod
    def get_config_info(self):
        pass
