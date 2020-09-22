from models.maml.maml import ModelAgnosticMetaLearningModel


class ANIL(ModelAgnosticMetaLearningModel):
    def __init__(self, set_of_frozen_layers, *args, **kwargs):
        self.set_of_frozen_layers = set_of_frozen_layers
        super(ANIL, self).__init__(*args, **kwargs)

    def get_only_outer_loop_update_layers(self):
        return self.set_of_frozen_layers

    def train(self, iterations=5):
        print('Freezing layers for ANIL inner loop: ')
        print(self.set_of_frozen_layers)

        super(ANIL, self).train(iterations)
