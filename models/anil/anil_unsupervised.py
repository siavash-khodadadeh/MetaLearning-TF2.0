# This is the implementation of ANIL algorithm proposed in 
# Rapid learning or feature reuse? towards understanding the effectiveness of maml
# by Aniruddh Raghu, Maithra Raghu, Samy Bengio, Oriol Vinyals.
# Please cite this paper if you use this algorithm.
# @article{raghu2019rapid,
#   title={Rapid learning or feature reuse? towards understanding the effectiveness of maml},
#   author={Raghu, Aniruddh and Raghu, Maithra and Bengio, Samy and Vinyals, Oriol},
#   journal={arXiv preprint arXiv:1909.09157},
#   year={2019}
# }

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
