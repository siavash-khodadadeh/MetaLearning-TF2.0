import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

epochs = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]) * 16
accs_mean_maml = [0.3302000164985657, 0.32919999957084656,  0.3354000151157379, 0.39420002698898315, 0.38420000672340393, 0.3540000319480896, 0.4254000186920166, 0.43140000104904175]
accs_stds_maml = [0.19911794364452362, 0.19664017856121063, 0.1935635358095169, 0.19941505789756775, 0.20811143517494202, 0.20833627879619598, 0.21511587500572205, 0.2115042507648468]


accs_mean_sp = [0.3325999975204468, 0.42100000381469727, 0.46000000834465027, 0.4182000160217285, 0.41360002756118774, 0.4267999827861786, 0.41840001940727234, 0.4034000039100647]
accs_stds_sp = [0.1913563460111618,  0.20899523794651031, 0.20119643211364746, 0.20849162340164185, 0.2066277712583542, 0.2124188393354416, 0.21293529868125916, 0.20733654499053955]


def plot(x, means, stds, label):
    bands = np.array(stds) * 1.96 / np.sqrt(1000)
    plt.errorbar(x, means, yerr=bands, label=label, fmt="--o")
    plt.ylim(0.3, 0.5)


if __name__ == '__main__':
    plot(epochs, accs_mean_maml, accs_stds_maml, label='MAML')
    plot(epochs, accs_mean_sp, accs_stds_sp, label='Ours')
    plt.legend(loc='upper left', fontsize=16)
    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.show()
    plt.imshow

