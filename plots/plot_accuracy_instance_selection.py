import os

import pandas
import seaborn as sns
import matplotlib.pyplot as plt


def plot(data_addresses, colors, names, output_name):
    fig, ax = plt.subplots()

    for i, data_address in enumerate(data_addresses):
        df = pandas.read_json(data_address)
        df.columns = ['Wall Time', 'Step', 'Accuracy']
        decay = 0.01
        df['Accuracy'] = df['Accuracy'].ewm(span=(2 / decay) - 1, adjust=False).mean()

        sns.relplot(x='Step', y='Accuracy', kind='line', data=df, ax=ax, color=colors[i])

    ax.legend(names, facecolor='w')

    plt.show()
    fig.savefig(output_name)


if __name__ == '__main__':
    imagenet_furthest_points_accuracy_address = os.path.join(
        'plots_data',
        'run-model-MiniImagenetModel_mbs-4_n-5_k-1_stp-5_furthest_point_seed30_logs_val-tag-Accuracy.json'
    )
    imagenet_random_accuracy_address = os.path.join(
        'plots_data',
        'run-model-MiniImagenetModel_mbs-4_n-5_k-1_stp-5_mini_imagenet_random_seed30_logs_val-tag-Accuracy.json'
    )
    imagenet_sp_address = os.path.join(
        'plots_data',
        'run-model-MiniImagenetModel_mbs-4_n-5_k-1_stp-5_mini_imagenet_sp_seed30_logs_val-tag-Accuracy.json'
    )
    colors = ['red', 'green', 'blue']
    names = ['Furthest Point', 'Random', 'SP']

    plot(
        [imagenet_furthest_points_accuracy_address, imagenet_random_accuracy_address, imagenet_sp_address],
        colors,
        names,
        output_name='Validation_Accuracy_Evolution_MiniImagenet.pdf'
    )

