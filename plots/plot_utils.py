import pandas
import seaborn as sns
import matplotlib.pyplot as plt


def plot(data_addresses, colors, names, output_name):
    fig, ax = plt.subplots()

    for i, data_address in enumerate(data_addresses):
        df = pandas.read_json(data_address)
        df.columns = ['Wall Time', 'Step', 'Accuracy']
        decay = 0.1
        df['Accuracy'] = df['Accuracy'].ewm(span=(2 / decay) - 1, adjust=False).mean()

        plt.plot(df['Step'], df['Accuracy'])
        # sns.relplot(x='Step', y='Accuracy', kind='line', data=df, ax=ax, color=colors[i])

    ax.legend(names, facecolor='w')

    plt.show()
    # fig.savefig(output_name)
