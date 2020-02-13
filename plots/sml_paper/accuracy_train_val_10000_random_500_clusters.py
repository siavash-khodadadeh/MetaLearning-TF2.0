from plots.plot_utils import plot
import os


if __name__ == '__main__':
    base_address = '../plots_data/sml'
    accuracy = os.path.join(
        base_address,
        'run-sml_model-MiniImagenetModel_mbs-4_n-5_k-1_stp-5_mini_imagenet_model_feature_10000_clusters_500_logs_train-tag-Accuracy.json'
    )
    accuracy_val = os.path.join(
        base_address,
        'run-sml_model-MiniImagenetModel_mbs-4_n-5_k-1_stp-5_mini_imagenet_model_feature_10000_clusters_500_logs_val-tag-Accuracy.json'
    )
    colors = ['red', 'green']
    names = ['Train', 'Validation']

    plot(
        [accuracy, accuracy_val],
        colors,
        names,
        output_name='Accuracy.pdf'
    )

