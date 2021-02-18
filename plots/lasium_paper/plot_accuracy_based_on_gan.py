import cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np


def plot_img(img_name, location, index, zoom=0.1):
    plt.scatter(index, accs[epochs.index(index)] + 1, color='#0D7377', linewidths=0.5, marker='v')
    plt.plot((index, location[0]), (accs[epochs.index(index)], location[1]), '--', color='#0D7377', alpha=1)
    img = plt.imread(f'./gan_images/{img_name}.png')
    img = cv2.resize(img, (350, 350))
    # img = img[50:-50, 50:-50, :]
    ax = plt.gca()
    im = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(im, location, xycoords='data', frameon=True, pad=0.2)
    ax.add_artist(ab)
    ax.update_datalim(np.column_stack(list(location)))
    ax.autoscale()
    return ab


def smooth_data(accs, weight):
    last = accs[0]
    for i in range(1, len(accs)):
        accs[i] = last * weight + (1 - weight) * accs[i]
        last = accs[i]
    return accs


epochs = [0, 10, 20, 30, 40, 50, 100, 150, 200, 300, 400, 500]
accs = [51.95, 67.50, 71.26, 77.34, 77.67, 77.35, 78.14, 79.99, 78.21, 77.94, 80.51, 76.49]
accs = smooth_data(accs, 0.7)
accs_ci = [0.66, 0.71, 0.68, 0.62, 0.63, 0.64, 0.63, 0.60, 0.63, 0.64, 0.60, 0.67]
training_from_scratch = [51.64] * len(accs)

bottom = [acc - ci for acc, ci in zip(accs, accs_ci)]
top = [acc + ci for acc, ci in zip(accs, accs_ci)]

plt.plot(epochs, accs, color='b', label='LASIUM-N')
plt.plot(epochs, bottom, '--', color='#32E0C4', alpha=0.2)
plt.plot(epochs, top, '--', color='#32E0C4', alpha=0.2)
plt.plot(epochs, training_from_scratch, '--', color='r', alpha=0.5, label='baseline')
plt.fill_between(epochs, bottom, top, color='#32E0C4', alpha=.1)
plt.xticks([10, 30, 50, 100, 200, 300, 400, 500])
plt.xlabel('# GAN training epochs', fontsize=14)
plt.yticks([40, 50, 60, 70, 80, 100])
plt.ylabel('Accuracy (%)', fontsize=14)

# plt images

plot_img('00_4', location=(10, 85), index=0)

plot_img('10_4', location=(40, 90), index=10)
plot_img('30_4', location=(70, 85), index=30)
plot_img('50_4', location=(100, 90), index=50)
plot_img('100_4', location=(130, 85), index=100)
plot_img('200_4', location=(190, 90), index=200)
plot_img('300_4', location=(300, 85), index=300)
plot_img('400_4', location=(400, 90), index=400)
plot_img('500_4', location=(500, 85), index=500)

plt.scatter(
    0, accs[epochs.index(0)] + 1, color='#0D7377', linewidths=0.5, marker='v', label='Generated image at epoch'
)

plt.subplots_adjust(bottom=0.1, top=0.9, right= 0.98, left=0.1)

plt.legend(loc='best')
# plt.show()
plt.savefig('./outputs/accuracy_based_on_gan.pdf')
