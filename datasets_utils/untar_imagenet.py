import os
import glob
import tarfile

from tqdm import tqdm

import settings


def untar(fname, targetd_dir):
    with tarfile.open(fname) as tar:
        tar.extractall(path=targetd_dir)


def untar_all_files(dir_address):
    files = glob.glob(os.path.join(dir_address, '*.tar'))

    for f in tqdm(files):
        class_name = f[f.rindex('/') + 1:-4]
        class_dir = os.path.join(settings.IMAGENET_RAW_DATA_ADDRESS, 'ILSVRC2012_img_train_unzip', class_name)

        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        untar(f, class_dir)


if __name__ == '__main__':
    train_files_dir = os.path.join(settings.IMAGENET_RAW_DATA_ADDRESS, 'ILSVRC2012_img_train')
    untar_all_files(train_files_dir)