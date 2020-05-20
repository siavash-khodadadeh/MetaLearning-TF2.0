import os
from typing import Tuple

import tensorflow as tf

import settings

from .data_bases import Database


class VoxCelebDatabase(Database):
    def __init__(self, input_shape=(64000, 1)):
        super(VoxCelebDatabase, self).__init__(
            raw_database_address=settings.VOXCELEB_RAW_DATASEST_ADDRESS,
            database_address='',
            random_seed=-1,
            input_shape=input_shape
        )

    def _get_parse_function(self):
        def parse_function(example_address):
            audio_track, sample_rate = tf.audio.decode_wav(
                tf.io.read_file(example_address),
                desired_samples=64000,
                desired_channels=1)
            audio_track = tf.cast(audio_track, tf.float32)

            return audio_track

        return parse_function

    def get_classes(self, item):
        base_address = os.path.join(self.raw_database_address, item)
        ids = [class_id for class_id in os.listdir(base_address)]
        ids_addresses = [os.path.join(base_address, class_id) for class_id in os.listdir(base_address)]
        id_audios = dict()
        for i, id_address in enumerate(ids_addresses):
            audio_folders = [os.path.join(id_address, audio_folder) for audio_folder in os.listdir(id_address)]
            audio_files = list()
            for audio_folder in audio_folders:
                audio_files.extend([os.path.join(audio_folder, audios) for audios in os.listdir(audio_folder)])
                id_audios[ids[i]] = audio_files
        return id_audios

    def get_train_val_test_folders(self) -> Tuple:
        """Returns train, val and test folders as three lists or three dictionaries.
        Note that the python random seed might have been
        set here based on the class __init__ function."""

        dev_folders = self.get_classes('vox1_dev_wav/wav')
        test_folders = self.get_classes('vox1_test_wav/wav')

        return dev_folders, test_folders, test_folders
