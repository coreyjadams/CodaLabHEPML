import os
import random

import h5py
import numpy


class io_manager(object):
    '''This class is responsible for reading and writing to h5 files
    for this competition

    This class can open a file, read batches, and write to an output file.
    '''
    def __init__(self, file_name, io_mode, batch_size, validation_fraction=0.1):
        super(io_manager, self).__init__()


        # Right now, only supporting read mode
        if io_mode not in ['TRAIN', 'ANA']:
            raise Exception("Need to specify training or inference mode")

        if io_mode != 'TRAIN': raise Exception()

        # Open the file:
        self._file = h5py.File(file_name)


        if 'data' not in self._file.keys():
            raise Exception("Missing key data")

        self._max_entries = self._file['data'].shape[0]

        if batch_size > self._max_entries * validation_fraction:
            raise Exception("Batch size can not exceed {}".format(self._max_entries*validation_fraction))

        if io_mode == 'TRAIN' and 'label' not in self._file.keys():
            raise Exception("Missing labels in training file")

        self._batch_size = batch_size

        if io_mode == 'TRAIN':
            self._validation_start = int(self._max_entries*validation_fraction)



    def train_batch(self):

        # Take a random batch of images:
        random_seed = random.randint(0, self._validation_start - self._batch_size)

        # Get that chunk of data:
        values = {}
        values['image']  = self._file['data'][random_seed:random_seed+self._batch_size]
        values['label'] = self._file['label'][random_seed:random_seed+self._batch_size]


        return values

    def test_batch(self):

        # Take a random batch of images:
        random_seed = random.randint(self._validation_start, self._max_entries - self._batch_size)

        # Get that chunk of data:
        values = {}
        values['image']  = self._file['data'][random_seed:random_seed+self._batch_size]
        values['label'] = self._file['label'][random_seed:random_seed+self._batch_size]


        return values


    def dims(self):
        '''Return the dimensions of the image

        '''

        return {'image' : (self._batch_size, 192, 192, 192) ,
                'label' : (self._batch_size, 192, 192, 192) }
