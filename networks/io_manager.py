import os
import random
import time
import threading

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
        self._file = h5py.File(file_name, 'r')


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

        # # This module contains 2 data pointers
        # # One is for the 'current' data, which is the ready-to-go data
        # # the other is for the 'next' data, which is read and preprocessed
        # # in a thread.

        # # when calling train_batch or test_batch, you will be delivered the current
        # # data, provided it is ready.  If it is not ready, you will get the next
        # # batch when it is delivered and the reading/processing of the next batch starts
        # # immediately.

        # self._current_train_batch = None
        # self._current_test_batch  = None
        # self._next_train_batch    = None
        # self._next_test_batch     = None


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

    def preprocess_batch(self):
        ''' Perform preprocessing operations on the batch
        '''

        # With 3 axes, and a batch size of N, we can flip the events
        # along any axis and any event randomly

        pass

    def dims(self):
        '''Return the dimensions of the image

        '''

        return {'image' : (self._batch_size, 192, 192, 192) ,
                'label' : (self._batch_size, 192, 192, 192) }
