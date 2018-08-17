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
    def __init__(self, file_name, io_mode, batch_size, validation_fraction=0.1, out_file=None):
        super(io_manager, self).__init__()


        # Right now, only supporting read mode
        if io_mode not in ['TRAIN', 'ANA']:
            raise Exception("Need to specify training or inference mode")


        # Open the file:
        self._file = h5py.File(file_name, 'r')
        if 'data' not in self._file.keys():
            raise Exception("Missing key data")

        if io_mode == 'ANA':
            if out_file is None:
                raise Exception("Must specify output file when doing ANA mode.")
            self._output = h5py.File(out_file, 'a')
            # Get the shape of the input dataset:
            shape = numpy.asarray(self._file['data'].shape)
            shape[0] = 4000
            print shape
            self._output.create_dataset('data',shape=shape, chunks=(1,192,192,192), compression='gzip')
            self._output.create_dataset('pred',shape=shape, chunks=(1,192,192,192), compression='gzip')
            if 'label' in self._file.keys():
                self._output.create_dataset('label', shape=shape, chunks=(1,192,192,192), compression='gzip')

            self._ana_entry = 0
        else:
            self._output = None


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

    def ana_batch(self):
        # For ana mode, read is purely sequential and in order.
        # Start at 0, and read batches until finished

        values = {}
        values['image']  = self._file['data'][self._ana_entry:self._ana_entry+self._batch_size]
        if 'label' in self._file.keys():
            values['label'] = self._file['label'][self._ana_entry:self._ana_entry+self._batch_size]

        values['entries'] = numpy.arange(self._ana_entry, self._ana_entry + self._batch_size)

        self._ana_entry += self._batch_size
        return values


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

    def write_ana_entries(self, values):
        # Make sure that 'pred' is part of the values:
        if 'entries' not in values:
            raise Exception("missing entries")
        if 'data' not in values:
            raise Exception("missing data")
        if 'pred' not in values:
            raise Exception("missing pred")

        # Validate we are writing the correct sequence of next entries:
        first_entry = values['entries'][0]
        last_entry = values['entries'][-1] + 1

        n_written_entries = self._output['pred'].shape[0]
        new_shape = (n_written_entries + self._batch_size, 192, 192, 192)

        # Next, write the events into the file:
        # self._output['data'].resize(new_shape)
        self._output['data'][first_entry:last_entry] = values['data']
        self._output['pred'][first_entry:last_entry] = values['pred']


        if 'label' in self._output.keys() and 'label' in values:
            self._output['label'][first_entry:last_entry] = values['label']


        return

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


    def finalize(self):
        if self._output is not None:
            print "finalizing"
            self._output.close()