import os
import random
import time
import threading

import h5py
import numpy


# Here is how this class works.  There is a base class for read file operations, and
# two sub classes (one for read/write, one for read/preprocess)

# The core class will take exactly one file, spin up a thread to check to see if
# the data is consumed, and when it is, it begins reading another.
# The read in functionality is also (optionally) multi threaded

# The read/write class will add ability to write to a new file, and copy old data

# The preprocess class can preprocess data to do transformations (flips, etc)

class file_reader(object):

    def __init__(self, config):
        super(file_reader, self).__init__()

        self._config = config

        # make sure the file name is in the config file:
        if 'FILE' not in config:
            raise Exception("Must pass a file name")

        if 'BATCH_SIZE' not in config:
            raise Exception("Must pass a batch size")

        # Get a handle to the file (h5 format)
        self._file = h5py.File(config['FILE'], 'r')

        # See if there are min and max events to read:
        if 'MIN' in config:
            self._min = config['MIN']
        else:
            self._min = 0

        if 'MAX' in config:
            self._max = config['MAX']
        else:
            self._max = self._file['data'].shape[0]

        if 'NUM_THREADS' in config:
            self._num_threads = config['NUM_THREADS']
        else:
            self._num_threads = 1

        # Prepare data holders:
        self._current_batch_data = None
        self._next_batch_data = None

        self._reading = False

    def start_reader(self):

        # Start a thread to fill data:
        self._stopped = False
        self._thread = threading.Thread(target=self._fill_batch_data)
        self._thread.daemon = True
        self._thread.start()
        return

    def end_reader(self):
        self._stopped = True

    def status(self):
        print "Reading: " + str(self._reading)
        print "Stopped: " + str(self._stopped)

    def _fill_batch_data(self):

        # This thread runs in an infinte loop until stopped
        while not self._stopped:
            # Check to see if next batch is filled:
            if self._next_batch_data is None:
                self._next_batch_data = self._read_data(
                    n_events = self._config['BATCH_SIZE'],
                    n_threads = self._num_threads
                )
                if 'PREPROCESS' in self._config and self._config['PREPROCESS']:
                    self._preprocess(self._next_batch_data,self._num_threads)

            while self._reading:
                wait(0.1)

            # If current batch data is None, move next to current
            if self._current_batch_data is None:
                self._current_batch_data = self._next_batch_data
                self._next_batch_data    = None

            # Wait a little to get the next iteration:
            time.sleep(0.1)

    def _preprocess(self, batch_data, n_threads):

        # Preprocess will do random (but syncronized)
        # flipping of the data and labels.  It can flip X, Y, Z all at random

        step = len(batch_data['image']) / n_threads
        assert step*n_threads == len(batch_data['image'])


        t_list = []
        for i in range(n_threads):
            if 'label' in batch_data:
                t = threading.Thread(
                        target = self._chunk_flip,
                        args = (
                            batch_data['image'][i*step:(i+1)*step],
                            batch_data['label'][i*step:(i+1)*step],
                        )
                    )
            else:
                t = threading.Thread(
                        target = self._chunk_flip,
                        args = (
                            batch_data['image'][i*step:(i+1)*step],
                        )
                    )
            t.start()
            t_list.append(t)

        # Collect the threads:
        _ = [t.join() for t in t_list]



    # def _block_flip(self, chunk1, chunk2=None):
    def _chunk_flip(self, chunk1, chunk2=None):
        for i in range(len(chunk1)):
            axis = []
            if random.choice([True, False]):
                axis.append(0)
            if random.choice([True, False]):
                axis.append(1)
            if random.choice([True, False]):
                axis.append(2)
            chunk1[i,:] = numpy.flip(chunk1[i], axis=axis)
            if chunk2 is not None:
                chunk2[i,:] = numpy.flip(chunk2[i], axis=axis)

        return

    def _chunk_read(self, destination, dataset, start, stop):
        destination[:] = dataset[start:stop]
        return

    def _read_data(self, n_events, n_threads):

        self._reading = True
        # Read in from the file, taking a random chunk of data
        # Draw a random number from the range (min, max-batch_size):
        rand_start = random.randint(self._min, self._max - n_events)


        # Read the data:
        values = {}
        # Prepare arrays to hold the data:
        values['image'] = numpy.zeros((self._config['BATCH_SIZE'], 192, 192, 192))

        step = self._config['BATCH_SIZE'] / n_threads

        if step*n_threads != self._config['BATCH_SIZE']:
            raise Exception("Nthreads does not evenly divide batch size")

        t_list = []
        for i in range(n_threads):
            t = threading.Thread(
                    target = self._chunk_read,
                    args = (
                        values['image'][i*step:(i+1)*step],
                        self._file['data'],
                        rand_start + i*step,
                        rand_start+(i+1)*step,
                    )
                )
            t.start()
            t_list.append(t)

        # Collect the threads:
        _ = [t.join() for t in t_list]

        # values['image']  = self._file['data'][rand_start:rand_start+n_events]
        if 'label' in self._file.keys():
            values['label'] = numpy.zeros((self._config['BATCH_SIZE'], 192, 192, 192))

            t_list = []
            for i in range(n_threads):
                t = threading.Thread(
                        target = self._chunk_read,
                        args = (
                            values['label'][i*step:(i+1)*step],
                            self._file['label'],
                            rand_start + i*step,
                            rand_start+(i+1)*step,
                        )
                    )
                t.start()
                t_list.append(t)

            # Collect the threads:
            _ = [t.join() for t in t_list]


        # print values['image'].shape
        self._reading = False
        return values


    def consume_batch_data(self):
        while self._current_batch_data is None:
            time.sleep(0.1)

        ret = self._current_batch_data
        self._current_batch_data = None
        return ret

# class file_writer(object):
    # pass


class io_manager(file_reader):
    '''Provide a convienient interface for reading and preprocessing

    '''

    def __init__(self, config):
        super(io_manager, self).__init__(config)

        self._config = config

        # Right now, only supporting read mode
        if config['MODE'] not in ['TRAIN', 'TEST' 'ANA']:
            raise Exception("Need to specify training, testing or inference mode")


        # Create a file_reader for this file:




#     def write_ana_entries(self, values):
#         # Make sure that 'pred' is part of the values:
#         if 'entries' not in values:
#             raise Exception("missing entries")
#         if 'data' not in values:
#             raise Exception("missing data")
#         if 'pred' not in values:
#             raise Exception("missing pred")

#         # Validate we are writing the correct sequence of next entries:
#         first_entry = values['entries'][0]
#         last_entry = values['entries'][-1] + 1

#         n_written_entries = self._output['pred'].shape[0]
#         new_shape = (n_written_entries + self._batch_size, 192, 192, 192)

#         # Next, write the events into the file:
#         # self._output['data'].resize(new_shape)
#         self._output['data'][first_entry:last_entry] = values['data']
#         self._output['pred'][first_entry:last_entry] = values['pred']


#         if 'label' in self._output.keys() and 'label' in values:
#             self._output['label'][first_entry:last_entry] = values['label']


#         return

    def dims(self):
        '''Return the dimensions of the image

        '''

        return {'image' : (self._config['BATCH_SIZE'], 192, 192, 192) ,
                'label' : (self._config['BATCH_SIZE'], 192, 192, 192) }


    # def finalize(self):
    #     if self._output is not None:
    #         print "finalizing"
    #         self._output.close()