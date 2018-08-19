import os
import sys
import time

import numpy

from io_manager import io_manager

import tensorflow as tf

class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, config):
        self._config        = config
        self._dataloaders   = dict()
        self._iteration     = 0
        self._batch_metrics = None
        self._output        = None

        self._core_training_params = [
            'MINIBATCH_SIZE',
            'SAVE_ITERATION',
            'LOGDIR',
            'RESTORE',
            'ITERATIONS',
            'IO',
            'TRAINING',
            'NETWORK'
        ]

        # Make sure that 'BASE_LEARNING_RATE' and 'TRAINING'
        # are in net network parameters:


        config['NETWORK']['BASE_LEARNING_RATE'] = config['BASE_LEARNING_RATE']
        config['NETWORK']['TRAINING'] = config['TRAINING']



    def check_params(self):
        for param in self._core_training_params:
            if param not in self._config:
                raise Exception("Missing paragmeter "+ str(param))
        return True


    def _report(self,metrics,descr):
        msg = ''
        for i,desc in enumerate(descr):
          if not desc: continue
          msg += '%s=%6.6f   ' % (desc,metrics[i])
        msg += '\n'
        sys.stdout.write(msg)
        sys.stdout.flush()


    def prepare_manager(self, mode):

        if 'IO' not in self._config:
            raise Exception("Missing IO config but trying to prepare manager.")
        else:
            start = time.time()

            config = self._config['IO'][mode]
            config['BATCH_SIZE'] = self._config['MINIBATCH_SIZE']
            config['ITERATIONS'] = self._config['ITERATIONS']
            config['MODE'] = mode

            self._dataloaders[mode] = io_manager(config)

            end = time.time()

            sys.stdout.write("Time to start {0} IO: {1:.2}s\n".format(mode, end - start))

        return

    def fetch_minibatch_data(self, mode):
        # Return a dictionary object with keys 'image', 'label', and others as needed
        #            minibatch_data   = self._dataloaders['train'].fetch_data(
        #        self._config['TRAIN_CONFIG']['KEYWORD_DATA']).data()

        raise NotImplementedError("Must implement fetch_minibatch_data in trainer.")

    def fetch_minibatch_dims(self, mode):
        # Return a dictionary object with keys 'image', 'label', and others as needed
        # self._dataloaders['train'].fetch_data(keyword_label).dim() as an example
        raise NotImplementedError("Must implement fetch_minibatch_dims in trainer.")

    def set_network_object(self, network):
        self._net = network

    def initialize(self):

        # Verify the network object is set:
        if not hasattr(self, '_net'):
            raise Exception("Must set network object by calling set_network_object() before initialize")


        # Prepare data manager:
        for mode in self._config['IO']:
            self.prepare_manager(mode)

        # Net construction:
        start = time.time()
        sys.stdout.write("Begin constructing network\n")

        # Make sure all required dimensions are present:

        # Either use TRAIN or ANA in the dim fetching
        if 'TRAIN' in self._dataloaders.keys():
            dims = self.fetch_minibatch_dims('TRAIN')
        else:
            dims = self.fetch_minibatch_dims('ANA')


        self._net.construct_network(dims=dims)


        end = time.time()
        sys.stdout.write("Done constructing network. ({0:.2}s)\n".format(end-start))


        # Configure global process (session, summary, etc.)
        self._sess = tf.Session()
        self._writer = tf.summary.FileWriter(self._config['LOGDIR'] + '/train/')
        self._saver = tf.train.Saver()

        if 'TEST' in self._config['IO'] and self._config['IO']['TEST']:
            self._writer_test = tf.summary.FileWriter(self._config['LOGDIR'] + '/test/')

        #
        # Network variable initialization
        #
        if not self._config['RESTORE']:
                self._sess.run(tf.global_variables_initializer())
                self._writer.add_graph(self._sess.graph)
        else:
            latest_checkpoint = tf.train.latest_checkpoint(self._config['LOGDIR']+"/train/checkpoints/")
            print "Restoring model from {}".format(latest_checkpoint)
            self._saver.restore(self._sess, latest_checkpoint)
            print "Successfully restored."

    def train_step(self):

        self._iteration = self._net.global_step(self._sess)
        report_step  = self._iteration % self._config['REPORT_ITERATION'] == 0
        summary_step = 'SUMMARY_ITERATION' in self._config and (self._iteration % self._config['SUMMARY_ITERATION']) == 0
        checkpt_step = 'SAVE_ITERATION' in self._config and (self._iteration % self._config['SAVE_ITERATION']) == 0 and self._iteration != 0

        # Nullify the gradients
        self._net.zero_gradients(self._sess)

        # Loop over minibatches
        for j in xrange(self._config['N_MINIBATCH']):

            minibatch_data = self.fetch_minibatch_data('TRAIN')


            res,doc = self._net.accum_gradients(sess   = self._sess,
                                                inputs = minibatch_data)



            if self._batch_metrics is None:
                self._batch_metrics = numpy.zeros((self._config['N_MINIBATCH'],len(res)-1),dtype=numpy.float32)
                self._descr_metrics = doc[1:]

            self._batch_metrics[j,:] = res[1:]

        # update
        self._net.apply_gradients(self._sess)

        # read-in test data set if needed (TEST = true, AND it's a report/summary step)
        test_data = None
        if (report_step or summary_step) and 'TEST' in self._config['IO']:

            # Read the next batch:
            # self._dataloaders['TEST'].next()


            test_data = self.fetch_minibatch_data('TEST')

        # Report
        if report_step:
            sys.stdout.write('@ iteration {}\n'.format(self._iteration))
            sys.stdout.write('Train set: ')
            self._report(numpy.mean(self._batch_metrics,axis=0),self._descr_metrics)
            if 'TEST' in self._config['IO'] and self._config['IO']['TEST']:
                res,doc = self._net.run_test(self._sess, test_data)
                sys.stdout.write('Test set: ')
                self._report(res,doc)

        # Save log
        if summary_step:
            # Run summary
            self._writer.add_summary(self._net.make_summary(self._sess, minibatch_data),
                                     self._iteration)
            if 'TEST' in self._config['IO'] and self._config['IO']['TEST']:
                self._writer_test.add_summary(self._net.make_summary(self._sess, test_data),
                                              self._iteration)

        # Save snapshot
        if checkpt_step:
            # Save snapshot
            ssf_path = self._saver.save(self._sess,
                self._config['LOGDIR']+"/train/checkpoints/save",
                global_step=self._iteration)
            sys.stdout.write('saved @ ' + str(ssf_path) + '\n')
            sys.stdout.flush()

    def ana(self, inputs):

        return  self._net.inference(sess   = self._sess,
                                    inputs = inputs)



    def ana_step(self):
        raise NotImplementedError("Must implement ana_step uniquely.")

    def batch_process(self):

        # Run iterations
        for i in xrange(self._config['ITERATIONS']):
            if self._config['TRAINING'] and self._iteration >= self._config['ITERATIONS']:
                print('Finished training (iteration %d)' % self._iteration)
                break

            # Start IO thread for the next batch while we train the network
            if self._config['TRAINING']:
                self.train_step()
            else:
                self.ana_step()
        if 'ANA' in self._dataloaders:
            self._dataloader['ANA'].finalize()
