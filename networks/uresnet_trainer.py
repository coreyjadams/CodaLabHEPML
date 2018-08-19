import os
import sys
import time

import numpy

import tensorflow as tf


import uresnet3d
import trainercore


class uresnet_trainer(trainercore.trainercore):

    def __init__(self, config):
        super(uresnet_trainer, self).__init__(config)

        if not self.check_params():
            raise Exception("Parameter check failed.")

        if '3d' in config['NAME']:
            net = uresnet3d.uresnet3d()
        else:
            net = uresnet.uresnet()

        net.set_params(config['NETWORK'])

        self.set_network_object(net)


    def fetch_minibatch_data(self, mode):
        # Return a dictionary object with keys 'image', 'label', and others as needed
        #            minibatch_data   = self._dataloaders['train'].fetch_data(
        #        self._config['TRAIN_CONFIG']['KEYWORD_DATA']).data()

        this_data = self._dataloader[mode].consume_batch_data()

        # If the weights for each pixel are to be normalized, compute the weights too:
        if self._config['NETWORK']['BALANCE_LOSS']:
            this_data['weight'] = self.compute_weights(this_data['label'])


        if 'DEBUG' in self._config['IO'][mode]:
            for i in range(len(this_data['label'])):
                print numpy.unique(this_data['label'][i])
                print numpy.mean(this_data['image'][i])
                if self._config['NETWORK']['BALANCE_LOSS']:
                    print numpy.unique(this_data['weight'][i])


        return this_data

    def fetch_minibatch_dims(self,mode):
        # Return a dictionary object with keys 'image', 'label', and others as needed
        # self._dataloaders['train'].fetch_data(keyword_label).dim() as an example
        this_dims = self._dataloader[mode].dims()


        # If the weights for each pixel are to be normalized, compute the weights too:
        if self._config['NETWORK']['BALANCE_LOSS']:
            this_dims['weight'] = this_dims['image']

        return this_dims



    def compute_weights(self, labels):
        # Take the labels, and compute the per-label weight

        # Prepare output weights:

        weights = numpy.zeros(labels.shape)

        # print "entering compute weights, batch_size: " + str(len(labels))

        i = 0
        for batch in labels:
            # First, figure out what the labels are and how many of each:
            values, counts = numpy.unique(batch, return_counts=True)

            n_pixels = numpy.sum(counts)
            for value, count in zip(values, counts):
                weight = 1.0*(n_pixels - count) / n_pixels
                # print "  B{i}, L{l}, weight: ".format(i=i, l=value) + str(weight)
                mask = labels[i] == value
                weights[i, mask] += weight

            # Normalize the weights to sum to 1 for each event:
            s =  numpy.sum(weights[i])
            if s < 0.001:
                weights[i] *= 0.0
                weights[i] += 1.0
            else:
                weights[i] *= 1. / s
            i += 1


        return weights


    def ana_step(self):

        # need to write an output hdf5 with the 'pred' column.
        # This
        minibatch_data = self.fetch_minibatch_data('ANA')


        softmax, metrics, doc = self.ana(minibatch_data)

        prediction = numpy.argmax(softmax, axis=-1)

        report_step  = self._iteration % self._config['REPORT_ITERATION'] == 0
        if report_step:
            self._report(metrics, doc)

        print "On entries " + str(minibatch_data['entries'])

        for entry in xrange(self._config['MINIBATCH_SIZE']):
            data = minibatch_data['image'][entry,:,:,:]
            zero_x, zero_y, zero_z  = numpy.where(data == 0.0)
            prediction[entry,zero_x, zero_y, zero_z] = 0

        output = {}
        output['data'] = minibatch_data['image']
        output['entries'] = minibatch_data['entries']
        output['pred'] = prediction

        # Save the entries to the output file:
        self._dataloader.write_ana_entries(output)


