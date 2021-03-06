import os
import sys
import time

import numpy

import tensorflow as tf

from ROOT import larcv

import uresnet3d
# import uresnet, uresnet3d
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

        this_data = dict()
        this_data['image'] = self._dataloaders[mode].fetch_data(
            self._config['IO'][mode]['KEYWORD_DATA']).data()

        if 'KEYWORD_LABEL' in self._config['IO'][mode]:
            this_data['label'] = self._dataloaders[mode].fetch_data(
                self._config['IO'][mode]['KEYWORD_LABEL']).data()


            # If the weights for each pixel are to be normalized, compute the weights too:
            if self._config['NETWORK']['BALANCE_LOSS']:
                this_data['weight'] = self.compute_weights(this_data['label'])

        return this_data

    def fetch_minibatch_dims(self, mode):
        # Return a dictionary object with keys 'image', 'label', and others as needed
        # self._dataloaders['train'].fetch_data(keyword_label).dim() as an example
        this_dims = dict()
        this_dims['image'] = self._dataloaders[mode].fetch_data(
            self._config['IO'][mode]['KEYWORD_DATA']).dim()

        if 'KEYWORD_LABEL' in self._config['IO'][mode]:
            this_dims['label'] = self._dataloaders[mode].fetch_data(
                self._config['IO'][mode]['KEYWORD_LABEL']).dim()

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


        minibatch_data = self.fetch_minibatch_data('ANA')
        minibatch_dims = self.fetch_minibatch_dims('ANA')


        # Reshape any other needed objects:
        for key in minibatch_data.keys():
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])


        if 'label' in minibatch_data:
            softmax, metrics, doc = self.ana(minibatch_data)
        else:
            softmax = self.ana(minibatch_data)



        report_step  = self._iteration % self._config['REPORT_ITERATION'] == 0
        if report_step:
            print "Iteration: " + str(self._iteration)
            if 'label' in minibatch_data:
                self._report(metrics, doc)


        if self._output:

            # if report_step:
            #     print "Step {} - Acc all: {}, Acc non zero: {}".format(self._iteration,
            #         acc_all, acc_nonzero)

            # for entry in xrange(len(softmax)):
            #   self._output.read_entry(entry)
            #   data  = numpy.array(minibatch_data[entry]).reshape(softmax.shape[1:-1])
            entries   = self._dataloaders['ANA'].fetch_entries()
            event_ids = self._dataloaders['ANA'].fetch_event_ids()


            for entry in xrange(self._config['MINIBATCH_SIZE']):
                self._output.read_entry(entries[entry])

                if '3d' in self._config['NAME']:
                    larcv_data   = self._output.get_data("sparse3d","data")
                    larcv_track  = self._output.get_data("sparse3d","track")
                    larcv_shower = self._output.get_data("sparse3d","shower")
                    larcv_proton = self._output.get_data("sparse3d","proton")
                    data = minibatch_data['image'][entry,:,:,:,0]
                    nonzero_x, nonzero_y, nonzero_z  = numpy.where(data > 0.0)
                    indexes = (nonzero_x*larcv_data.meta().num_voxel_y() + nonzero_y) * larcv_data.meta().num_voxel_x() + nonzero_z
                    indexes = indexes.astype(dtype=numpy.uint64)

                    track_scores  = softmax[entry,:,:,:,1]
                    shower_scores = softmax[entry,:,:,:,2]
                    proton_scores = softmax[entry,:,:,:,3]

                    mapped_track_score  = track_scores[nonzero_x,nonzero_y,nonzero_z].astype(dtype=numpy.float32)
                    mapped_shower_score = shower_scores[nonzero_x, nonzero_y,nonzero_z].astype(dtype=numpy.float32)
                    mapped_proton_score = proton_scores[nonzero_x, nonzero_y,nonzero_z].astype(dtype=numpy.float32)

                    track_vs = larcv.as_tensor2d(mapped_track_score, indexes)
                    larcv_track.set(track_vs, larcv_data.meta())

                    shower_vs = larcv.as_tensor2d(mapped_shower_score, indexes)
                    larcv_shower.set(shower_vs, larcv_data.meta())

                    proton_vs = larcv.as_tensor2d(mapped_proton_score, indexes)
                    larcv_proton.set(proton_vs, larcv_data.meta())


                self._output.save_entry()

                if 'SAVE_ALL_LAYERS' in self._config and self._config['SAVE_ALL_LAYERS']:
                    ops_list, ops_names = self.inference_all_layers(minibatch_data)

                    print len(ops_list)


        self._dataloaders['ANA'].next(store_entries   = (not self._config['TRAINING']),
                                      store_event_ids = (not self._config['TRAINING']))


