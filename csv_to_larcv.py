import os
import h5py

from ROOT import larcv
import numpy

# this janky script converts a csv file into an h5 file.
# it also does larcv on the fly, if desired, to use the viewer

_f_name = "train_5-6.csv"
basename = os.path.splitext(_f_name)[0]

has_labels = True

_h5_out = h5py.File(basename+".h5", 'a')
_h5_out.create_dataset('data',  shape=(1,192,192,192),
                        maxshape=(5000,192,192,192),
                       compression='gzip', chunks=True)
_h5_out.create_dataset('label', shape=(1,192,192,192),
                        maxshape=(5000,192,192,192),
                       compression='gzip', chunks=True)

_larcv_out = basename + "_larcv.root"

larcv_io = larcv.IOManager(larcv.IOManager.kWRITE)
larcv_io.set_out_file(_larcv_out)
larcv_io.initialize()

larcv_meta = larcv.Voxel3DMeta()
larcv_meta.set(0,0,0,192,192,192,192,192,192)

stored_entries = 0

max_entries = 50

with open(_f_name, 'r') as _csv:
    labels = _csv.readline().split(',')
    print labels
    # current_entry = -1
    current_entry = 0

    current_data  = numpy.zeros((192,192,192))
    current_label = numpy.zeros((192,192,192))
    larcv_data  = larcv_io.get_data("sparse3d", "data")
    larcv_label = larcv_io.get_data("sparse3d", "label")
    larcv_data.meta(larcv_meta)
    larcv_label.meta(larcv_meta)

    for line in _csv.readlines():
        vals = line.split(',')
        entry = int(vals[0])
        if has_labels:
            label = int(float(vals[1]))
        value = float(vals[-4])
        x = int(vals[-1])
        y = int(vals[-2])
        z = int(vals[-3])


        if entry != current_entry:

            stored_entries += 1
            print "Saving entry " + str(entry)
            # H5 persistence:
            print _h5_out['data'].shape
            _h5_out['data'].resize((stored_entries, 192,192,192))
            _h5_out['label'].resize((stored_entries, 192,192,192))
            _h5_out['data'][current_entry]  = current_data
            _h5_out['label'][current_entry] = current_label

            print "h5 persisted"

            # larcv persistence:
            larcv_io.set_id(0,0,current_entry)
            larcv_io.save_entry()

            print "larcv persisted"


            # Reset data holders
            current_data  = numpy.zeros((192,192,192))
            current_label = numpy.zeros((192,192,192))
            larcv_data  = larcv_io.get_data("sparse3d", "data")
            larcv_label = larcv_io.get_data("sparse3d", "label")
            larcv_data.meta(larcv_meta)
            larcv_label.meta(larcv_meta)
            # Move entry counter:
            current_entry = entry

        current_data[x,y,z] = value
        if has_labels:
            current_label[x,y,z] = label

        larcv_data.emplace(x,y,z,value)
        if has_labels:
            larcv_label.emplace(x,y,z,label)


        if entry > max_entries:
            break

larcv_io.finalize()
_h5_out.close()