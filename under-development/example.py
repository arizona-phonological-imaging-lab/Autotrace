#!/usr/bin/env python3

from __future__ import absolute_import

import os
import logging
import h5py

import a3

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    roi = a3.ROI(140.,320.,250.,580.)
    if not os.path.isfile('test.hdf5'):
        a3.get_from_files('test_data','test.hdf5',roi,buff=10,scale=.1)
    if not os.path.isfile('train.hdf5'):
        with h5py.File('test.hdf5','r') as h:
            a3.get_from_files('apil-data/Interspeech2014_exp/',
                'train.hdf5',roi,scale=.1,blacklist=set(h['name']))
    a = a3.Autotracer('train.hdf5','test.hdf5',roi)
    a.train()
    a.save('example.a3.npy')
    with h5py.File('test.hdf5','r') as h:
        a.trace(a.X_valid,'traces.json',h['name'],'autotest','042')
