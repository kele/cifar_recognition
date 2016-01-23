#!/usr/bin/env python

import lasagne
from mnist.data import DATA_SHAPE


def build_network(input_variable, batch_size=None):
    network = lasagne.layers. \
    InputLayer(shape=(batch_size,) + DATA_SHAPE,
               input_var=input_variable)

    network = lasagne.layers. \
    DropoutLayer(network,
                 p=0.2)

    network = lasagne.layers. \
    DenseLayer(network,
               num_units=800,
               nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers. \
    DropoutLayer(network,
                 p=0.5)

    network = lasagne.layers. \
    DenseLayer(network,
               num_units=800,
               nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers. \
    DropoutLayer(network,
                 p=0.5)

    network = lasagne.layers. \
    DenseLayer(network,
               num_units=10,
               nonlinearity=lasagne.nonlinearities.softmax)

    return network
