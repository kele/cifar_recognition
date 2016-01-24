#!/usr/bin/env python

import lasagne
from cifar.data import DATA_SHAPE


def build_multilayer_network(input_variable, batch_size=None):
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


# TODO: make this more general, use **kwargs for specifying the net architecture
def build_cnn_network(input_variable, batch_size=None):
    network = lasagne.layers. \
        InputLayer(shape=(batch_size,) + DATA_SHAPE,
                   input_var=input_variable)

    network = lasagne.layers. \
        Conv2DLayer(network,
                    num_filters=128,
                    filter_size=(5, 5),
                    nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers. \
        MaxPool2DLayer(network,
                       pool_size=(2, 2))

    network = lasagne.layers. \
        Conv2DLayer(network,
                    num_filters=128,
                    filter_size=(5, 5),
                    nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers. \
        MaxPool2DLayer(network,
                       pool_size=(2, 2))

    network = lasagne.layers. \
        DropoutLayer(network,
                     p=0.5)

    network = lasagne.layers. \
        DenseLayer(network,
                   num_units=256,
                   nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers. \
        DenseLayer(network,
                   num_units=10,
                   nonlinearity=lasagne.nonlinearities.softmax)

    return network
