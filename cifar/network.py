#!/usr/bin/env python

import lasagne
from cifar.data import DATA_SHAPE

from lasagne.layers import *
from lasagne.nonlinearities import *

from network_building.network_building import build_network


# elu
def elu(x):
    return theano.tensor.switch(x > 0, x, theano.tensor.exp(x) - 1)


def build(layers, input_var, batch_size):
    return build_network(layers, input_var, (batch_size,) + DATA_SHAPE)


best_network = [
    {'type': DropoutLayer,
        'args': {'p': 0.25}},

    {'type': Conv2DLayer,
        'args': {
            'num_filters': 128,
            'filter_size': (5, 5),
            'nonlinearity': rectify}},

    {'type': Conv2DLayer,
        'args': {
            'num_filters': 64,
            'filter_size': (5, 5),
            'nonlinearity': rectify}},

    {'type': MaxPool2DLayer,
        'args': {'pool_size': (2, 2)}},

    {'type': DropoutLayer,
        'args': {'p': 0.5}},

    {'type': Conv2DLayer,
        'args': {
            'num_filters': 64,
            'filter_size': (5, 5),
            'nonlinearity': rectify}},

    {'type': MaxPool2DLayer,
        'args': {'pool_size': (3, 3), 'stride': 2}},

    {'type': DropoutLayer,
        'args': {'p': 0.5}},

    {'type': FlattenLayer,
        'args': {'outdim': 2}},

    {'type': DenseLayer,
        'args': {
            'num_units': 800,
            'nonlinearity': rectify}},

    {'type': DropoutLayer,
        'args': {'p': 0.5}},

    {'type': DenseLayer,
        'args': {
            'num_units': 256,
            'nonlinearity': rectify}},

    {'type': DropoutLayer,
        'args': {'p': 0.5}},

    {'type': DenseLayer,
        'args': {
            'num_units': 10,
            'nonlinearity': softmax}},
]
