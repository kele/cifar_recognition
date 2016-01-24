#!/usr/bin/env python

import lasagne
from cifar.data import DATA_SHAPE

from lasagne.layers import *
from lasagne.nonlinearities import *

from network_building.network_building import build_network


def build_cnn_network(input_variable, batch_size=None):
    layers = [
        {'type': Conv2DLayer,
         'args': {
             'num_filters': 32,
             'filter_size': (5, 5),
             'nonlinearity': rectify}},

        {'type': MaxPool2DLayer,
         'args': {'pool_size': (2, 2)}},

        {'type': Conv2DLayer,
         'args': {
             'num_filters': 32,
             'filter_size': (5, 5),
             'nonlinearity': rectify}},

        {'type': MaxPool2DLayer,
         'args': {'pool_size': (2, 2)}},

        {'type': DropoutLayer,
         'args': {'p': 0.5}},

        {'type': DenseLayer,
         'args': {
             'num_units': 256,
             'nonlinearity': rectify}},

        {'type': DenseLayer,
         'args': {
             'num_units': 10,
             'nonlinearity': softmax}},
    ]

    return build_network(layers, input_variable, (batch_size,) + DATA_SHAPE), layers
