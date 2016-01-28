#!/usr/bin/env python

import lasagne

def build_network(layers, input_variable, input_shape):
    network = lasagne.layers. \
        InputLayer(shape=input_shape,
                   input_var=input_variable)

    for lay in layers:
        network = lay['type'](network, **lay['args'])

    return network
