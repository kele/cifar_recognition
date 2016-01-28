#!/usr/bin/env python

import lasagne
from cifar.data import DATA_SHAPE

from lasagne.layers import *
from lasagne.nonlinearities import *

from network_building.network_building import build_network

def build_cnn_network75(input_variable, batch_size=None):
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


def build_cnn_network75extended(input_variable, batch_size=None):
    layers = [
        {'type': Conv2DLayer,
         'args': {
             'num_filters': 64,
             'filter_size': (5, 5),
             'nonlinearity': rectify,
             'pad': 1}},

        {'type': DropoutLayer,
         'args': {'p': 0.25}},

        {'type': Conv2DLayer,
         'args': {
             'num_filters': 32,
             'filter_size': (5, 5),
             'nonlinearity': rectify,
             'pad': 1}},

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

        {'type': DropoutLayer,
         'args': {'p': 0.5}},

        {'type': DenseLayer,
         'args': {
             'num_units': 128,
             'nonlinearity': rectify}},

        {'type': DenseLayer,
         'args': {
             'num_units': 10,
             'nonlinearity': softmax}},
    ]

    return build_network(layers, input_variable, (batch_size,) + DATA_SHAPE), layers

# elu
def elu(x):
    return theano.tensor.switch(x > 0, x, theano.tensor.exp(x) - 1)

def build_cnn_network_from_paper(input_variable, batch_size=None):
    layers = [
        {'type': DropoutLayer,
         'args': {'p': 0.25}},

        {'type': Conv2DLayer,
         'args': {
             'num_filters': 96,
             'filter_size': (5, 5),
             'nonlinearity': elu,
             'pad': 1}},

        {'type': MaxPool2DLayer,
         'args': {'pool_size': (3, 3), 'stride': 2}},

        {'type': DropoutLayer,
         'args': {'p': 0.25}},

        {'type': Conv2DLayer,
         'args': {
             'num_filters': 192,
             'filter_size': (5, 5),
             'nonlinearity': elu,
             'pad': 1}},

        {'type': MaxPool2DLayer,
         'args': {'pool_size': (3, 3), 'stride': 2}},

        {'type': DropoutLayer,
         'args': {'p': 0.25}},

        {'type': Conv2DLayer,
         'args': {
             'num_filters': 192,
             'filter_size': (3, 3),
             'nonlinearity': elu}},

        {'type': Conv2DLayer,
         'args': {
             'num_filters': 192,
             'filter_size': (1, 1),
             'nonlinearity': elu}},

        {'type': MaxPool2DLayer,
         'args': {'pool_size': (2, 2)}},

        {'type': DropoutLayer,
         'args': {'p': 0.25}},

        {'type': DenseLayer,
         'args': {
             'num_units': 10,
             'nonlinearity': softmax}},
    ]

    return build_network(layers, input_variable, (batch_size,) + DATA_SHAPE), layers


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


def build_cnn_network2(input_variable, batch_size=None):
    layers = [
        {'type': Conv2DLayer,
         'args': {
             'num_filters': 64,
             'filter_size': (5, 5),
             'nonlinearity': rectify}},

        {'type': DropoutLayer,
         'args': {'p': 0.25}},

        {'type': Conv2DLayer,
         'args': {
             'num_filters': 64,
             'filter_size': (5, 5),
             'nonlinearity': rectify}},

        {'type': DropoutLayer,
         'args': {'p': 0.25}},

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
             'num_units': 512,
             'nonlinearity': rectify}},

        {'type': DenseLayer,
         'args': {
             'num_units': 10,
             'nonlinearity': softmax}},
    ]

    return build_network(layers, input_variable, (batch_size,) + DATA_SHAPE), layers


def build_cnn_network3(input_variable, batch_size=None):
    layers = [
        {'type': Conv2DLayer,
         'args': {
             'num_filters': 64,
             'filter_size': (5, 5),
             'nonlinearity': rectify}},

        {'type': DropoutLayer,
         'args': {'p': 0.25}},

        {'type': Conv2DLayer,
         'args': {
             'num_filters': 64,
             'filter_size': (5, 5),
             'nonlinearity': rectify}},

        {'type': DropoutLayer,
         'args': {'p': 0.25}},

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
             'num_units': 512,
             'nonlinearity': rectify}},

        {'type': DenseLayer,
         'args': {
             'num_units': 10,
             'nonlinearity': softmax}},
    ]

    return build_network(layers, input_variable, (batch_size,) + DATA_SHAPE), layers
