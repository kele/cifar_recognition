#!/usr/bin/env python

import engine
import theano.tensor as T
import lasagne
import datetime
import time


def save_results(layers, params, filename):
    import pickle
    pickle.dump((layers, params), open(filename, 'w'))


def main_mnist():
    import mnist.network
    BATCH_SIZE = 200

    input_var = T.tensor4('inputs')
    targets_var = T.ivector('targets')

    # MNIST
    hyperparams = {
        'learning_rate': 0.01,
        'momentum': 0.9
    }

    print('Building the network...')
    network = mnist.network.build_network(input_var, batch_size=BATCH_SIZE)

    print('Starting training...')
    engine.train(
        input_var=input_var,
        targets_var=targets_var,
        data=mnist.data.load_datastream(BATCH_SIZE),
        network=network,
        hyperparams=hyperparams,
        num_epochs=20,
        verbose=2,
        patience=4)
    print('Training finished')


def main_cifar():
    import cifar.network
    BATCH_SIZE = 200

    input_var = T.tensor4('inputs')
    targets_var = T.ivector('targets')

    # CIFAR
    hyperparams = {
        'learning_rate': 0.01,
        'momentum': 0.9
    }

    print('Building the network...')
    network, layers = cifar.network.build_cnn_network(input_var, batch_size=BATCH_SIZE)

    print('Network:')
    for l in layers:
        print '  ', l['type'], '\n    ', l['args']

    print('Starting training...')
    _, test_acc = \
        engine.train(
            input_var=input_var,
            targets_var=targets_var,
            data=cifar.data.load_datastream(BATCH_SIZE),
            network=network,
            hyperparams=hyperparams,
            num_epochs=100,
            verbose=1,
            patience=4)
    print('Training finished')

    t = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d__%H-%M-%S')
    save_results(layers, params=lasagne.layers.get_all_param_values(network),
                 filename='saved_nets/{:.2f}accuracy_{}.params'.format(test_acc, t))

if __name__ == '__main__':
    main_cifar()
