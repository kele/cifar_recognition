#!/usr/bin/env python

import engine
import theano.tensor as T


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
    network = cifar.network.build_cnn_network(input_var, batch_size=BATCH_SIZE)

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


if __name__ == '__main__':
    main_cifar()
