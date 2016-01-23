#!/usr/bin/env python

import mnist.network
import engine
import theano.tensor as T


def main():
    BATCH_SIZE = 200

    input_var = T.tensor4('inputs')
    results_var = T.ivector('results')

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
        results_var=results_var,
        data=mnist.data.load_datastream(BATCH_SIZE),
        network=network,
        hyperparams=hyperparams,
        num_epochs=20,
        verbose=True)
    print('Training finished')


if __name__ == '__main__':
    main()
