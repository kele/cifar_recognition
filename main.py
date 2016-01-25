#!/usr/bin/env python

import engine
import theano.tensor as T
import lasagne
import datetime
import time


def save_results(layers, params, filename, training_schedule):
    import pickle
    pickle.dump((layers, params, training_schedule), open(filename, 'w'))


def main():
    import cifar.network
    BATCH_SIZE = 200

    full_datastream = cifar.data.load_datastream(BATCH_SIZE)
    small_datastream = cifar.data.load_datastream(BATCH_SIZE,
                                                  training_set_size=4000,
                                                  validation_set_size=1000)

    input_var = T.tensor4('inputs')
    targets_var = T.ivector('targets')

    print('Building the network...')
    network, layers = cifar.network.build_cnn_network2(input_var, batch_size=BATCH_SIZE)

    print('Network:')
    for l in layers:
        print '  ', l['type'], '\n    ', l['args']

    print('Learning schedule:')
    schedule = [
        {'num_epochs': 10,
         'hyperparams': {
            'learning_rate': 0.01,
            'momentum': 0.95,
            'weight_decay': 0.0001
            }},

        {'num_epochs': 30,
         'hyperparams': {
            'learning_rate': 0.005,
            'momentum': 0.9,
            'weight_decay': 0.0001
            }},

        {'num_epochs': 100,
         'hyperparams': {
            'learning_rate': 0.001,
            'momentum': 0.9,
            'weight_decay': 0.0001
            }}
    ]

    for s in schedule:
        print '  ', s

    print('Starting training...')
    for s in schedule:
        _, test_acc = \
            engine.train(
                input_var=input_var,
                targets_var=targets_var,
                data=datastream,
                network=network,
                verbose=2,
                patience=10,
                **s)
    print('Training finished')

    t = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d__%H-%M-%S')
    save_results(layers, params=lasagne.layers.get_all_param_values(network),
                 training_schedule=schedule,
                 filename='saved_nets/{:.2f}accuracy_{}.params'.format(test_acc, t))

if __name__ == '__main__':
    main()
