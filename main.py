#!/usr/bin/env python

import engine
import theano.tensor as T
import lasagne
import datetime
import time

import pickle

def save_results(layers, params, filename, training_schedule):
    pickle.dump((layers, params, training_schedule), open(filename, 'w'))


def main():
    import cifar.network
    BATCH_SIZE = 100

    full_datastream = cifar.data.load_datastream(BATCH_SIZE)
    small_datastream = cifar.data.load_datastream(BATCH_SIZE,
                                                  training_set_size=4000,
                                                  validation_set_size=1000)
    tiny_datastream = cifar.data.load_datastream(BATCH_SIZE,
                                                 training_set_size=2000,
                                                 validation_set_size=500)

    input_var = T.tensor4('inputs')
    targets_var = T.ivector('targets')

    print('Building the network...')
    network, layers = cifar.network.build_cnn_network_from_paper(input_var, batch_size=BATCH_SIZE)

    #print('Loading old network')
    #_, prm, _ = pickle.load(open('saved_nets/partial_49.30accuracy_2016-01-26__01-18-01.params'))
    #lasagne.layers.set_all_param_values(network, prm)
    #print('Network loaded!')

    print('Network:')
    for l in layers:
        print '  ', l['type'], '\n    ', l['args']

    for basic_lr in [0.01, 0.005, 0.001]:
        print('Learning schedule:')
        schedule = [
            {'num_epochs': 20,
            'hyperparams': {
                'learning_rate': basic_lr,
                'weight_decay': 0.001,
                'momentum': 0.9,
                }},
            {'num_epochs': 20, # tutaj jeszcze przed chwila bylo 5
            'hyperparams': {
                'learning_rate': basic_lr / 2,
                'weight_decay': 0.001,
                'momentum': 0.9,
                }},
            {'num_epochs': 20,
            'hyperparams': {
                'learning_rate': basic_lr / 10,
                'weight_decay': 0.001,
                'momentum': 0.9,
                }},
            {'num_epochs': 20,
            'hyperparams': {
                'learning_rate': basic_lr / 10 / 2,
                'weight_decay': 0.001,
                'momentum': 0.9,
                }},
            {'num_epochs': 20,
            'hyperparams': {
                'learning_rate': basic_lr / 100,
                'weight_decay': 0.001,
                'momentum': 0.9,
                }},

            {'num_epochs': 20,
            'hyperparams': {
                'learning_rate': basic_lr / 100 / 2,
                'weight_decay': 0.001,
                'momentum': 0.9,
                }},

            {'num_epochs': 20,
            'hyperparams': {
                'learning_rate': basic_lr / 1000,
                'weight_decay': 0.001,
                'momentum': 0.9,
                }},
        ]

        for s in schedule:
            print '  ', s

        print('Starting training...')

        schedule_passed = []
        for s in schedule:
            print 'Current schedule: ', s
            _, test_acc = \
                engine.train(
                    input_var=input_var,
                    targets_var=targets_var,
                    data=full_datastream,
                    network=network,
                    verbose=1,
                    patience=5,
                    **s)

            print('Saving results in 5 seconds...')
            time.sleep(5)
            schedule_passed += s

            t = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d__%H-%M-%S')
            save_results(layers, params=lasagne.layers.get_all_param_values(network),
                        training_schedule=schedule_passed,
                        filename='saved_nets/partial_{:.2f}accuracy_lr{}_{}.params'.format(
                            test_acc, s['hyperparams']['learning_rate'], t))
        print('Training finished')

if __name__ == '__main__':
    main()
