#!/usr/env/bin python

import time
import theano
import theano.tensor as T
import lasagne
import lasagne.regularization
from lasagne.regularization import l2

import numpy as np


def train(input_var, targets_var, data, network, hyperparams,
          num_epochs=100, verbose=0, patience=5, validate_per_batches=None,
          max_iters=None, augment=None):

    if verbose:
        print('Compiling stuff...')

    learning_rate = theano.shared(np.array(hyperparams['learning_rate'], dtype=theano.config.floatX))

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, targets_var)
    loss = loss.mean()

    if 'weight_decay' in hyperparams:
        loss = loss \
            + (lasagne.regularization.regularize_network_params(network, l2)
               * hyperparams['weight_decay'])

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=learning_rate,
        momentum=hyperparams['momentum'])

    train_fn = theano.function([input_var, targets_var], loss, updates=updates)

    # No dropout here!
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss_fn = lasagne.objectives.categorical_crossentropy(test_prediction, targets_var)
    test_loss_fn = test_loss_fn.mean()

    test_acc_fn = T.mean(T.eq(T.argmax(test_prediction, axis=1), targets_var),
                         dtype=theano.config.floatX)

    val_fn = theano.function([input_var, targets_var], [test_loss_fn, test_acc_fn])

    if verbose:
        print('Finally training starts!')

    prev_best = 0
    best_val_acc = 0
    stall_count = 0
    best_params = lasagne.layers.get_all_param_values(network)

    iteration_count = 0

    for epoch in range(num_epochs):
        if max_iters and iteration_count >= max_iters:
            break

        if verbose:
            print('Epoch {} of {}:'.format(epoch + 1, num_epochs))

        try:
            start_time = time.time()

            # TODO: reuse the code more
            # Training
            train_loss = 0
            train_batches = 0

            epoch_iterator = data['train'].get_epoch_iterator()
            for inputs, targets in epoch_iterator:

                if augment:
                    inputs = augment(inputs, train=True)

                iteration_count += 1

                current_train_loss = train_fn(inputs, targets.ravel())
                train_loss += current_train_loss
                train_batches += 1

                decay = np.array(1.0 / (max(1.0, 1.5 ** (iteration_count / 4000))),
                                 dtype=theano.config.floatX)
                learning_rate.set_value(
                    np.array(hyperparams['learning_rate'], dtype=theano.config.floatX) * decay)

                if verbose >= 2:
                    print('  [{:5}] training loss: {:10.6f} | avg: {:10.6f}'.format(
                        iteration_count, float(current_train_loss), train_loss / train_batches))

                if max_iters and iteration_count >= max_iters:
                    break

            train_loss = train_loss / train_batches

            # Validation
            val_loss = 0
            val_acc = 0
            val_batches = 0
            for inputs, targets in data['validation'].get_epoch_iterator():
                if augment:
                    inputs = augment(inputs, train=False)

                loss, acc = val_fn(inputs, targets.ravel())
                val_loss += loss
                val_acc += acc
                val_batches += 1

                if verbose >= 2:
                    print('  validation accuracy: {:.2f}'.format(float(acc)))

            val_acc = (val_acc * 100.0) / val_batches
            val_loss = val_loss / val_batches

            prev_best = best_val_acc
            best_val_acc = max(best_val_acc, val_acc)

            if best_val_acc - prev_best < 0.01:
                stall_count += 1
            else:
                best_params = lasagne.layers.get_all_param_values(network)
                stall_count = 0

            stop_time = time.time()
            delta_time = stop_time - start_time


            if verbose:
                print('  --------------------------------')
                print('  took {:.2f}s'.format(delta_time))
                print('  learning_rate {:10.10f}'.format(float(learning_rate.get_value().ravel()[0])))
                print('  training loss (avg): {:10.6f}'.format(train_loss))
                print('  validation loss: {:10.6f}'.format(val_loss))
                print('  validation accuracy: {:.2f}'.format(val_acc))
                print('  best validation acc so far: {:.2f}'.format(best_val_acc))
                print('  stall_count: {}'.format(stall_count))

            if stall_count >= patience:
                break

        except KeyboardInterrupt:
            print('Interrupted! Going directly to testing.')
            break

    lasagne.layers.set_all_param_values(network, best_params)
    # Test
    test_loss = 0
    test_acc = 0
    test_batches = 0
    for inputs, targets in data['test'].get_epoch_iterator():
        if augment:
            inputs = augment(inputs, train=False)

        loss, acc = val_fn(inputs, targets.ravel())
        test_loss += loss
        test_acc += acc
        test_batches += 1

    test_acc = test_acc / test_batches * 100.0
    test_loss = test_loss / test_batches

    if verbose:
        print('Final results:')
        print('  test loss: {:10.6f}'.format(test_loss))
        print('  test accuracy: {:.2f}'.format(test_acc))
        print('')

    return test_loss, test_acc
