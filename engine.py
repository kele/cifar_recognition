#!/usr/env/bin python

import time
import theano
import theano.tensor as T
import lasagne


# TODO: make a comment here
def train(input_var, results_var, data, network, hyperparams, num_epochs=100, verbose=False):

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, results_var)
    # TODO: add weight decay here
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=hyperparams['learning_rate'],
        momentum=hyperparams['momentum'])

    train_fn = theano.function([input_var, results_var], loss, updates=updates)


    # No dropout here!
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, results_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), results_var),
                      dtype=theano.config.floatX)

    val_fn = theano.function([input_var, results_var], [test_loss, test_acc])

    for epoch in range(num_epochs):

        start_time = time.time()

        # TODO: reuse the code more
        # Training
        train_loss = 0
        train_batches = 0
        for inputs, results in data['train'].get_epoch_iterator():
            train_loss += train_fn(inputs, results.ravel())
            train_batches += 1
        train_loss = train_loss / train_batches

        # Validation
        val_loss = 0
        val_acc = 0
        val_batches = 0
        for inputs, results in data['validation'].get_epoch_iterator():
            loss, acc = val_fn(inputs, results.ravel())
            val_loss += loss
            val_acc += acc
            val_batches += 1
        val_acc = val_acc / val_batches * 100.0
        val_loss = val_loss / val_batches

        stop_time = time.time()
        delta_time = stop_time - start_time

        if verbose:
            print('Epoch {} of {} took {:.2f}s'.format(epoch + 1, num_epochs, delta_time))
            print('  training loss: {:10.6f}'.format(train_loss))
            print('  validation loss: {:10.6f}'.format(val_loss))
            print('  validation accuracy: {:.2f}'.format(val_acc))

    # Test
    test_loss = 0
    test_acc = 0
    test_batches = 0
    for inputs, results in data['test'].get_epoch_iterator():
        loss, acc = val_fn(inputs, results.ravel())
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
