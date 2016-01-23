#!/usr/bin/env python

import numpy as np

DATA_SHAPE = (1, 28, 28)

def load_datastream(train_batch_size=100):
    from fuel.datasets.mnist import MNIST
    from fuel.transformers import ScaleAndShift, Cast, Flatten, Mapping
    from fuel.streams import DataStream
    from fuel.schemes import SequentialScheme, ShuffledScheme

    MNIST.default_transformers = (
        (ScaleAndShift, [2.0 / 255.0, -1], {'which_sources': 'features'}),
        (Cast, [np.float32], {'which_sources': 'features'}),
    )

    mnist_train = MNIST(('train',), subset=slice(None, 50000))
    mnist_train_stream = DataStream.default_stream(
        mnist_train,
        iteration_scheme=ShuffledScheme(mnist_train.num_examples, train_batch_size)
    )

    mnist_validation = MNIST(('train',), subset=slice(50000, None))
    mnist_validation_stream = DataStream.default_stream(
        mnist_validation,
        iteration_scheme=SequentialScheme(mnist_validation.num_examples, 250)
    )

    mnist_test = MNIST(('test',))
    mnist_test_stream = DataStream.default_stream(
        mnist_test,
        iteration_scheme=SequentialScheme(mnist_test.num_examples, 250)
    )

    return {
        'train': mnist_train_stream,
        'validation': mnist_validation_stream,
        'test': mnist_test_stream
    }
