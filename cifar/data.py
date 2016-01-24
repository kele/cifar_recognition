#!/usr/bin/env python

import numpy as np

DATA_SHAPE = (3, 32, 32)


def load_datastream(train_batch_size=100,
                    training_set_size=40000,
                    validation_set_size=10000):

    from fuel.datasets.cifar10 import CIFAR10
    from fuel.transformers import ScaleAndShift, Cast, Flatten, Mapping
    from fuel.streams import DataStream
    from fuel.schemes import SequentialScheme, ShuffledScheme

    CIFAR10.default_transformers = (
        (ScaleAndShift, [2.0 / 255.0, -1], {'which_sources': 'features'}),
        (Cast, [np.float32], {'which_sources': 'features'}),
    )

    cifar_train = CIFAR10(('train',), subset=slice(None, training_set_size))
    cifar_train_stream = DataStream.default_stream(
        cifar_train,
        iteration_scheme=ShuffledScheme(cifar_train.num_examples, train_batch_size)
    )

    cifar_validation = CIFAR10(('train',), subset=slice(training_set_size,
                                                        training_set_size + validation_set_size))
    cifar_validation_stream = DataStream.default_stream(
        cifar_validation,
        iteration_scheme=SequentialScheme(cifar_validation.num_examples, train_batch_size)
    )

    cifar_test = CIFAR10(('test',))
    cifar_test_stream = DataStream.default_stream(
        cifar_test,
        iteration_scheme=SequentialScheme(cifar_test.num_examples, train_batch_size)
    )

    return {
        'train': cifar_train_stream,
        'validation': cifar_validation_stream,
        'test': cifar_test_stream
    }
