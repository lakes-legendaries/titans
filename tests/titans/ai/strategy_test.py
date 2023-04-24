from copy import deepcopy
from os import remove
from os.path import isfile
import pickle
import random

import numpy as np
import tensorflow as tf

from titans.ai import NUM_CHOICES, NUM_FEATURES, StandardStrategy


def test__nanmse_loss():
    y_true = tf.convert_to_tensor([0, np.NaN, 2])
    y_pred = tf.convert_to_tensor([1, 1, 4.])
    assert StandardStrategy._nanmse_loss(y_true, y_pred) == 2.5


def test_standard_strategy():

    # parameters
    num_samples = 100

    # make data
    rng = np.random.default_rng(271828)
    X = rng.random((num_samples, NUM_FEATURES))
    y = np.zeros((num_samples, NUM_CHOICES))
    for sample_num in range(num_samples):
        for choice_num in range(NUM_CHOICES):
            y[sample_num, choice_num] = X[sample_num, choice_num]

    # fit model
    random.seed(271828)
    tf.random.set_seed(271827)
    strategy = StandardStrategy().fit(X, y)

    # test model
    pred = strategy.predict(X)
    assert abs(np.mean(y - pred)) < 0.02

    # test copy
    copy = deepcopy(strategy)
    assert (copy.predict(X) == pred).all()

    # modify model, validate copy unchanged
    strategy.fit(X[0:10, :], y[0:10, :])
    assert not ((strategy.predict(X) == pred).all())
    assert (copy.predict(X) == pred).all()

    # test pickle
    fname = "/tmp/scratch.pickle"
    try:
        pickle.dump(copy, open(fname, "wb"))
        copy2 = pickle.load(open(fname, "rb"))
        assert (copy2.predict(X) == pred).all()
    finally:
        if isfile(fname):
            remove(fname)
