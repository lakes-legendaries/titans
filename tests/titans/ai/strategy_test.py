import numpy as np
import tensorflow as tf

from titans.ai import NUM_CHOICES, Player, Strategy


def test_strategy():

    # parameters
    num_samples = 100
    num_features = Player._get_global_state_size()

    # make data
    rng = np.random.default_rng(271828)
    X = rng.random((num_samples, num_features))
    y = np.zeros((num_samples, NUM_CHOICES))
    for sample_num in range(num_samples):
        for choice_num in range(NUM_CHOICES):
            y[sample_num, choice_num] = X[sample_num, choice_num]

    # fit model
    tf.random.set_seed(271827)
    strategy = Strategy().fit(X, y)

    # test model
    pred = strategy.predict(X)
    assert abs(np.mean(y - pred)) < 0.02
