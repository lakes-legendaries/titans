"""Strategy module"""

from __future__ import annotations

from abc import ABC, abstractmethod

import keras
from keras import callbacks, layers, optimizers
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from titans.ai.constants import NUM_CHOICES, NUM_FEATURES


class Strategy(ABC):
    """Strategy abstract base class"""

    def fit(self, X: np.ndarray, y: np.ndarray, /) -> Strategy:
        """Fit model

        Parameters
        ----------
        X: np.ndarray
            data to use for fitting
        y: np.ndarray
            labels

        Returns
        -------
        Strategy
            calling instance
        """

    @abstractmethod
    def predict(self, X: np.ndarray, /) -> np.ndarray:
        """Predict best course of action

        Parameters
        ----------
        X: np.ndarray
            data to predict for

        Returns
        -------
        np.ndarray
            predictions, returned as the predicted probability of winning given
            each possible choice
        """


class RandomStrategy(Strategy):
    """Strategy for making random decisions

    Parameters
    ----------
    random_state: int, optional, default=None
        random seed
    """
    def __init__(self, random_state: int = None):
        self._rng: np.random.Generator = np.random.default_rng(random_state)

    def predict(self, X: np.ndarray, /) -> np.ndarray:
        return np.array([
            self._rng.random(NUM_CHOICES)
            for _ in range(X.shape[0])
        ]) if len(X.shape) > 1 else self._rng.random(NUM_CHOICES)


class StandardStrategy(RandomStrategy):
    """Standard strategy for making decisions

    This class conforms to the sklearn API. You can substitute for it any class
    that provides a fit and predict method.

    Parameters
    ----------
    scale: bool, optional, default=True
        use `sklearn.preprocessing.StandardScaler` to scale data
    **kwargs: Any
        passed to `RandomStrategy` on `__init__()`
    """
    def __init__(
        self,
        *,
        scale: bool = True,
        **kwargs,
    ):
        # initialize parent
        RandomStrategy.__init__(self, **kwargs)

        # initialize scaler
        self._scaler = StandardScaler() if scale else None
        self._scaler_fitted = False

        # initialize model
        input_layer = layers.Input(shape=(NUM_FEATURES))
        x = layers.Dense(100, activation="relu")(input_layer)
        output_layer = layers.Dense(NUM_CHOICES)(x)
        model = keras.Model(input_layer, output_layer)
        model.compile(
            loss=self._nanmse_loss,
            optimizer=optimizers.Adam(),
        )
        self._model = model
        self._model_fitted = False

    @staticmethod
    def _nanmse_loss(y_true, y_pred):
        """MSE that ignores NaN entries"""
        mask = ~tf.math.is_nan(y_true)
        return tf.reduce_mean(tf.square(y_true[mask] - y_pred[mask]))

    def fit(self, X: np.ndarray, y: np.ndarray, /) -> Strategy:
        """Fit model

        Parameters
        ----------
        X: np.ndarray
            data to use for fitting
        y: np.ndarray
            labels

        Returns
        -------
        Strategy
            calling instance
        """

        # scale data
        if self._scaler is not None:
            if not self._scaler_fitted:
                self._scaler.fit(X)
                self._scaler_fitted = True
            X = self._scaler.transform(X)

        # train model
        self._model.fit(
            X,
            y,
            callbacks=[callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True,
            )],
            epochs=100,
            validation_split=0.25,
            verbose=False,
        )
        self._model_fitted = True

        # return
        return self

    def predict(self, X: np.ndarray, /) -> np.ndarray:
        """Predict best course of action

        Parameters
        ----------
        X: np.ndarray
            data to predict for

        Returns
        -------
        np.ndarray
            predictions, returned as the predicted probability of winning given
            each possible choice
        """

        # use random if untrained
        if not self._model_fitted:
            return RandomStrategy.predict(self, X)

        # make 2D
        is_one_dimensional = len(X.shape) == 1
        if is_one_dimensional:
            X = np.array([X])

        # scale data
        if self._scaler is not None:
            X = self._scaler.transform(X)

        # make predictions
        pred = self._model(X).numpy()

        # return (matching input shape)
        return (
            pred
            if not is_one_dimensional
            else pred[0]
        )
