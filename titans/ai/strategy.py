"""Strategy module"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

import keras
from keras import callbacks, layers
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

    This class scales data (via z-score normalization) and then using an MLP
    (feedforward regression ANN) to predict best courses of action (given the
    player's / game's state).

    This model is designed so that you can resume fitting at any point: If you
    call the fit function multiple times, the ANN will simply resume fitting
    from where you left off. (The scaler will be static, though, configured
    through your first time you called fit. This ensures that new data is on
    the same scale as old data.)

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
            optimizer=tf.keras.optimizers.Adam(),
        )
        self._model = model
        self._model_fitted = False

    def __deepcopy__(self, memo: dict) -> Strategy:

        # initialize copy
        copy = StandardStrategy()

        # copy inheritted
        copy._rng = deepcopy(self._rng)

        # copy standard attributes
        copy._scaler = deepcopy(self._scaler)
        copy._scaler_fitted = deepcopy(self._scaler_fitted)
        copy._model_fitted = deepcopy(self._model_fitted)

        # copy neural network
        copy._model = keras.models.clone_model(self._model)
        copy._model.set_weights(self._model.get_weights())
        copy._model.compile(
            loss=self._nanmse_loss,
            optimizer=tf.keras.optimizers.Adam(),
        )

        # return copy
        return copy

    @staticmethod
    @keras.utils.register_keras_serializable("titans.ai.strategy")
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
