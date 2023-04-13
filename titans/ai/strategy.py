"""Strategy module"""

from __future__ import annotations

import keras
from keras import callbacks, layers, optimizers
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from titans.ai.constants import NUM_CHOICES


class Strategy:
    """Strategy for making decisions

    This class conforms to the sklearn API. You can substitute for it any class
    that provides a fit and predict method.

    Parameters
    ----------
    scale: bool, optional, default=True
        use `sklearn.preprocessing.StandardScaler` to scale data
    """
    def __init__(
        self,
        *,
        scale: bool = True,
    ):
        # import here to avoid circular import
        from titans.ai.player import Player

        # initialize scaler
        self._scaler = StandardScaler() if scale else None
        self._scaler_fitted = False

        # initialize model
        input_layer = layers.Input(shape=(Player._get_global_state_size()))
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
            # verbose=False,
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

        # make 2D
        is_one_dimensional = len(X.shape) == 1
        if is_one_dimensional:
            X = np.array([X])

        # use random if untrained
        if not self._model_fitted:
            pred = np.array([
                np.random.sample(NUM_CHOICES)
                for _ in range(X.shape[0])
            ])

        # make predictions
        else:

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
