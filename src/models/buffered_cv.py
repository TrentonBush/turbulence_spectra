from sklearn.model_selection import BaseShuffleSplit
from sklearn.utils.validation import _num_samples, _deprecate_positional_args
import numpy as np


# note to anyone reading: sklearn doesn't currently use type hints, so I didn't see much benefit to adding them on this thin wrapper.
class BufferedBlockedSplit(BaseShuffleSplit):
    """Sklearn splitter to mitigate autocorrelation leaking through the boundaries between train and test sets. This is done by buffering (excluding) data within a radius of the boundaries. Buffer is entirely in the test set. For data with one sequential dimension only, eg time series."""
    
    @_deprecate_positional_args
    def __init__(self, n_splits=10, *, buffer_radius, n_blocks, test_size=None, train_size=None, random_state=None):
        if not isinstance(buffer_radius, int) or buffer_radius < 0:
            raise ValueError(f"'buffer_radius' must be a non-negative integer; it is used for indexing. Given {buffer_radius}")

        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state)
        self._default_test_size = 0.2
        self._radius = buffer_radius
        self._n_blocks = n_blocks

    def _iter_indices(self, X, y=None, groups=None):
        """y and groups are ignored; for compatibilty only"""
        # define blocks
        n_samples = len(X)
        block_size = np.round(n_samples / self._n_blocks)
        # define n_test_blocks from test_size
        for i in range(self.n_splits):
            # randomly select blocks
            # make index arrays
            yield train_indcices, test_indicies

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        return super().split(X, y, groups)