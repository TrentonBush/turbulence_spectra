from sklearn.model_selection import BaseShuffleSplit
from sklearn.utils.validation import (
    _num_samples,
    _deprecate_positional_args,
    _validate_shuffle_split,
    check_random_state,
)
import numpy as np


class BufferedBlockedSplit(BaseShuffleSplit):
    """Similar to GroupShuffleSplit, but mitigates data leakage from autocorrelation.

    Provides randomized train/test indices to split data into
    permutations of sequential blocks.

    Autocorrelation causes information to leak into neighboring points, which erodes
    the boundaries between train and test sets. One way to mitigate this is
    by 'buffering' (excluding) data within a radius of the boundary values.
    But buffering costs data. To reduce data loss, the number of boundary points
    is reduced by 'blocking' test points together.

    Notes
    ----------
    Buffered data is taken from the test set, so set test_size and n_blocks accordingly.
    This is designed for data with one sequential dimension only, eg time series.
    Assumes data is already sorted in the sequential dimension.

    Parameters
    ----------
    n_splits : int, default=5
        Number of re-shuffling & splitting iterations.
    buffer_width : int
        number of points to buffer (exclude) at each train/test interface.
        Applied only to the test side. Total points lost is between 2 * buffer_width
        and n_blocks * 2 * buffer_width, depending on whether the randomly selected
        test blocks happen to be adjacent (no train/test interface).
    n_blocks : int
        number of blocks to partition the data into. Indices are assigned
        assuming the data are already sorted in the sequential dimension.
    test_size : float, int, default=0.2
        If float, should be between 0.0 and 1.0 and represent the proportion
        of groups to include in the test split (rounded up). If int,
        represents the absolute number of test groups. If None, the value is
        set to the complement of the train size.
    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the groups to include in the train split. If
        int, represents the absolute number of train groups. If None,
        the value is automatically set to the complement of the test size.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.
    """

    @_deprecate_positional_args
    def __init__(
        self,
        n_splits=10,
        *,
        test_size=None,
        train_size=None,
        random_state=None,
        buffer_width,
        n_blocks,
    ):
        if not isinstance(buffer_width, int) or buffer_width < 0:
            raise ValueError(
                f"'buffer_width' must be a non-negative integer; it is used for indexing. Given {buffer_width}"
            )

        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )

        self._default_test_size = 0.2
        self._width = buffer_width
        self._n_blocks = n_blocks

    def _iter_indices(self, X, y=None, groups=None):
        """y and groups are ignored; included for compatibility with base class"""

        n_samples = _num_samples(X)
        block_size = n_samples // self._n_blocks  # extra points go in last block
        n_train, n_test = _validate_shuffle_split(
            self._n_blocks,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        rng = check_random_state(self.random_state)

        for i in range(self.n_splits):
            # choose random blocks
            permutation = rng.permutation(self._n_blocks)
            test_blocks = permutation[:n_test]
            train_blocks = permutation[n_test : (n_test + n_train)]

            test_indicies = self._index_from_blocks(
                test_blocks, block_size, n_samples, test_set=True
            )
            train_indices = self._index_from_blocks(train_blocks, block_size, n_samples)
            yield train_indices, test_indicies

    def _index_from_blocks(self, blocks, block_size, n_samples, test_set=False):
        """convert block numbers to their corresponding indices.

        Example: 102 data points are blocked into 10 blocks. Blocks 2 and 9 are chosen for the test set. Buffer width = 0.
        The indices returned by this method are: np.array([20, 21, ... , 29, 90, 91, ... , 99, 100, 101]).
        Note that the last block contains extra points equal to n_samples modulo n_blocks.
        A buffer of width=1 would remove points 20, 29, and 90."""
        buffer_width = self._width if test_set else 0

        blocks = np.sort(blocks)
        boundary_indices = []

        if blocks[0] == 0:  # start of dataset needs no buffering
            boundary_indices.append(0)
        else:
            boundary_indices.append(blocks[0] * block_size + buffer_width)

        # non start/end boundaries
        for i, block_num in enumerate(blocks[1:]):
            previous = blocks[i - 1]
            if block_num - previous == 1:  # adjacent, no boundary
                continue
            else:
                # end of the last block
                boundary_indices.append((previous + 1) * block_size - buffer_width)
                # start of the current block
                boundary_indices.append(block_num * block_size + buffer_width)

        if blocks[-1] == (self._n_blocks - 1):  # end of dataset needs no buffering
            boundary_indices.append(n_samples - 1)
        else:
            boundary_indices.append((blocks[-1] + 1) * block_size - buffer_width)

        merged_block_indices = [
            np.arange(start, end)
            for start, end in zip(boundary_indices[::2], boundary_indices[1::2])
        ]
        return np.concatenate(merged_block_indices)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Not used in this splitter. Included for compatibility with the base class.
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
