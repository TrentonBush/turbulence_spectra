import pytest
import numpy as np
try:
    from src.models.buffered_cv import BufferedBlockedSplit
except ImportError:
    raise ImportError("Note to future self: the tested module imports from a private sklearn module. Check that BaseShuffleSplit and _validate_shuffle_split are still in sklearn.model_selection._split")


@pytest.fixture
def example_data():
    X = np.arange(16)
    y = X / 10
    yield X, y


@pytest.fixture
def Split_5_blocks_buffer_1():
    instance = BufferedBlockedSplit(n_splits=2, test_size=2, random_state=1, n_blocks=5, buffer_width=1,)
    yield instance


class Test_index_from_blocks:
    """
    example data:      [ 0, 1, 2,   3, 4, 5,   6, 7, 8,   9, 10, 11,   12, 13, 14, 15 ]
    expected blocking: [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14, 15)]
    expected buffer=1 blocks (if each one was independent):
                       [(0, 1,  ), (   4,  ), (   7,  ), (   10,   ), (    13, 14, 15)]
    """

    def test__train_set_middle_block(self, example_data, Split_5_blocks_buffer_1):
        cv = Split_5_blocks_buffer_1
        X, y = example_data
        expected = np.array([6, 7, 8])  # middle block
        actual = cv._index_from_blocks(np.array([2]), block_size=3, n_samples=len(X))
        message = f"index central block, no buffering.\nExpected: {expected}\nActual:   {actual}"
        assert np.array_equal(expected, actual), message

    # boundary logic
    def test__train_set_last_block(self, example_data, Split_5_blocks_buffer_1):
        cv = Split_5_blocks_buffer_1
        X, y = example_data
        expected = np.array([12, 13, 14, 15])  # last block
        actual = cv._index_from_blocks(np.array([4]), block_size=3, n_samples=len(X))
        message = f"index last block, no buffering.\nExpected: {expected}\nActual:   {actual}"
        assert np.array_equal(expected, actual), message

    # integration
    def test__train_set_all_blocks(self, example_data, Split_5_blocks_buffer_1):
        cv = Split_5_blocks_buffer_1
        X, y = example_data
        expected = np.arange(16)  # all blocks
        actual = cv._index_from_blocks(np.arange(5), block_size=3, n_samples=len(X))
        message = f"recreate index from all blocks, no buffering.\nExpected: {expected}\nActual:   {actual}"
        assert np.array_equal(expected, actual), message

    def test__buffered_middle_block(self, example_data, Split_5_blocks_buffer_1):
        cv = Split_5_blocks_buffer_1
        X, y = example_data
        expected = np.array([7])  # middle block (6, 7, 8) loses both edges to buffering
        actual = cv._index_from_blocks(np.array([2]), block_size=3, n_samples=len(X), test_set=True)
        message = f"index central block with buffer=1\nExpected: {expected}\nActual:   {actual}"
        assert np.array_equal(expected, actual), message

    # boundary logic
    def test__buffered_first_block(self, example_data, Split_5_blocks_buffer_1):
        cv = Split_5_blocks_buffer_1
        X, y = example_data
        expected = np.array([0, 1])  # first block (0, 1, 2) loses inner edge to buffering
        actual = cv._index_from_blocks(np.array([0]), block_size=3, n_samples=len(X), test_set=True)
        message = f"index first block with buffer=1\nExpected: {expected}\nActual:   {actual}"
        assert np.array_equal(expected, actual), message

    # boundary logic
    def test__buffered_last_block(self, example_data, Split_5_blocks_buffer_1):
        cv = Split_5_blocks_buffer_1
        X, y = example_data
        expected = np.array([13, 14, 15])  # last block (12, 13, 14, 15) loses inner edge to buffering
        actual = cv._index_from_blocks(np.array([4]), block_size=3, n_samples=len(X), test_set=True)
        message = f"index last block with buffer=1\nExpected: {expected}\nActual:   {actual}"
        assert np.array_equal(expected, actual), message

    # special case (but common)
    def test__buffered_adjacent_blocks(self, example_data, Split_5_blocks_buffer_1):
        cv = Split_5_blocks_buffer_1
        X, y = example_data
        expected = np.array([0, 1, 2, 3, 4])  # (0, 1, 2), (3, 4, 5) loses outer edge only
        actual = cv._index_from_blocks(np.array([0, 1]), block_size=3, n_samples=len(X), test_set=True)
        message = f"index adjacent blocks with buffer=1\nExpected: {expected}\nActual:   {actual}"
        assert np.array_equal(expected, actual), message

    # exception
    def test__buffer_too_large(self, example_data, Split_5_blocks_buffer_1):
        cv = Split_5_blocks_buffer_1
        X, y = example_data
        cv._width = 2
        with pytest.raises(ValueError) as exc:
            cv._index_from_blocks(np.array([0]), block_size=3, n_samples=len(X), test_set=True)
        message = "Buffer_width is larger than half a block - no data will be left!\nbuffer_width: 2\nblock_size: 3"
        assert exc.match(message)


# integration
def test_BufferedBlockedSplit_split(example_data, Split_5_blocks_buffer_1):
    cv = Split_5_blocks_buffer_1
    X, y = example_data
    """The following code is how I determined the expected rng for this test

    >>> from sklearn.utils import check_random_state
    >>> rng = check_random_state(1)  # 1 == cv.random_state
    >>> rng.permutation(5)  # 5 == cv._n_blocks
    array([2, 1, 4, 0, 3])
    >>> rng.permutation(5)
    array([0, 2, 4, 3, 1])
    """
    expected_idx_train_1 = np.array([0, 1, 2, 9, 10, 11, 12, 13, 14, 15])
    expected_idx_test_1 = np.array([4, 5, 6, 7])
    expected_idx_train_2 = np.array([3, 4, 5, 9, 10, 11, 12, 13, 14, 15])
    expected_idx_test_2 = np.array([0, 1, 7])
    expected_dict = {'split 1 of 2' : (expected_idx_train_1, expected_idx_test_1),
                     'split 2 of 2' : (expected_idx_train_2, expected_idx_test_2)}
    gen_actual = cv.split(X, y)

    for split, expected in expected_dict.items():
        actual = next(gen_actual)
        train_message = f"cv.split(), {split}, train set:\nExpected: {expected[0]}\nActual:   {actual[0]}"
        test_message = f"cv.split(), {split}, test set:\nExpected: {expected[1]}\nActual:   {actual[1]}"
        assert np.array_equal(expected[0], actual[0]), train_message
        assert np.array_equal(expected[1], actual[1]), test_message

    with pytest.raises(StopIteration):
        next(gen_actual)
