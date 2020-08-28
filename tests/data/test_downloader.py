from pathlib import Path
from src.data.downloader import dense_samples


def test_dense_samples_with_preexisting_file():
    files_to_skip = set([Path("./01_01_2020_00_10_00_000.mat").name])
    expected = [
        ("2020/01/01/", "01_01_2020_00_00_00_000.mat"),
        ("2020/01/01/", "01_01_2020_00_20_00_000.mat"),
        ("2020/01/01/", "01_01_2020_00_30_00_000.mat"),
    ]
    actual = dense_samples(
        start_timestamp="2020-1-1 00:00",
        end_timestamp="2020-1-1 00:30",
        files_to_skip=files_to_skip,
    )
    assert (
        actual == expected
    ), f"dense_samples() with a preexisting file (at 10 minutes) to skip.\nExpected: {expected}\nActual:{actual}"
