from pathlib import Path
from scipy.io import loadmat
from tqdm import tqdm
from multiprocessing import Pool
from typing import List, Sequence, Optional
import logging
import typer

import src.data.downloader as downloader


def check_is_corrupt(file: Path) -> bool:
    """Check if a .mat file is corrupt by opening with scipy.io.loadmat()

    Parameters
    ----------
    file : Path
        .mat file to open

    Returns
    -------
    bool
        Answers "is this file corrupt?"
        True if loadmat() throws OSError; False if loadmat() is successful
    """
    try:
        _ = loadmat(file)
        return False
    except OSError:  # a partially-downloaded file
        return True


def corrupt_file_filter_multiprocess(candidate_files: Sequence[Path], n_processes: Optional[int] = None) -> List[Path]:
    """Filter candidate_files to a list of only the corrupt files.

    Parameters
    ----------
    candidate_files : Sequence[Path]
        File paths to check
    n_processes : Optional[int], optional
        Number of processes to use, by default None, which is converted to os.cpu_count()

    Returns
    -------
    List[Path]
        Corrupt files
    """
    logging.info(f"Checking integrity of {len(candidate_files)} files")
    # check_is_corrupt() is strongly CPU-bound.
    # 4-core multiprocessing resulted in 4x speedup and still only 20% SSD utilization
    # TODO: change logging so corrupt files are logged as they are found. More robust to uncaught exceptions.
    with Pool(processes=n_processes) as pool:
        boolean_mask = list(
            tqdm(pool.imap(check_is_corrupt, candidate_files, chunksize=20), total=len(candidate_files))
        )

    corrupt_files = [candidate_files[i] for i, boolean in enumerate(boolean_mask) if boolean]
    str_files = '\n'.join([path.name for path in corrupt_files])
    logging.info(
        f"Found {len(corrupt_files)} corrupt or incomplete files:\n{str_files}"
    )
    return corrupt_files


@downloader.coro
async def main(
    directory: Path, max_concurrent=5, max_per_second=0.75,
):
    """Check if .mat files load correctly. Re-download those that throw OSError.

    Parameters
    ----------
    directory : Path
        directory containing .mat files to check
    max_concurrent : int, optional
        Max simultaneous async downloads, by default 5
    max_per_second : float, optional
        Max rate of GET requests, by default 0.75
    """
    directory = Path(directory)
    files = list(directory.glob("*.mat"))  # need list() for tqdm
    corrupt_files = corrupt_file_filter_multiprocess(files)

    if corrupt_files:
        urls = list(map(downloader.url_from_filename, corrupt_files))
        for path in corrupt_files:
            path.replace(path.with_suffix(".corrupt"))
        await downloader.download_many(
            urls,
            directory,
            max_concurrent=max_concurrent,
            max_per_second=max_per_second,
        )


if __name__ == "__main__":
    logging.basicConfig(
        filename=Path("./data/raw/") / "integrity.log",
        format="%(asctime)s %(message)s",
        level=logging.INFO,
    )
    typer.run(main)
