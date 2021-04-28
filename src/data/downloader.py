"""CLI to download 20 Hz data from NREL NWTC mast 5"""
import asyncio
import httpx
import aiometer
import aiofiles
from pathlib import Path
from functools import partial, wraps
import pandas as pd
from typing import List, Tuple, Optional, Sequence
import logging
import typer
from httpcore import ReadTimeout

# NREL NWTC mast M5, 20hz data, matlab files (only file format available)
BASE_URL = "https://wind.nrel.gov/MetData/135mData/M5Twr/20Hz/mat/"


def _dense_samples(
    *, start_timestamp: str, end_timestamp: str, files_to_skip: Optional[set] = None,
) -> List[Tuple[str, str]]:
    """Create URL parts (date string and filename) between start_timestamp and end_timestamp, inclusive.
    URL format is BASE_URL/date_string/filename.mat

    Example:
    >>> _dense_samples(start_timestamp='2019-04-17 03:50', end_timestamp='2019-04-17 04:10')
    [
        ("2019/04/17/", "04_17_2019_03_50_00_000.mat"),
        ("2019/04/17/", "04_17_2019_04_00_00_000.mat"),
        ("2019/04/17/", "04_17_2019_04_10_00_000.mat")
    ]

    Parameters
    ----------
    start_timestamp : str
        first timestamp of interval. Inclusive.
    end_timestamp : str
        last timestamp of interval. Inclusive.
    files_to_skip : Optional[set], optional
        filter out a set of filenames, by default None

    Returns
    -------
    List[Tuple[str, str]]
        URL parts in the format consumed by download_many()
    """
    times = pd.date_range(start_timestamp, end_timestamp, freq="10min")
    samples = pd.DataFrame(
        {
            "urls": times.strftime("%Y/%m/%d/"),
            "filenames": times.strftime("%m_%d_%Y_%H_%M_%S_000.mat"),
        }
    )
    if files_to_skip is not None:
        filter_ = ~samples["filenames"].isin(files_to_skip)
        logging.info(
            f"downloader._dense_samples skipped {filter_.size - filter_.sum()} preexisting files"
        )
        samples = samples[filter_]  # exclude preexisting
    return list(samples[["urls", "filenames"]].itertuples(index=False, name=None))


def _url_from_filename(file: Path) -> Tuple[str, str]:
    """Generate url parts from input NREL-format.mat filename

    Example:
    >>> _url_from_filename(Path("./04_17_2019_03_50_00_000.mat"))
    ("2019/04/17/", "04_17_2019_03_50_00_000.mat")

    Parameters
    ----------
    file : Path
        path to .mat file

    Returns
    -------
    Tuple[str, str]
        (date_url, filename.mat)
        Format consumed by the url_parts kwarg of download_file()
    """
    ts = pd.to_datetime(file.name.split("_000.mat")[0], format="%m_%d_%Y_%H_%M_%S")
    date_url = ts.strftime("%Y/%m/%d/")
    return (date_url, file.name)


async def download_file(
    client: httpx.Client, out_dir: Path, url_parts: Tuple[str, str]
):
    """download a single file from NREL, identified by url_parts

    Parameters
    ----------
    client : httpx.Client
        async web client
    out_dir : Path
        destination directory
    url_parts : Tuple[str, str]
        tuple of (date_url, filename.mat), like ("2019/04/17/", "04_17_2019_03_50_00_000.mat")
        Use _dense_samples() to produce a series of them.
    """
    url = "".join([BASE_URL, *url_parts])
    filepath = out_dir / url_parts[1]
    try:
        async with client.stream("GET", url) as resp:
            try:
                resp.raise_for_status()
                async with aiofiles.open(filepath, "wb") as f:
                    async for data in resp.aiter_bytes():
                        if data:
                            await f.write(data)
                print(
                    f"Downloaded {url_parts[1]} at {pd.Timestamp('now').strftime('%H:%M:%S')}"
                )
            except httpx.HTTPError:
                logging.info(f"HTTPError for {url_parts[1]}")
            except ReadTimeout:
                logging.info(f"ReadTimeout for {url_parts[1]}")
    except (httpx.ConnectTimeout, httpx._exceptions.ConnectTimeout):
        logging.warning(f"Timeout for {url_parts[1]} Needs re-download.")


async def download_many(
    urls: Sequence[Tuple[str, str]],
    output_directory: Path,
    max_concurrent=5,
    max_per_second=0.75,
):
    """Download a list of files asynchronously

    Parameters
    ----------
    urls : Sequence[Tuple[str, str]]
        url parts as created by _dense_samples() or similar
    output_directory : Path
        destination directory
    max_concurrent : int, optional
        maximum simultaneous connections, by default 5
    max_per_second : float, optional
        maximum connections per second, by default 0.75
    """
    directory = Path(output_directory)
    count = len(urls)
    message = f"Starting Download of {count} files"
    logging.info(message)
    start = pd.Timestamp("now")

    async with httpx.AsyncClient() as client:
        await aiometer.run_on_each(
            partial(download_file, client, directory),
            urls,
            max_at_once=int(max_concurrent),
            max_per_second=float(max_per_second),
        )

    end = pd.Timestamp("now")
    time = end - start
    message = "\n".join(
        [
            f"Elapsed time: {time.round('s')}",
            f"URLs: {count}",
            f"Seconds per URL: {(time / count).total_seconds():.2f}",
        ]
    )
    logging.info(message)


def coro(func):
    """call async function. Necessary for typer CLI integration."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


@coro
async def main(
    start_timestamp: str,
    end_timestamp: str,
    output_directory: Path,
    max_concurrent=5,  # include for typer CLI, instead of **kwargs
    max_per_second=0.75,  # include for typer CLI, instead of **kwargs
):
    """download any missing .mat files in a time interval

    Parameters
    ----------
    start_timestamp : str
        first timestamp of interval, inclusive
    end_timestamp : str
        last timestamp of interval, inclusive
    output_directory : Path
        destination directory. Any pre-existing .mat files will be skipped
    max_concurrent : int, optional
        maximum simultaneous connections, by default 5
    max_per_second : float, optional
        maximum connections per second, by default 0.75
    """
    directory = Path(output_directory)
    files_to_skip = set([path.name for path in directory.glob("*.mat")])
    if not files_to_skip:
        files_to_skip = None  # type: ignore

    urls = _dense_samples(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        files_to_skip=files_to_skip,
    )

    await download_many(
        urls, directory, max_concurrent=max_concurrent, max_per_second=max_per_second
    )


if __name__ == "__main__":
    logging.basicConfig(
        filename=Path("./data/raw/") / "download.log",
        format="%(asctime)s %(message)s",
        level=logging.INFO,
    )
    typer.run(main)
