import httpx
import aiometer
import aiofiles
import asyncio
from pathlib import Path
from functools import partial, wraps
import pandas as pd
from typing import List, Tuple, Optional
import logging
import typer

# NREL NWTC mast M5, 20hz data, matlab files (only file format available)
BASE_URL = "https://wind.nrel.gov/MetData/135mData/M5Twr/20Hz/mat/"


def dense_samples(
    *, start_timestamp: str, end_timestamp: str, files_to_skip: Optional[set] = None,
) -> List[Tuple[str, str]]:
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
            f"downloader.dense_samples skipped {filter_.size - filter_.sum()} preexisting files"
        )
        samples = samples[filter_]  # exclude preexisting
    return list(samples[["urls", "filenames"]].itertuples(index=False, name=None))


async def download_file(
    client: httpx.Client, out_dir: Path, url_parts: Tuple[str, str]
):
    url = "".join([BASE_URL, *url_parts])
    filepath = out_dir / url_parts[1]
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
        except httpx.ConnectTimeout:
            logging.warning(f"Timeout for {url_parts[1]} Needs re-download.")


def coro(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


@coro
async def main(
    start_timestamp: str,
    end_timestamp: str,
    output_directory: str,
    max_concurrent=8,
    max_per_second=1,
):
    directory = Path(output_directory)
    files_to_skip = set([path.name for path in directory.glob("*.mat")])
    if not files_to_skip:
        files_to_skip = None  # type: ignore

    urls = dense_samples(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        files_to_skip=files_to_skip,
    )

    print("Starting Download")
    logging.info("Starting download")
    begin = pd.Timestamp("now")
    async with httpx.AsyncClient() as client:
        await aiometer.run_on_each(
            partial(download_file, client, directory),
            urls,
            max_at_once=max_concurrent,
            max_per_second=max_per_second,
        )
    end = pd.Timestamp("now")
    time = end - begin
    count = len(urls)
    message = "\n".join(
        [
            f"Elapsed time: {time.round('s')}",
            f"URLs: {count}",
            f"Seconds per URL: {(time / count).total_seconds():.2f}",
        ]
    )

    logging.info(message)
    print(message)


if __name__ == "__main__":
    logging.basicConfig(
        filename=Path("./data/raw/") / "download.log",
        format="%(asctime)s %(message)s",
        level=logging.INFO,
    )
    typer.run(main)
