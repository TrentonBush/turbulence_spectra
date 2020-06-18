import httpx
import asyncio
import aiometer
import aiofiles
from pathlib import Path
from functools import partial
import pandas as pd
import numpy as np
from typing import List, Tuple

# NREL NWTC mast M5, 20hz data, matlab files (only file format available)
BASE_URL = "https://wind.nrel.gov/MetData/135mData/M5Twr/20Hz/mat/"


def time_sampler(
    start_date: pd.Timestamp,
    samples_per_day: int,
    samples_per_year: int,
    num_years: int = 1,
) -> List[Tuple[str]]:

    # Sample output:
    # [('2019/01/01/', '01_01_2019_00_00_00_000.mat'),
    # ('2019/01/01/', '01_01_2019_06_00_00_000.mat')]
    days = start_date + pd.Timedelta("1D") * pd.Series(
        np.round(np.arange(samples_per_year * num_years) * 365.25 / samples_per_year)
    )
    times = pd.Timedelta("10min") * pd.Series(
        np.round(np.arange(samples_per_day) * 144 / samples_per_day)
    )
    samples = pd.DataFrame(pd.concat([times + day for day in days], ignore_index=True))
    samples["url_date"] = samples[0].dt.strftime("%Y/%m/%d/")
    samples["filename"] = samples[0].dt.strftime("%m_%d_%Y_%H_%M_%S_000.mat")
    return list(samples[["url_date", "filename"]].itertuples(index=False, name=None))


def filename_from_url(url: str) -> str:
    return url.split("/")[-1]


async def download(client: httpx.Client, out_dir: Path, url_parts: tuple):
    url = BASE_URL + url_parts[0] + url_parts[1]
    filepath = out_dir / url_parts[1]
    async with client.stream("GET", url) as resp:
        try:
            resp.raise_for_status()
            async with aiofiles.open(filepath, "wb") as f:
                async for data in resp.aiter_bytes():
                    if data:
                        await f.write(data)
            print(
                f"Done with {url_parts[1]} at {pd.Timestamp('now').strftime('%H:%M:%S')}"
            )
        except httpx.HTTPError:
            print(f"HTTPError for {url_parts[1]}")


async def main(urls: list, out_dir: Path, max_concurrent=8, max_per_second=1):
    async with httpx.AsyncClient() as client:
        await aiometer.run_on_each(
            partial(download, client, out_dir),
            urls,
            max_at_once=max_concurrent,
            max_per_second=max_per_second,
        )


urls = time_sampler(pd.Timestamp("2019-1-1"), 24, 24, 1)
asyncio.run(main(urls, Path("./data/raw")))
# started 15:33:44
# HTTPError for 06_17_2019
# HTTPError for 07_18_2019
# finished 15:51:28
# replaced those two dates with the one after, to get 100% day coverage
