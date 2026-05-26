import asyncio
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import time

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RateLimiter:
    """Thread-safe rate limiter that caps the global call rate to ``max_per_second``.

    Each ``acquire()`` call reserves the next slot and sleeps if the caller
    is ahead of schedule, so N worker threads sharing one limiter will not
    exceed the target rate in aggregate.
    """

    def __init__(self, max_per_second: float):
        self.min_interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._next_allowed = time.monotonic()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            self._next_allowed = max(now, self._next_allowed) + self.min_interval
        if wait > 0:
            time.sleep(wait)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))


# Configure session with automatic retries
def create_resilient_session(token: str = None, max_retries: int = 3, pool_size: int = 128) -> Session:
    """Create a requests session with built-in retry and timeout handling.

    ``pool_size`` controls the urllib3 connection pool; it must be at least
    as large as the number of concurrent download workers to avoid
    "Connection pool is full" warnings and serialized requests.
    """
    session = Session()

    # Retry strategy: exponential backoff on connection errors and timeouts
    retry_strategy = Retry(
        total=max_retries,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        backoff_factor=1,  # 1s, 2s, 4s... exponential backoff
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    if token:
        session.headers.update({"Authorization": token})

    # Set default timeout (connect, read)
    session.timeout = (10, 30)  # 10s connect, 30s read

    return session


def download_dataset_file(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    filename: str,
    directory: str,
    overwrite: bool,
    max_retries: int = 3,
) -> tuple[bool, str]:
    # if a file from this dataset already exists, skip downloading it.
    file_path = Path(directory, filename).resolve()
    if not overwrite and file_path.exists():
        logger.debug(f"Dataset file '{filename}' was already downloaded.")
        return True, filename

    endpoint = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files/{filename}/url"

    # Retry getting the file URL
    for attempt in range(max_retries):
        try:
            get_file_response = session.get(endpoint, timeout=(10, 30))
            break
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(f"Error getting download URL for {filename}, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to get download URL for {filename} after {max_retries} attempts")
                return False, filename

    # retrieve download URL for dataset file
    if get_file_response.status_code != 200:
        logger.warning(f"Unable to get file: {filename}")
        logger.warning(get_file_response.content)
        return False, filename

    # use download URL to GET dataset file. We don't need to set the 'Authorization' header,
    # The presigned download URL already has permissions to GET the file contents
    download_url = get_file_response.json().get("temporaryDownloadUrl")
    return download_file_from_temporary_download_url(download_url, directory, filename, max_retries)


def download_file_from_temporary_download_url(download_url, directory, filename, max_retries=3):
    """Download file with automatic retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            # Create a session with proper timeout for this download
            session = requests.Session()
            with session.get(download_url, stream=True, timeout=(10, 60)) as r:
                r.raise_for_status()
                with open(f"{directory}/{filename}", "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            logger.debug(f"Downloaded dataset file '{filename}'")
            return True, filename
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # 1s, 2s, 4s exponential backoff
                logger.warning(
                    f"Connection error downloading {filename}, retrying in {wait_time}s (attempt {attempt+1}/{max_retries}): {e}"
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download {filename} after {max_retries} attempts: {e}")
                return False, filename
        except Exception as e:
            logger.exception(f"Unexpected error downloading {filename}: {e}")
            return False, filename


def list_dataset_files(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    params: dict[str, str],
) -> tuple[list[str], dict[str, Any]]:
    logger.info(f"Retrieve dataset files with query params: {params}")

    list_files_endpoint = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files"
    list_files_response = session.get(list_files_endpoint, params=params)

    if list_files_response.status_code != 200:
        raise Exception(
            f"Unable to list dataset files: {list_files_response.status_code} {list_files_response.text}"
        )

    try:
        list_files_response_json = list_files_response.json()
        dataset_files = list_files_response_json.get("files")
        dataset_filenames = list(map(lambda x: x.get("filename"), dataset_files))
        return dataset_filenames, list_files_response_json
    except Exception as e:
        logger.exception(e)
        raise Exception(e)


def parse_file_utc_timestamp(filename: str):
    # Date-timestamp format in KNMI filenames: [_VERSION_]KMDS__OPER_P___10M_OBS_L2_YYYYMMDDHHMM.nc
    # Examples:
    #   KMDS__OPER_P___10M_OBS_L2_202604011320.nc
    #   _1.0_KMDS__OPER_P___10M_OBS_L2_201208310210.nc
    import re
    from datetime import datetime, timezone

    # Try 10-min format: YYYYMMDDHHMM.nc
    m = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})\.nc$", filename)
    if m is not None:
        year, month, day, hour, minute = m.groups()
        try:
            return datetime(int(year), int(month), int(day), int(hour), int(minute), tzinfo=timezone.utc)
        except ValueError:
            return None

    # Try hourly format: YYYYMMDD-HH.nc
    m = re.search(r"(\d{4})(\d{2})(\d{2})-(\d{2})\.nc$", filename)
    if m is not None:
        year, month, day, hour = m.groups()
        try:
            return datetime(int(year), int(month), int(day), int(hour), 0, tzinfo=timezone.utc)
        except ValueError:
            return None

    return None


def compute_local_filepath(dataset_name: str, filename: str, base_target: str = "D:/thesis"):
    dt = parse_file_utc_timestamp(filename)
    if dt is None:
        raise ValueError(f"Unexpected filename format (no UTC timestamp): {filename}")
    dir_path = Path(f"{base_target}/{dataset_name}/{dt.year}/{dt.month:02d}")
    return dir_path, dir_path / filename


def _list_filenames_in_range(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    start_datetime,
    end_datetime,
    max_keys: int = 1000,
) -> list[str]:
    """List all filenames whose parsed timestamp falls in [start, end].

    Uses the KNMI ``begin`` query param for server-side filtering and stops
    paginating as soon as a file past ``end_datetime`` is seen (files are
    returned in ascending ``created`` order).
    """
    next_page_token = None
    begin_timestamp = start_datetime.isoformat()
    collected: list[str] = []
    pages = 0

    while True:
        params = {
            "maxKeys": str(max_keys),
            "orderBy": "created",
            "begin": begin_timestamp,
        }
        if next_page_token:
            params["nextPageToken"] = next_page_token

        filenames, resp_json = list_dataset_files(session, base_url, dataset_name, dataset_version, params)
        pages += 1

        reached_end = False
        for filename in filenames:
            dt = parse_file_utc_timestamp(filename)
            if dt is None:
                continue
            if dt > end_datetime:
                reached_end = True
                break
            if dt >= start_datetime:
                collected.append(filename)

        if pages % 10 == 0:
            logger.info(f"Listing progress: {pages} pages, {len(collected)} files collected so far")

        if reached_end:
            break
        next_page_token = resp_json.get("nextPageToken")
        if not next_page_token:
            break

    return collected


def download_files_date_range(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    start_datetime,
    end_datetime,
    base_target: str = "D:/thesis",
    max_keys: int = 1000,
    overwrite: bool = False,
    resume: bool = True,
    max_workers: int = 50,
    rate_limit_per_sec: float = 90.0,
):
    """List files for ``[start_datetime, end_datetime]`` then download in parallel.

    Parallelism is capped by a shared ``RateLimiter`` so the aggregate KNMI
    API call rate stays below ``rate_limit_per_sec`` (default 90 req/s,
    leaving headroom below the 100 req/s published limit). Increase
    ``max_workers`` if downloads (the S3 fetch, not rate-limited) become
    the bottleneck.
    """
    if start_datetime.tzinfo is None or end_datetime.tzinfo is None:
        raise ValueError("start_datetime and end_datetime must be timezone-aware (UTC)")

    progress_key = f"{dataset_name}_{start_datetime.date()}_{end_datetime.date()}"

    # Phase 1: enumerate all filenames in the requested range.
    logger.info(f"Listing files for {dataset_name} between {start_datetime} and {end_datetime}...")
    filenames_in_range = _list_filenames_in_range(
        session, base_url, dataset_name, dataset_version,
        start_datetime, end_datetime, max_keys=max_keys,
    )
    logger.info(f"Found {len(filenames_in_range)} files in range")

    # Phase 2: figure out what still needs downloading.
    already_downloaded = _scan_downloaded_files(base_target, dataset_name) if resume else set()
    progress = _load_progress(base_target, progress_key) if resume else {"downloaded": [], "failed": []}
    failed_set = set(progress.get("failed", []))

    if overwrite:
        to_download = list(filenames_in_range)
    else:
        to_download = [f for f in filenames_in_range if f not in already_downloaded]
    logger.info(
        f"{len(to_download)} files to fetch "
        f"({len(already_downloaded)} already on disk, {len(failed_set)} previously failed)"
    )

    downloaded_set = set(already_downloaded)
    if not to_download:
        _save_progress(base_target, progress_key, {"downloaded": list(downloaded_set), "failed": list(failed_set)})
        return list(downloaded_set)

    # Pre-create the year/month directories so worker threads don't race on mkdir.
    dirs_needed = set()
    for filename in to_download:
        dirs_needed.add(compute_local_filepath(dataset_name, filename, base_target)[0])
    for d in dirs_needed:
        d.mkdir(parents=True, exist_ok=True)

    # Phase 3: parallel download under a global rate limit.
    rate_limiter = RateLimiter(rate_limit_per_sec)
    state_lock = threading.Lock()
    total = len(to_download)
    processed = 0
    new_successes = 0
    new_failures = 0
    checkpoint_every = 1000
    start_wall = time.monotonic()

    def _worker(filename: str):
        local_dir, _ = compute_local_filepath(dataset_name, filename, base_target)
        rate_limiter.acquire()
        return download_dataset_file(
            session, base_url, dataset_name, dataset_version,
            filename, str(local_dir), overwrite,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, f): f for f in to_download}
        for future in as_completed(futures):
            filename = futures[future]
            try:
                success, _ = future.result()
            except Exception as e:
                logger.exception(f"Worker crashed for {filename}: {e}")
                success = False

            with state_lock:
                if success:
                    downloaded_set.add(filename)
                    failed_set.discard(filename)
                    new_successes += 1
                else:
                    failed_set.add(filename)
                    new_failures += 1
                processed += 1

                if processed % checkpoint_every == 0 or processed == total:
                    elapsed = max(time.monotonic() - start_wall, 1e-6)
                    rate = processed / elapsed
                    eta_s = (total - processed) / rate if rate > 0 else float("inf")
                    logger.info(
                        f"Progress: {processed}/{total} "
                        f"(ok={new_successes}, fail={new_failures}) "
                        f"rate={rate:.1f} files/s, eta={eta_s/60:.1f} min"
                    )
                    _save_progress(
                        base_target, progress_key,
                        {"downloaded": list(downloaded_set), "failed": list(failed_set)},
                    )

    _save_progress(
        base_target, progress_key,
        {"downloaded": list(downloaded_set), "failed": list(failed_set)},
    )
    logger.info(
        f"Done. {new_successes} newly downloaded, {new_failures} failed, "
        f"{len(downloaded_set)} total for {dataset_name} in range "
        f"{start_datetime} to {end_datetime}"
    )
    return list(downloaded_set)


def _get_progress_file(base_target: str, dataset_name: str) -> Path:
    """Get the path to the progress tracking file."""
    return Path(base_target) / ".download_progress" / f"{dataset_name}_progress.json"


def _scan_downloaded_files(base_target: str, dataset_name: str) -> set:
    """Scan the directory structure to find all already-downloaded files.

    Returns a set of filenames (without path) that are already present.
    """
    downloaded_files = set()
    target_path = Path(base_target) / dataset_name

    if not target_path.exists():
        logger.info(f"Target directory does not exist yet: {target_path}")
        return downloaded_files

    # Scan all subdirectories for .nc files
    for nc_file in target_path.glob("**/*.nc"):
        filename = nc_file.name
        downloaded_files.add(filename)

    if len(downloaded_files) > 0:
        logger.info(f"Found {len(downloaded_files)} already-downloaded files in {target_path}")

    return downloaded_files


def _load_progress(base_target: str, dataset_name: str) -> dict:
    """Load download progress from checkpoint file."""
    progress_file = _get_progress_file(base_target, dataset_name)
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
    return {"downloaded": [], "failed": []}


def _save_progress(base_target: str, dataset_name: str, progress: dict):
    """Save download progress to checkpoint file."""
    progress_file = _get_progress_file(base_target, dataset_name)
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save progress file: {e}")


def download_full_dataset(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    base_target: str = "D:/thesis",
    max_keys: int = 500,
    overwrite: bool = False,
    resume: bool = True,
    max_workers: int = 64,
    rate_limit_per_sec: float = 90.0,
):
    """Download full dataset with resume capability.

    Args:
        resume: If True, resume from last checkpoint on connection failure.
    """
    next_page_token = None
    filenames = []

    while True:
        params = {"maxKeys": f"{max_keys}"}
        if next_page_token:
            params["nextPageToken"] = next_page_token

        dataset_filenames, response_json = list_dataset_files(session, base_url, dataset_name, dataset_version, params)

        filenames.extend(dataset_filenames)
        next_page_token = response_json.get("nextPageToken")
        if not next_page_token:
            break

    logger.info(f"Found {len(filenames)} files for dataset {dataset_name}/{dataset_version}")

    # First, scan the directory for already-downloaded files
    already_downloaded_in_dir = _scan_downloaded_files(base_target, dataset_name) if resume else set()

    # Then, load progress file for any tracked failed files
    progress = _load_progress(base_target, dataset_name) if resume else {"downloaded": [], "failed": []}

    downloaded_set = already_downloaded_in_dir.copy()
    failed_set = set(progress.get("failed", []))

    if len(downloaded_set) > 0:
        logger.info(
            f"Resuming download: {len(downloaded_set)} already in directory, {len(failed_set)} tracked failures"
        )

    # Filter out already-processed and unparseable filenames
    to_download = []
    for filename in filenames:
        if filename in downloaded_set or filename in failed_set:
            continue
        if parse_file_utc_timestamp(filename) is None:
            logger.warning(f"Skipping file with unexpected filename: {filename}")
            failed_set.add(filename)
            continue
        to_download.append(filename)

    logger.info(f"{len(to_download)} files to fetch ({len(downloaded_set)} already on disk, {len(failed_set)} skipped/failed)")

    if not to_download:
        _save_progress(base_target, dataset_name, {"downloaded": list(downloaded_set), "failed": list(failed_set)})
        return list(downloaded_set)

    # Pre-create year/month dirs to avoid mkdir races
    dirs_needed = set()
    for filename in to_download:
        dirs_needed.add(compute_local_filepath(dataset_name, filename, base_target)[0])
    for d in dirs_needed:
        d.mkdir(parents=True, exist_ok=True)

    rate_limiter = RateLimiter(rate_limit_per_sec)
    state_lock = threading.Lock()
    total = len(to_download)
    processed = 0
    new_successes = 0
    new_failures = 0
    checkpoint_every = 1000
    start_wall = time.monotonic()

    def _worker(filename: str):
        local_dir, _ = compute_local_filepath(dataset_name, filename, base_target)
        rate_limiter.acquire()
        return download_dataset_file(
            session, base_url, dataset_name, dataset_version,
            filename, str(local_dir), overwrite,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, f): f for f in to_download}
        for future in as_completed(futures):
            filename = futures[future]
            try:
                success, _ = future.result()
            except Exception as e:
                logger.exception(f"Worker crashed for {filename}: {e}")
                success = False

            with state_lock:
                if success:
                    downloaded_set.add(filename)
                    failed_set.discard(filename)
                    new_successes += 1
                else:
                    failed_set.add(filename)
                    new_failures += 1
                processed += 1

                if processed % checkpoint_every == 0 or processed == total:
                    elapsed = max(time.monotonic() - start_wall, 1e-6)
                    rate = processed / elapsed
                    eta_s = (total - processed) / rate if rate > 0 else float("inf")
                    logger.info(
                        f"Progress: {processed}/{total} (ok={new_successes}, fail={new_failures}) "
                        f"rate={rate:.1f} files/s, eta={eta_s/60:.1f} min"
                    )
                    _save_progress(base_target, dataset_name, {"downloaded": list(downloaded_set), "failed": list(failed_set)})

    _save_progress(base_target, dataset_name, {"downloaded": list(downloaded_set), "failed": list(failed_set)})
    logger.info(f"Full dataset download complete! ok={new_successes} fail={new_failures} total_on_disk={len(downloaded_set)}")
    return list(downloaded_set)


def get_max_worker_count(filesizes):
    size_for_threading = 10_000_000  # 10 MB
    average = sum(filesizes) / len(filesizes)
    # to prevent downloading multiple half files in case of a network failure with big files
    if average > size_for_threading:
        threads = 1
    else:
        threads = 10
    return threads


async def main():
    api_key = "<API_KEY>"
    dataset_name = "EV24"
    dataset_version = "2"
    base_url = "https://api.dataplatform.knmi.nl/open-data/v1"
    # When set to True, if a file with the same name exists the output is written over the file.
    # To prevent unnecessary bandwidth usage, leave it set to False.
    overwrite = False

    download_directory = "./dataset-download"

    # Make sure to send the API key with every HTTP request
    session = requests.Session()
    session.headers.update({"Authorization": api_key})

    # Verify that the download directory exists
    if not Path(download_directory).is_dir() or not Path(download_directory).exists():
        raise Exception(f"Invalid or non-existing directory: {download_directory}")

    filenames = []
    max_keys = 500
    next_page_token = None
    file_sizes = []
    # Use the API to get a list of all dataset filenames
    while True:
        # Retrieve dataset files after given filename
        dataset_filenames, response_json = list_dataset_files(
            session,
            base_url,
            dataset_name,
            dataset_version,
            {"maxKeys": f"{max_keys}", "nextPageToken": next_page_token},
        )
        file_sizes.extend(file["size"] for file in response_json.get("files"))
        # Store filenames
        filenames += dataset_filenames

        # If the result is not truncated, we retrieved all filenames
        next_page_token = response_json.get("nextPageToken")
        if not next_page_token:
            logger.info("Retrieved names of all dataset files")
            break

    logger.info(f"Number of files to download: {len(filenames)}")

    worker_count = get_max_worker_count(file_sizes)
    loop = asyncio.get_event_loop()

    # Allow up to 10 separate threads to download dataset files concurrently
    executor = ThreadPoolExecutor(max_workers=worker_count)
    futures = []

    # Create tasks that download the dataset files
    for dataset_filename in filenames:
        # Create future for dataset file
        future = loop.run_in_executor(
            executor,
            download_dataset_file,
            session,
            base_url,
            dataset_name,
            dataset_version,
            dataset_filename,
            download_directory,
            overwrite,
        )
        futures.append(future)

    # # Wait for all tasks to complete and gather the results
    future_results = await asyncio.gather(*futures)
    logger.info(f"Finished '{dataset_name}' dataset download")

    failed_downloads = list(filter(lambda x: not x[0], future_results))

    if len(failed_downloads) > 0:
        logger.warning("Failed to download the following dataset files:")
        logger.warning(list(map(lambda x: x[1], failed_downloads)))


if __name__ == "__main__":
    asyncio.run(main())
