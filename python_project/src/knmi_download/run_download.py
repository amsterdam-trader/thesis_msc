"""CLI runner for threaded full-dataset KNMI downloads.

Example:
    python run_download.py \
        --dataset hourly-in-situ-meteorological-observations-validated \
        --version 1.0 \
        --token YOUR_TOKEN \
        --target "C:/Users/floris/Desktop/MSC/thesis_msc/data/knmi_obs1926" \
        --workers 64 --rate 90
"""
import argparse
import logging

from download_full_dataset import create_resilient_session, download_full_dataset

BASE_URL = "https://api.dataplatform.knmi.nl/open-data/v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--rate", type=float, default=90.0)
    p.add_argument("--max-keys", type=int, default=500)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    session = create_resilient_session(token=args.token, pool_size=max(128, args.workers * 2))
    download_full_dataset(
        session=session,
        base_url=BASE_URL,
        dataset_name=args.dataset,
        dataset_version=args.version,
        base_target=args.target,
        max_keys=args.max_keys,
        overwrite=args.overwrite,
        max_workers=args.workers,
        rate_limit_per_sec=args.rate,
    )


if __name__ == "__main__":
    main()
