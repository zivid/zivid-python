import zipfile
from pathlib import Path
import argparse
import tempfile
import shutil
import requests


def _extract_sample_zip(sample_data_zip, destination):
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_data_dir = Path(tmp_dir)
        with zipfile.ZipFile(str(sample_data_zip), "r") as zip_ref:
            zip_ref.extractall(str(sample_data_dir))
        sample_data_file = (sample_data_dir / "MiscObjects.zdf").resolve()
        shutil.move(str(sample_data_file), str(destination))


def _download_sample_data_zip(sample_data_zip):
    url = "https://zivid.com/software/ZividSampleData.zip"

    response = requests.get(url)
    sample_data_zip.write_bytes(response.content)


def download_and_extract(destination):
    with tempfile.TemporaryDirectory() as tmp_dir:
        zipped_dir = Path(tmp_dir) / "sample_data.zip"
        _download_sample_data_zip(zipped_dir)
        _extract_sample_zip(zipped_dir, destination)


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True, type=Path)
    return parser.parse_args()


def _main():
    args = _args()
    download_and_extract(args.destination)


if __name__ == "__main__":
    _main()
