import zipfile
from pathlib import Path
import argparse
import tempfile
import shutil
import requests


def _extract_sample_zip(
    sample_data_zip, file_camera_destination, point_cloud_destination
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_data_dir = Path(tmp_dir)
        with zipfile.ZipFile(str(sample_data_zip), "r") as zip_ref:
            zip_ref.extractall(str(sample_data_dir))
        file_camera = (
            sample_data_dir / "ZividSampleData2" / "FileCameraZividOne.zfc"
        ).resolve()
        shutil.move(str(file_camera), str(file_camera_destination))
        sample_data_file = (
            sample_data_dir / "ZividSampleData2" / "Zivid3D.zdf"
        ).resolve()
        shutil.move(str(sample_data_file), str(point_cloud_destination))


def _download_sample_data_zip(sample_data_zip):
    url = "http://www.zivid.com/hubfs/softwarefiles/ZividSampleData2.zip"

    response = requests.get(url)
    sample_data_zip.write_bytes(response.content)


def download_and_extract(file_camera_destination, point_cloud_destination):
    with tempfile.TemporaryDirectory() as tmp_dir:
        zipped_dir = Path(tmp_dir) / "sample_data.zip"
        _download_sample_data_zip(zipped_dir)
        _extract_sample_zip(
            zipped_dir, file_camera_destination, point_cloud_destination
        )


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination-file-camera", required=True, type=Path)
    parser.add_argument("--destination-point-cloud", required=True, type=Path)
    return parser.parse_args()


def _main():
    args = _args()
    download_and_extract(
        file_camera_destination=args.destination_file_camera,
        point_cloud_destination=args.destination_point_cloud,
    )


if __name__ == "__main__":
    _main()
