from pathlib import Path
import argparse
import requests


def _download_sample(source_folder: str, sample_path: Path):
    url = f"https://raw.githubusercontent.com/zivid/python-samples/master/source/{source_folder}/{sample_path.name}"
    print(f"Download {source_folder}/{sample_path.name} from {url}")
    response = requests.get(url)
    sample_path.write_bytes(response.content)


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True, type=Path)
    return parser.parse_args()


def _main():
    args = _args()
    _download_sample("camera/basic", Path(args.destination) / "capture_from_file.py")
    _download_sample("camera/info_util_other", Path(args.destination) / "print_version_info.py")


if __name__ == "__main__":
    _main()
