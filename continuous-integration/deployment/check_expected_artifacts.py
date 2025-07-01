import argparse
import os
from pathlib import Path


def _expected_artifacts(commit_hash):
    parent_dir = Path(__file__).resolve().parent
    artifacts_file = parent_dir / "expected-artifacts.txt"
    artifacts = artifacts_file.read_text().splitlines()
    artifacts.sort()
    version_file = parent_dir / "expected-version.txt"
    version = version_file.read_text().strip()
    return [artifact.format(version=version, commit_hash=commit_hash[:8]) for artifact in artifacts]


def _present_artifacts():
    artifacts_dir = Path(__file__).resolve().parents[2] / "distribution"
    artifacts = [file_path.name for file_path in artifacts_dir.glob("*")]
    artifacts.sort()
    return artifacts


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--commit-hash",
        help="The commit hash to check against the expected artifacts.",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    present_artifacts = _present_artifacts()
    print("Present artifacts:\n  " + "\n  ".join(present_artifacts))
    expected_artifacts = _expected_artifacts(args.commit_hash)
    print("Expected artifacts:\n  " + "\n  ".join(expected_artifacts))

    assert present_artifacts == expected_artifacts


if __name__ == "__main__":
    _main()
