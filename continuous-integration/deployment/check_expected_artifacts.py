import argparse
from pathlib import Path

def _expected_artifacts(repository, commit_hash, ref):
    parent_dir = Path(__file__).resolve().parent
    artifacts_file = parent_dir / "expected-artifacts.txt"
    artifacts = artifacts_file.read_text().splitlines()
    artifacts.sort()
    version_file = parent_dir / "expected-version.txt"
    version = version_file.read_text().strip()
    # Note: unlike on master, the 2.11 branch hardcodes the package version in setup.py,
    # so no .dev0 suffix is added on non-release refs.
    return [artifact.format(version=version) for artifact in artifacts]


def _present_artifacts():
    artifacts_dir = Path(__file__).resolve().parents[2] / "distribution"
    artifacts = [file_path.name for file_path in artifacts_dir.glob("*")]
    artifacts.sort()
    return artifacts


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        help="The repository the artifacts are built from, e.g., zivid/zivid-python.",
        type=str,
        required=True
    )
    parser.add_argument(
        "--commit-hash",
        help="The commit hash to check against the expected artifacts.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ref",
        help="The ref the artifacts are built from, e.g., a branch name or tag.",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    present_artifacts = _present_artifacts()
    print("Present artifacts:\n  " + "\n  ".join(present_artifacts))
    expected_artifacts = _expected_artifacts(args.repository, args.commit_hash, args.ref)
    print("Expected artifacts:\n  " + "\n  ".join(expected_artifacts))

    assert present_artifacts == expected_artifacts


if __name__ == "__main__":
    _main()
