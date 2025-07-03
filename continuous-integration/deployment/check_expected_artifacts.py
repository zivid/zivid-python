import argparse
from pathlib import Path

ZIVID_PYTHON = "zivid/zivid-python"
MASTER_REF = "refs/heads/master"

def _expected_artifacts(repository, commit_hash, ref):
    parent_dir = Path(__file__).resolve().parent
    artifacts_file = parent_dir / "expected-artifacts.txt"
    artifacts = artifacts_file.read_text().splitlines()
    artifacts.sort()
    version_file = parent_dir / "expected-version.txt"
    version = version_file.read_text().strip()
    if repository != ZIVID_PYTHON or ref != MASTER_REF:
        # On all branches but master in the zivid/zivid-python repo, a .dev0 suffix will be added to the version.
        version += ".dev0" + f"+{commit_hash[:8]}"

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
