from pathlib import Path


def test_data_dir():
    return (Path(__file__).parent.parent / "test" / "test_data").resolve()
