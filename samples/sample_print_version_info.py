"""Print version information for Python, zivid-python and Zivid SDK."""
import platform
import zivid


def _main():
    print("Python:       " + platform.python_version())
    print("zivid-python: " + zivid.__version__)
    print("Zivid SDK:    " + zivid.SDKVersion.full)


if __name__ == "__main__":
    _main()
