"""Print version information for Python, zivid-python and Zivid SDK."""
import platform
import zivid


def _main():
    print("Python: {}".format(platform.python_version()))
    print("zivid-python: {}".format(zivid.__version__))
    print("Zivid SDK: {}".format(zivid.SDKVersion.full))


if __name__ == "__main__":
    _main()
