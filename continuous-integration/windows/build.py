from common import repo_root, run_process


def _main():
    root = repo_root()
    run_process(("pip", "install", "--verbose", str(root)))


if __name__ == "__main__":
    _main()
