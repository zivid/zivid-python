from common import repo_root, run_process, install_pip_dependencies


def _main():
    root = repo_root()
    install_pip_dependencies(
        root / "continuous-integration" / "python-requirements" / "build.txt"
    )
    run_process(("python", str(root / "setup.py"), "bdist_wheel"), workdir=root)


if __name__ == "__main__":
    _main()
