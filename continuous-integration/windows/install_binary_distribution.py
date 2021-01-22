from common import repo_root, run_process


def _find_wheel(root):
    extension = ".whl"
    dist_dir = root / "dist"
    wheels = list(dist_dir.glob("*" + extension))

    if len(wheels) == 0:
        raise RuntimeError(
            "Failed to find any {ext} file in {dir}.".format(
                ext=extension, dir=dist_dir
            )
        )
    if len(wheels) > 1:
        raise RuntimeError(
            "Found multiple {ext} files in {dir}.".format(ext=extension, dir=dist_dir)
        )

    return wheels[0]


def _main():
    root = repo_root()
    wheel_file = _find_wheel(root)
    print(
        "Found wheel: {path}. Will attempt to install.".format(path=wheel_file),
        flush=True,
    )
    run_process(("python", "-m", "pip", "install", str(wheel_file)))


if __name__ == "__main__":
    _main()
