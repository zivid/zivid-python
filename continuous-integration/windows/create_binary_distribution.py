from common import repo_root, run_process


def _main():
    dist_dir = repo_root() / "dist"

    source_dist_paths = list(dist_dir.glob("zivid*.tar.gz"))
    if not source_dist_paths:
        return ScriptFailure(error_message="No source distribution found matching 'zivid*.tar.gz'")
    if len(source_dist_paths) > 1:
        return ScriptFailure(
            error_message=f"Expected exactly one source distribution,"
                          f" but found {len(source_dist_paths)}: {source_dist_paths}"
        )

    source_dist_path = str(source_dist_paths[0])

    run_process(
        args=(
            "pip",
            "wheel",
            "--no-deps",
            source_dist_path,
            "--wheel-dir",
            str(dist_dir),
        ),
    )


if __name__ == "__main__":
    _main()
