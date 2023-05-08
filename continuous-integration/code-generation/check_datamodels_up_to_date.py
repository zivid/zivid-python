from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

from datamodel_frontend_generator import generate_all_datamodels


def _check_datamodels_up_to_date(repo_root: Path):
    regeneration_script = Path(
        "continuous-integration/code-generation/docker_generate_datamodels.sh"
    )
    if not (repo_root / regeneration_script).is_file():
        raise RuntimeError(
            f"Could not find {regeneration_script}. Please update path in {Path(__file__).name}."
        )

    module_dir = repo_root / "modules" / "zivid"

    with TemporaryDirectory() as tmpdir:
        tmpdirpath = Path(tmpdir)
        generate_all_datamodels(dest_dir=tmpdirpath)
        generated_files = tmpdirpath.rglob("*.py")
        print()
        print("-" * 70)
        print(f"Finished generating files to {tmpdirpath}")
        print("-" * 70)
        for generated_file in generated_files:
            committed_file = module_dir / generated_file.relative_to(tmpdirpath)
            print(f"Comparing {str(committed_file):70} <-->  {generated_file}")
            committed_content = committed_file.read_text(encoding="utf-8")
            generated_content = generated_file.read_text(encoding="utf-8")
            if not committed_content == generated_content:
                raise RuntimeError(
                    dedent(
                        f"""
                        Found difference in {generated_file.name}.
                        Please run './{regeneration_script}' to regenerate data models.
                        """
                    )
                )
        print("All OK")


if __name__ == "__main__":
    _check_datamodels_up_to_date(repo_root=Path(__file__).parents[2])
