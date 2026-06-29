import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from collection.collection import DatasetCollection
from study.builder import build_study_design_from_yaml
from study.storage import StorageConfig
from study.study import Study
from results.exporter import export_to_csv


def _resolve_storage_path(value: str) -> Path:
    """Resolve a storage YAML argument.

    A bare filename (no directory component) is looked up inside
    configs/storage/.  A path that already contains a separator is
    used as-is, so absolute paths and relative paths with directories
    both work.
    """
    p = Path(value)
    if p.parent == Path("."):
        return Path("configs/storage") / p
    return p


def main():
    parser = argparse.ArgumentParser(description="Run a domain shift study.")
    parser.add_argument("--collection", required=True, help="Path to collection YAML")
    parser.add_argument("--study", help="Path to study YAML (required without --download)")
    parser.add_argument("--save-path", help="Path to save raw metrics in CSV file")
    parser.add_argument(
        "--storage",
        nargs="?",
        const="default.yaml",
        default="default.yaml",
        metavar="YAML",
        help="Storage policy YAML. "
             "Omit the flag entirely or pass it without a value to use "
             "configs/storage/default.yaml. Pass a filename to use "
             "configs/storage/<name>.yaml, or a full path to use that file directly. "
             "Controls which artifacts are persisted: model weights, config "
             "snapshots, and study design.",
    )
    parser.add_argument("--download", action="store_true",
                        help="Download dataset files listed in the collection YAML")
    args = parser.parse_args()

    if args.download:
        from collection.downloader import download_collection
        download_collection(args.collection)
        return

    if args.study is None:
        parser.error("--study is required when not using --download")

    collection = DatasetCollection(args.collection)
    reader = collection.reader
    if reader is None:
        raise ValueError(
            f"Collection '{collection.name}' has no reader configured. "
            "Add a 'reader:' key pointing to a reader YAML."
        )

    storage_path = _resolve_storage_path(args.storage)
    storage_config = StorageConfig.from_yaml(storage_path)

    study_design, _, _ = build_study_design_from_yaml(args.study, collection)
    study = Study(collection, reader, storage_config=storage_config)
    study_solution, save_dir = study.run_and_save(study_design)

    if args.save_path:
        csv_path = args.save_path
    else:
        csv_path = str(save_dir / f"{study_design.name}.csv")

    export_to_csv(study_solution, csv_path, study_design.name)
    print(f"CSV metrics saved to {csv_path}")


if __name__ == '__main__':
    main()
