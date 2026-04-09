import argparse
import tempfile
from pathlib import Path

from src.collection.collection import DatasetCollection
from src.study.builder import build_study_design_from_yaml
from src.study.study import Study
from src.results.exporter import export_to_csv


def main():
    parser = argparse.ArgumentParser(description="Run a domain shift study.")
    parser.add_argument("--collection", required=True, help="Path to collection YAML")
    parser.add_argument("--study", help="Path to study YAML (required without --download)")
    parser.add_argument("--save-path", help="Path to save raw metrics in csv file")
    parser.add_argument("--download", action="store_true",
                        help="Download dataset files listed in the collection YAML")
    args = parser.parse_args()

    if args.download:
        from src.collection.downloader import download_collection
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
    study_design, _, _ = build_study_design_from_yaml(args.study, collection)
    study = Study(collection, reader)
    study_solution = study.run(study_design)
    if args.save_path:
        save_path = args.save_path
    else:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            suffix='.csv', delete=False, prefix=f"{study_design.name}_", dir=results_dir
        )
        tmp.close()
        save_path = tmp.name

    export_to_csv(study_solution, save_path, study_design.name)
    print(f"Results saved to {save_path}")


if __name__ == '__main__':
    main()
