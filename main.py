import argparse

from src.collection.collection import DatasetCollection
from src.study.builder import build_study_design_from_yaml
from src.study.study import Study
from src.results.exporter import export_to_csv


def _get_reader(collection: DatasetCollection):
    from src.reader.reader import CWRUFileReader, PaderbornFileReader
    readers = {'cwru': CWRUFileReader, 'paderborn': PaderbornFileReader}
    name = collection.name
    if name not in readers:
        raise ValueError(f"No reader for collection '{name}'. Known: {list(readers)}")
    return readers[name]()


def main():
    parser = argparse.ArgumentParser(description="Run a domain shift study.")
    parser.add_argument("--collection", help="Path to collection YAML")
    parser.add_argument("--study", help="Path to study YAML")
    parser.add_argument("--save-path", help="Path to save raw metrics in csv file")
    args = parser.parse_args()

    collection = DatasetCollection(args.collection)
    reader = _get_reader(collection)
    study_design, _, _ = build_study_design_from_yaml(args.study, collection)
    study = Study(collection, reader)
    study_solution = study.run(study_design)
    export_to_csv(study_solution, args.save_path, study_design.name)


if __name__ == '__main__':
    main()
