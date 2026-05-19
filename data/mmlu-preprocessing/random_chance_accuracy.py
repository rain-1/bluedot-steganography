import argparse
from pathlib import Path

from datasets import load_from_disk


DEFAULT_DATASET_PATH = "outputs/MMLU-Pro-filtered"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", nargs="?", default=DEFAULT_DATASET_PATH)
    return parser.parse_args()


def random_chance_accuracy(dataset):
    chances = [1 / len(options) for options in dataset["options"] if options]
    return sum(chances) / len(chances)


def main():
    args = parse_args()
    dataset = load_from_disk(str(Path(args.dataset_path)))
    accuracy = random_chance_accuracy(dataset)

    print(f"Rows: {len(dataset)}")
    print(f"Expected random chance accuracy: {accuracy:.6f}")


if __name__ == "__main__":
    main()
