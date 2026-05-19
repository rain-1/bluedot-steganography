import argparse
import random
from pathlib import Path

from datasets import load_from_disk


DEFAULT_INPUT_PATH = "outputs/MMLU-Pro-filtered"
DEFAULT_OUTPUT_PATH = "outputs/MMLU-Pro-filtered-shuffled"
DEFAULT_SEED = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def shuffled_questions(dataset, seed):
    rng = random.Random(seed)
    questions = list(dataset["question"])
    rng.shuffle(questions)

    return dataset.map(
        lambda row, index: {
            "question": questions[index],
            "original_question": row["question"],
        },
        with_indices=True,
    )


def main():
    args = parse_args()
    dataset = load_from_disk(str(Path(args.input)))
    shuffled_dataset = shuffled_questions(dataset, args.seed)
    shuffled_dataset.save_to_disk(str(Path(args.output)))

    print(f"Wrote {len(shuffled_dataset)} rows to {args.output}")
    print(f"Seed: {args.seed}")

    for row in shuffled_dataset.select(range(min(5, len(shuffled_dataset)))):
        print()
        print(f"Question: {row['question']}")
        print("Answers:")
        for option in row["options"]:
            print(f"- {option}")


if __name__ == "__main__":
    main()
