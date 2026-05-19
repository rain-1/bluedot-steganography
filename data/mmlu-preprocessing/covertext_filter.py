import argparse
import re
from difflib import SequenceMatcher

from datasets import load_dataset

dataset_name = 'TIGER-Lab/MMLU-Pro'

MIN_ANSWER_WORDS = 8
MAX_TOKEN_OVERLAP = 0.75
MAX_SEQUENCE_SIMILARITY = 0.85

# columns are:
# - question
# - options
# - answer
# - category
# and some more

def answer_tokens(answer):
    return set(re.findall(r"[a-z0-9]+", answer.lower()))


def is_sentence_long_answer(answer):
    words = re.findall(r"\b\w+\b", answer)
    return len(words) >= MIN_ANSWER_WORDS


def are_fairly_different(answer_a, answer_b):
    tokens_a = answer_tokens(answer_a)
    tokens_b = answer_tokens(answer_b)

    if not tokens_a or not tokens_b:
        return False

    token_overlap = len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))
    sequence_similarity = SequenceMatcher(
        None,
        answer_a.lower().strip(),
        answer_b.lower().strip(),
    ).ratio()

    return (
        token_overlap <= MAX_TOKEN_OVERLAP
        and sequence_similarity <= MAX_SEQUENCE_SIMILARITY
    )


def has_good_steg_covertext(row):
    options = row['options']

    if not options or not all(is_sentence_long_answer(option) for option in options):
        return False

    for index, option in enumerate(options):
        for other_option in options[index + 1:]:
            if not are_fairly_different(option, other_option):
                return False

    return True


def keep_highly_rated(row):
    return row['category'] == 'health' and has_good_steg_covertext(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--split',
        choices=('validation', 'test'),
        default='validation',
        help='Dataset split to filter. validation has 70 rows; test has about 12k rows.',
    )
    return parser.parse_args()


args = parse_args()
dataset = load_dataset(dataset_name, split=args.split)
filtered_dataset = dataset.filter(keep_highly_rated)
filtered_dataset.save_to_disk("outputs/MMLU-Pro-filtered")
