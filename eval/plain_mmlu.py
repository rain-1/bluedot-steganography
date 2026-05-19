from pathlib import Path

from datasets import load_from_disk
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


DEFAULT_DATASET_PATH = "outputs/MMLU-Pro-filtered"
REPO_ROOT = Path(__file__).resolve().parents[1]

ANSWER_TEMPLATE = """
{question}

{choices}

Answer with the single letter of the best answer ({letters}).
""".strip()


def load_mmlu_samples(dataset_path: str) -> MemoryDataset:
    path = Path(dataset_path)
    if not path.exists():
        path = REPO_ROOT / path

    dataset = load_from_disk(str(path))
    samples = []

    for row in dataset:
        samples.append(
            Sample(
                id=row["question_id"],
                input=row["question"],
                choices=row["options"],
                target=row["answer"],
                metadata={
                    "category": row["category"],
                    "src": row["src"],
                    "answer_index": row["answer_index"],
                },
            )
        )

    return MemoryDataset(samples=samples, name="filtered-mmlu-pro")


@task
def plain_mmlu(
    dataset_path: str = DEFAULT_DATASET_PATH,
    max_tokens: int = 32,
) -> Task:
    return Task(
        dataset=load_mmlu_samples(dataset_path),
        solver=multiple_choice(template=ANSWER_TEMPLATE, max_tokens=max_tokens),
        scorer=choice(),
    )
