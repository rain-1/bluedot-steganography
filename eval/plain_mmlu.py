from pathlib import Path

from datasets import load_from_disk
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import (
    NOANSWER,
    Metric,
    SampleScore,
    Score,
    Target,
    accuracy,
    choice,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from inspect_ai.solver import multiple_choice


DEFAULT_DATASET_PATH = "outputs/MMLU-Pro-filtered"
DEFAULT_MAX_TOKENS = 256
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


@metric
def no_answer_rate() -> Metric:
    def metric(scores: list[SampleScore]) -> float:
        no_answers = [
            score for score in scores
            if score.score.value == NOANSWER
        ]
        return len(no_answers) / len(scores)

    return metric


@scorer(metrics=[accuracy(), stderr(), no_answer_rate()])
def choice_or_no_answer_on_truncation():
    choice_scorer = choice()

    async def score(state: TaskState, target: Target) -> Score:
        if state.output.stop_reason == "max_tokens":
            return Score(
                value=NOANSWER,
                answer=state.output.completion,
                explanation="Model output hit max_tokens before a scorable answer.",
                metadata={"stop_reason": state.output.stop_reason},
            )

        return await choice_scorer(state, target)

    return score


@task
def plain_mmlu(
    dataset_path: str = DEFAULT_DATASET_PATH,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Task:
    return Task(
        dataset=load_mmlu_samples(dataset_path),
        solver=multiple_choice(template=ANSWER_TEMPLATE, max_tokens=max_tokens),
        scorer=choice_or_no_answer_on_truncation(),
    )
