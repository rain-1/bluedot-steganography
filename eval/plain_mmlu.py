from pathlib import Path

from datasets import load_from_disk
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
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
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_MIN_P = 0.0
DEFAULT_PRESENCE_PENALTY = 1.5
DEFAULT_REPETITION_PENALTY = 1.0
REPO_ROOT = Path(__file__).resolve().parents[1]

ANSWER_TEMPLATE = """
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.
Do not think step by step. Do not explain your reasoning.
/no_think

{question}

{choices}
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
    disable_reasoning: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    min_p: float = DEFAULT_MIN_P,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
) -> Task:
    extra_body = {
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
    }
    if disable_reasoning:
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": False,
        }

    config = GenerateConfig(
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        extra_body=extra_body,
    )

    return Task(
        dataset=load_mmlu_samples(dataset_path),
        solver=multiple_choice(template=ANSWER_TEMPLATE, max_tokens=max_tokens),
        scorer=choice_or_no_answer_on_truncation(),
        config=config,
    )
