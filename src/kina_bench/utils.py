import json
import logging
from string import ascii_uppercase
from typing import Any, Dict, List, Optional

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.dynamic_metrics import multilingual_extractive_match_metric
from lighteval.metrics.utils.extractive_match_utils import IndicesExtractionConfig
from lighteval.utils.language import Language
from lighteval.tasks.requests import Doc

logger = logging.getLogger(__name__)


_EXTRACTION_METRIC = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=6,
)

# n parallel samples per question -> lighteval Pass@1 (compatible with A–J letter answers)
KINA_METRICS = {
    1: Metrics.gpqa_instruct_pass_at_1_1n.value,
    4: Metrics.gpqa_instruct_pass_at_1_4n.value,
    8: Metrics.gpqa_instruct_pass_at_1_8n.value,
}


def _options_list_to_map(options: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for opt in options or []:
        key = opt.get("key")
        if not key or not isinstance(key, str):
            continue
        key = key.strip().upper()
        out[key] = {
            "id": key,
            "answer": (opt.get("answer") or "").strip() if opt.get("answer") is not None else "",
            "explanation": opt.get("explanation"),
            "source": opt.get("source"),
        }
    return out


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load KINA items from a JSON array (e.g. KINA-899.json).

    Expected fields per item:
    - index: int, global question index (0..N-1)
    - question: str
    - options: list of {key, answer, explanation?, source?}
    - correct_answer: str (A–J)
    - discipline, question_source, question_material: optional metadata
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Root JSON must be a list, got {type(data)}")

    parsed: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "index" not in item or "question" not in item:
            raise ValueError("Each item must include 'index' and 'question'")

        idx = item["index"]
        if not isinstance(idx, int):
            try:
                idx = int(idx)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid index: {item.get('index')!r}") from e

        options = _options_list_to_map(item.get("options") or [])
        ground = item.get("correct_answer")
        if ground is not None and isinstance(ground, str):
            ground = ground.strip().upper()
        if not ground or len(ground) != 1 or ground not in ascii_uppercase[:10]:
            logger.warning("Item index=%s: invalid or missing correct_answer: %r", idx, item.get("correct_answer"))

        parsed.append(
            {
                "id": idx,
                "question": item["question"],
                "options": options,
                "ground_truth": ground,
                "category": item.get("discipline"),
                "source": item.get("question_source"),
                "materials": item.get("question_material"),
            }
        )

    parsed.sort(key=lambda d: d["id"])
    return parsed


def get_kina_metric(n_sampling: int):
    """Return the lighteval Pass@1 metric for n parallel completions (1, 4, or 8)."""
    if n_sampling in KINA_METRICS:
        return KINA_METRICS[n_sampling]
    raise ValueError(
        f"n_sampling={n_sampling} is not supported. "
        f"Supported values are: {list(KINA_METRICS.keys())}."
    )


prompt_template = """Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J. Think step by step before answering.
Question:
{question}
Options:
{options}"""


def get_messages(doc: Dict[str, Any]):
    question = doc["question"]
    options = doc["options"]
    keys = sorted(options.keys())
    option_str = "\n".join([f"{letter}: {options[letter]['answer'].strip()}" for letter in keys])

    messages = [
        {"role": "user", "content": prompt_template.format(question=question, options=option_str)}
    ]
    return messages


def judge_score(metadata_doc: Dict, responses: List[str]):
    """
    Compute Pass@1 vs ground truth and extract predicted letters per response (A–J).

    lighteval's PassAtK + extractive match overwrite `doc.specific["extracted_predictions"]`
    on each inner call, so we run extraction once per response with a fresh Doc, same as before.
    """
    question = metadata_doc["question"]
    options = metadata_doc["options"]
    ground_truth = metadata_doc["ground_truth"]

    n_sampling = len(responses)
    keys = sorted(options.keys())
    choices = keys
    if ground_truth not in options:
        raise ValueError(f"ground_truth {ground_truth!r} is not in options {keys}")
    correct_index = keys.index(ground_truth)

    doc_for_score = Doc(
        query=question,
        choices=choices,
        gold_index=correct_index,
        instruction=question,
    )

    metric = get_kina_metric(n_sampling)
    result = metric.compute(
        golds=[ground_truth],
        predictions=responses,
        formatted_doc=doc_for_score,
    )
    score = list(result.values())[0]

    extracted_predictions: List[List[str]] = []
    for resp in responses:
        doc_for_extraction = Doc(
            query=question,
            choices=choices,
            gold_index=correct_index,
            instruction=question,
        )
        _EXTRACTION_METRIC.sample_level_fn([ground_truth], [resp], doc_for_extraction)
        if doc_for_extraction.specific:
            preds = doc_for_extraction.specific.get("extracted_predictions", [])
            extracted_predictions.append(preds)
        else:
            extracted_predictions.append([])

    return dict(score=score, extracted_predictions=extracted_predictions)


if __name__ == "__main__":
    from kina_bench.config import PROJECT_ROOT
    from os.path import join

    sample = join(PROJECT_ROOT, "data", "KINA-899.json")
    data = load_data(sample)
    for doc in data[:3]:
        messages = get_messages(doc)
        print(messages)
