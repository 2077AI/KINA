import json
import logging
import re
from string import ascii_uppercase
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

KINA_SAMPLE_COUNTS = (1, 4, 8)
FINAL_ANSWER = re.compile(r"Answer\s*[:：]\s*(.+)", re.IGNORECASE)
LETTER_ANSWER = re.compile(r"([A-J])|\$([A-J])\$?", re.IGNORECASE)
MARKDOWN_WRAPPER = re.compile(r"(\*\*|__|`)(.+)\1")


def unwrap_markdown(text: str) -> str:
    match = MARKDOWN_WRAPPER.fullmatch(text)
    return match[2].strip() if match else text


def extract_final_answer(response: str) -> str | None:
    lines = response.rstrip().splitlines()
    if not lines:
        return None

    final_line = unwrap_markdown(lines[-1].strip().removesuffix(".").rstrip())
    answer = FINAL_ANSWER.fullmatch(final_line)
    if answer is None:
        return None

    value = unwrap_markdown(answer[1].strip().removesuffix(".").rstrip())
    letter = LETTER_ANSWER.fullmatch(value)
    return (letter[1] or letter[2]).upper() if letter else None


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
    """Score Pass@1 as the fraction of samples with the correct final answer."""
    options = metadata_doc["options"]
    ground_truth = metadata_doc["ground_truth"]
    if ground_truth not in options:
        raise ValueError(f"ground_truth {ground_truth!r} is not in options {sorted(options)}")
    if len(responses) not in KINA_SAMPLE_COUNTS:
        raise ValueError(f"Expected one of {KINA_SAMPLE_COUNTS} responses, got {len(responses)}")

    predictions = [extract_final_answer(response) for response in responses]
    score = predictions.count(ground_truth) / len(responses)
    extracted_predictions = [[letter] if letter in options else [] for letter in predictions]
    return dict(score=score, extracted_predictions=extracted_predictions)


if __name__ == "__main__":
    from kina_bench.config import PROJECT_ROOT
    from os.path import join

    sample = join(PROJECT_ROOT, "data", "KINA-899.json")
    data = load_data(sample)
    for doc in data[:3]:
        messages = get_messages(doc)
        print(messages)
