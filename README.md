# KINA-Benchmark: Knowledge Index of Noah's Ark Benchmark 🚢

Tools to run model inference and **Pass@1** scoring on the **KINA** benchmark. 
**KINA** is an 899-item knowledge benchmark across 261 fine-grained disciplines. It provides a highly diagnostic evaluation of frontier LLMs with multi-discipline questions featuring **one correct** option among lettered choices **A–J**.

> Scoring reuses [lighteval](https://github.com/huggingface/lighteval) letter-extraction and Pass@*k* metrics (the same multi-choice path used for other letter-style tasks in the library). The **dataset file format** is the indexed JSON described below, not the legacy labeling-platform export.

## 🌟 Key Highlights

* **Disciplinary Representativeness (Theorem 1):** Unlike standard scaling-driven designs, KINA casts representativeness as a coverage-style objective over expert-elicited anchors, yielding a $(1-1/e)$ greedy approximation to ensure domain core competencies are truly tested.
* **Bonus-on-Bar Tournament Review:** Replaces the flat-payment annotation that typically permits "lazy consensus." Our tournament mechanism strictly FOSD-dominates flat payment in review quality, significantly improving dataset rigor.
* **Highly Diagnostic:** Evaluated on 42 frontier models, KINA reveals that current capability differentiation at the absolute frontier is heavily driven by soft-science content (humanities and social sciences) rather than hard sciences.
* 
## Environment setup

```bash
conda create -n kina_bench python=3.10 -y && conda activate kina_bench
pip install -e .
```

### SGLang (optional)

For local OpenAI-compatible servers (e.g. vLLM, SGLang), install the server stack you use; SGLang example:

```bash
pip install sglang[all]
```

See the [SGLang documentation](https://github.com/sgl-project/sglang) for details.

## Data file
(Note: To mitigate data contamination, the test split is typically distributed as an encrypted archive. Ensure you are using the authorized KINA-899.json)

1. Place the dataset JSON under **`KINA-Benchmark/data/`** (or symlink it from the monorepo root), using the name expected by `--data_name` (default: `KINA-899` → `data/KINA-899.json`).
2. Example (from the parent KINA repo):

   ```bash
   mkdir -p data
   ln -sf ../../data/KINA-899.json data/KINA-899.json
   ```

   If your file lives elsewhere or uses another name, point `ln -sf <source> data/KINA-899.json` to that path.

The expected format is a **JSON array** of 899 objects; each object includes at least:

| Field | Type | Description |
|--------|------|-------------|
| `index` | int | Global index `0 .. 898` (stable id for runs and results) |
| `discipline` | string | Subject / domain |
| `question` | string | Full question text |
| `options` | array | `{ "key", "answer", "explanation"?, "source"? }` per row |
| `correct_answer` | string | Single letter `A`–`J` |
| `question_source` | string | optional |
| `question_material` | string | optional (e.g. images / context) |

Internal runtime representation maps `discipline` → `metadata.category`, `question_source` → `metadata.source`, `question_material` → `metadata.materials`, and uses **`id` = `index`** for resume keys and result lines.

## Usage

### Commercial API (OpenAI, OpenRouter, etc.)

```bash
export OPENAI_BASE="https://api.openai.com/v1"   # or https://openrouter.ai/api/v1
export OPENAI_KEY="your-api-key"

python src/kina_bench/run_openai_chat.py \
    --model_id "gpt-4o" \
    --data_name "KINA-899" \
    --n_thread 32
```

You can also use a `.env` in the project root (see `.env.example`).

### Local OpenAI-compatible server (e.g. SGLang)

**1. Start the server** (example):

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --port 8000
```

**2. Run evaluation**

```bash
export OPENAI_BASE="http://localhost:8000/v1"
export OPENAI_KEY="EMPTY"

python src/kina_bench/run_openai_chat.py \
    --model_id "Qwen/Qwen3-8B" \
    --data_name "KINA-899" \
    --n_thread 64
```

### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | (required) | Model id passed to the API |
| `--data_name` | `KINA-899` | Basename of `data/{data_name}.json` (no extension) |
| `--n_sampling` | `1` | Parallel completions per question: **1, 4, or 8** |
| `--max_tokens` | `16384` | Max output tokens (or completion budget where applicable) |
| `--think_mode` | `none` | `none`, `think`, or `nothink` (Qwen-style chat template) |
| `--reasoning_effort` | `None` | For o1 / GPT-5 family: `low`, `medium`, `high` |
| `--n_thread` | `1` | Concurrent requests |
| `--overwrite` | `False` | Overwrite an existing `jsonl` from scratch |
| `--timeout` | `300` | Per-request timeout (seconds) |
| `--limit` | `None` | Cap how many *remaining* items to run (debug) |
| `--skip_inference` | `False` | Skip API calls; only evaluate existing `jsonl` if complete |

> **`--skip_inference`**: If some items failed in inference, post-processing (aggregate JSON) is skipped until the set is complete. Re-run the same command to resume, or use `--skip_inference` to force scoring on whatever finished.

### View aggregated scores

```bash
python src/kina_bench/pretty_print.py --data_name KINA-899
```

## Output layout

### Intermediate: `results/{model_id}/n{n}_tokens{max}/{data_name}.jsonl`

One JSON object per line: `id` (question index), `request`, `responses` (list of `content` / `reasoning`), `metadata` (full parsed row), token usage.

### Final: `results/{model_id}/n{n}_tokens{max}/{data_name}.json`

List of:

```json
{
    "id": 0,
    "score": 0.75,
    "extracted_predictions": [["A", "A"], ["B", "B"]],
    "gt": "A"
}
```

- `score`: Pass@1 in `[0, 1]`
- `extracted_predictions`: one list of extracted letters per parallel sample
- `gt`: `correct_answer` from the dataset

## Repository layout

```
KINA-Benchmark/
├── data/
│   └── KINA-899.json                  # add or symlink (not committed by default)
├── results/
│   └── {model_id}/
│       └── n{n}_tokens{max}/
│           ├── {data_name}.jsonl
│           └── {data_name}.json
├── src/
│   └── kina_bench/
│       ├── run_openai_chat.py
│       ├── utils.py
│       ├── pretty_print.py
│       └── config.py
└── pyproject.toml
```
