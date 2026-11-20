# KINA: Knowledge Index of Noah's Ark Benchmark 🚢
![KINA](images/kina_banner_white.svg)

Tools to run model inference and **Pass@1** scoring on the **KINA** benchmark. 
**KINA** is an 899-item knowledge benchmark across 261 fine-grained disciplines. It provides a highly diagnostic evaluation of frontier LLMs with multi-discipline questions featuring **one correct** option among lettered choices **A–J** (as in the public [KINA.json](https://huggingface.co/datasets/2077AIDataFoundation/KINA) release).

> Scoring reuses [lighteval](https://github.com/huggingface/lighteval) letter-extraction and Pass@*k* metrics (the same multi-choice path used for other letter-style tasks in the library). The **dataset file format** is the indexed JSON described below, not the legacy labeling-platform export.
  
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

1. Download the dataset from [KINA](https://huggingface.co/datasets/2077AIDataFoundation/KINA).
2. Place the dataset JSON under **`KINA/data/`**. For `data/KINA.json`, pass `--data_name KINA`. The CLI defaults to `data/KINA-899.json`.
3. Example (from the parent KINA repo):

   ```bash
   mkdir -p data
   ln -sf ../../data/KINA.json data/KINA.json
   ```

   If your file lives elsewhere or uses another name, point `ln -sf <source> data/KINA.json` to that path.

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

### OpenAI-compatible Chat Completions

```bash
export OPENAI_BASE="https://api.openai.com/v1"   # or https://openrouter.ai/api/v1
export OPENAI_KEY="your-api-key"

python src/kina_bench/run_openai_chat.py \
    --model_id "gpt-4o" \
    --data_name "KINA" \
    --n_thread 32
```

You can also use a `.env` in the project root (see `.env.example`).

To run a small streaming smoke test with an endpoint that expects
`max_completion_tokens`:

```bash
python -m kina_bench.run_openai_chat \
    --model_id "your-model-id" \
    --data_name KINA \
    --limit 3 --stream --max_completion_tokens 16384 --timeout 300
```

Use `--max_tokens` instead for endpoints that expect that field. These options
are mutually exclusive. Omitting both sends `max_tokens: 16384` for every
model. The runner uses the endpoint defaults for reasoning and sampling
settings such as temperature and top-p. Use `--n_sampling 1` for endpoints that
accept only one completion per request.

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
    --data_name "KINA" \
    --n_thread 64
```

### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | (required) | Model id passed to the API |
| `--data_name` | `KINA-899` | Basename of `data/{data_name}.json` (no extension) |
| `--n_sampling` | `1` | Parallel completions per question: **1, 4, or 8** |
| `--max_tokens` | `16384` | Send the budget in the `max_tokens` field |
| `--max_completion_tokens` | unset | Send the budget in the `max_completion_tokens` field instead |
| `--stream` | off | Read Chat Completions chunks and collect each answer before scoring |
| `--n_thread` | `1` | Concurrent requests |
| `--requests_per_second` | `1` | Maximum request starts per second for this CLI process, including retries. Accepts positive decimal values. |
| `--overwrite` | `False` | Overwrite an existing `jsonl` from scratch |
| `--timeout` | `300` | Deadline per attempt in seconds, including reading the response stream |
| `--limit` | unset | Score the first N questions in a separate smoke run |
| `--skip_inference` | off | Score completed items from an existing JSONL locally |

If inference fails for some items, re-run the same command to resume. To score
the completed items immediately, add `--skip_inference` and keep the same
model, dataset, sampling, token and stream options. Include the same
`--limit` if the original run was limited.

For example, `--n_thread 20 --requests_per_second 1` starts the first request
immediately and waits at least one second between starts. Up to 20 requests can
be in flight. When all slots are busy, the next request waits for a slot and for
the start interval to elapse.

### View aggregated scores

```bash
python src/kina_bench/pretty_print.py --data_name KINA
```

## Output layout

New runs write to `results/{model_id}/{generation}/`. The generation folder
includes the sample count, token value, token field, response mode and optional
question limit. For example, `n1_tokens16384_max_completion_tokens_stream_limit3`.
This keeps old and new runs, streaming and JSON runs, and limited and full runs
in separate folders. `pretty_print.py` can still read existing scored files.

### Intermediate: `{data_name}.jsonl`

One JSON object per line: `id` (question index), `request`, `responses` (list of `content` / `reasoning`), `metadata` (full parsed row), token usage.

### Final: `{data_name}.json`

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
│   └── KINA.json                  # add or symlink (not committed by default)
├── results/
│   └── {model_id}/
│       └── {generation}/
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
