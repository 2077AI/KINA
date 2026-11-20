import argparse
import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass

from os.path import join, dirname
from typing import Dict

from openai import APIConnectionError, AsyncOpenAI, AsyncStream, InternalServerError, RateLimitError
from openai.types.chat import ChatCompletionChunk
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from kina_bench.config import PROJECT_ROOT
from dotenv import find_dotenv, load_dotenv

from kina_bench.utils import load_data, get_messages, judge_score, KINA_SAMPLE_COUNTS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


@dataclass
class Completion:
    responses: list[dict[str, str | None]]
    input_tokens: int = 0
    output_tokens: int = 0


class RequestRateLimiter:
    def __init__(self, requests_per_second: float):
        self.interval = 1 / requests_per_second
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        async with self.lock:
            await asyncio.sleep(max(0.0, self.next_start - loop.time()))
            self.next_start = loop.time() + self.interval


def reasoning_text(message) -> str | None:
    return getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)


async def collect_stream(stream: AsyncStream[ChatCompletionChunk], expected_choices: int) -> Completion:
    content_parts: dict[int, list[str]] = {}
    reasoning_parts: dict[int, list[str]] = {}
    finished: set[int] = set()
    usage = None
    async with stream:
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            for choice in chunk.choices:
                content_parts.setdefault(choice.index, [])
                reasoning_parts.setdefault(choice.index, [])
                if choice.delta is not None:
                    if choice.delta.content is not None:
                        content_parts[choice.index].append(choice.delta.content)
                    reasoning = reasoning_text(choice.delta)
                    if reasoning is not None:
                        reasoning_parts[choice.index].append(reasoning)
                if choice.finish_reason is not None:
                    finished.add(choice.index)

    if finished != set(range(expected_choices)):
        raise ConnectionError("Chat stream ended before every choice reported a finish reason")

    return Completion(
        responses=[
            {
                "content": "".join(content_parts[index]) if content_parts[index] else None,
                "reasoning": "".join(reasoning_parts[index]) if reasoning_parts[index] else None,
            }
            for index in range(expected_choices)
        ],
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


async def request_completion(client: AsyncOpenAI, messages, model, gen_args: Dict) -> Completion:
    response = await client.chat.completions.create(model=model, messages=messages, **gen_args)
    if gen_args.get("stream"):
        return await collect_stream(response, gen_args["n"])

    usage = response.usage
    return Completion(
        responses=[
            {"content": choice.message.content, "reasoning": reasoning_text(choice.message)}
            for choice in response.choices or []
        ],
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


async def run_model_by_message(client: AsyncOpenAI, messages, data_id, model, gen_args: Dict,
                               rate_limiter: RequestRateLimiter, timeout: int = 300):
    retry_num, break_flag = 0, False
    responses = None
    while not break_flag:
        try:
            await rate_limiter.acquire()
            responses = await asyncio.wait_for(
                request_completion(client, messages, model, gen_args),
                timeout=timeout
            )
            break_flag = True

        except (asyncio.TimeoutError, ConnectionError, APIConnectionError, RateLimitError, InternalServerError) as e:
            logger.error(f"Network exception occurred for {data_id}: {e}")

        except Exception as e:
            if (
                    "rate limit" in str(e).lower()
                    or "too many requests" in str(e).lower()
                    or "server error" in str(e).lower()
                    or "connection error" in str(e).lower()
            ):
                logger.error(f"Sever error occurred for {data_id}: {e}")
            else:
                logger.error(f"Other error occurred for {data_id}: {e}. skip it")
                break_flag = True

        if not break_flag:
            retry_num = retry_num + 1
            if retry_num > 10:
                logger.info(f"Max Retry reached for {data_id}, skip it")
                break_flag = True
            else:
                wait_time = min(2 ** retry_num, 10)
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

    return responses


async def process_item(client: AsyncOpenAI, doc, args, rate_limiter: RequestRateLimiter) -> Dict:
    messages = get_messages(doc)

    gen_args = {args.token_field: args.token_limit, "n": args.n_sampling}
    if args.stream:
        gen_args.update(stream=True, stream_options={"include_usage": True})

    completion = await run_model_by_message(
        client=client, messages=messages, data_id=doc["id"], model=args.model_id, gen_args=gen_args,
        rate_limiter=rate_limiter, timeout=args.timeout
    )
    responses = completion.responses if completion else []
    missing = max(0, args.n_sampling - len(responses))

    return dict(
        id=doc["id"],
        metadata=doc,
        request=json.dumps(messages, ensure_ascii=False),
        responses=responses + [{"content": None, "reasoning": None} for _ in range(missing)],
        total_input_tokens=completion.input_tokens if completion else 0,
        total_output_tokens=completion.output_tokens if completion else 0,
    )


async def process_with_semaphore(client, data_item, args, semaphore, rate_limiter):
    async with semaphore:
        return await process_item(client, data_item, args, rate_limiter)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument(
        '--data_name',
        type=str,
        default="KINA-899",
        help="Dataset JSON basename under data/ (without .json), e.g. KINA-899",
    )
    parser.add_argument('--n_sampling', type=int, default=1)
    token_options = parser.add_mutually_exclusive_group()
    token_options.add_argument('--max_tokens', type=int, help="Send max_tokens (default: 16384)")
    token_options.add_argument('--max_completion_tokens', type=int, help="Send max_completion_tokens instead")
    parser.add_argument('--stream', action='store_true', help="Stream and collect completions before scoring")
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip_inference', action='store_true')
    parser.add_argument("--n_thread", type=int, default=1)
    parser.add_argument("--requests_per_second", type=float, default=1.0,
                        help="Maximum request starts per second, including retries (default: 1)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for each API request (default: 300)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of data samples for debugging")
    args = parser.parse_args()
    args.token_field = "max_completion_tokens" if args.max_completion_tokens is not None else "max_tokens"
    args.token_limit = getattr(args, args.token_field)
    if args.token_limit is None:
        args.token_limit = 16384
    if args.n_sampling not in KINA_SAMPLE_COUNTS:
        parser.error(f"--n_sampling must be one of {list(KINA_SAMPLE_COUNTS)}")
    if min(args.token_limit, args.n_thread, args.timeout) <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("Token limit, --n_thread, --timeout and --limit must be positive")
    if not math.isfinite(args.requests_per_second) or args.requests_per_second <= 0:
        parser.error("--requests_per_second must be finite and positive")

    reformat_model_id = args.model_id.replace('/', '--')
    transport = "stream" if args.stream else "json"
    generation = f"n{args.n_sampling}_tokens{args.token_limit}_{args.token_field}_{transport}"
    if args.limit is not None:
        generation += f"_limit{args.limit}"

    result_save_path = join(
        PROJECT_ROOT, "results", reformat_model_id, generation,
        f"{args.data_name}.jsonl"
    )
    os.makedirs(dirname(result_save_path), exist_ok=True)

    finish_ids = []
    if not args.overwrite and os.path.exists(result_save_path):
        with open(result_save_path) as f:
            for line in f.readlines():
                if not line:
                    continue

                try:
                    data_dict = json.loads(line.strip())
                    if all(resp["content"] is not None for resp in data_dict["responses"]):
                        finish_ids.append(data_dict["id"])
                except json.JSONDecodeError:
                    continue

    if len(finish_ids) > 0:
        logger.info(f"Resume from {len(finish_ids)} finished samples from {result_save_path}")
    else:
        logger.info(f"Start from scratch")

    data = load_data(file_path=join(PROJECT_ROOT, "data", f"{args.data_name}.json"))
    if args.limit is not None:
        data = data[:args.limit]
    total_data_ids = [doc["id"] for doc in data]
    data = [doc for doc in data if doc["id"] not in finish_ids]
    if data and not args.skip_inference:
        load_dotenv(find_dotenv(usecwd=True), override=False)
        async with AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE"), api_key=os.environ.get("OPENAI_KEY"),
                               max_retries=0) as client:
            semaphore = asyncio.Semaphore(args.n_thread)
            rate_limiter = RequestRateLimiter(args.requests_per_second)
            tasks = [
                asyncio.create_task(process_with_semaphore(client, doc, args, semaphore, rate_limiter))
                for doc in data
            ]
            try:
                with open(result_save_path, "a" if finish_ids else "w", encoding="utf-8") as f:
                    for task in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Inference"):
                        f.write(json.dumps(await task, ensure_ascii=False) + "\n")
                        f.flush()
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    if not os.path.exists(result_save_path):
        logger.info("No result file found, exiting")
        return

    if os.path.getsize(result_save_path) == 0:
        logger.warning("No results generated, skipping post-processing")
        return

    with open(result_save_path) as f:
        finished_data = {}
        for line in f.readlines():
            if not line:
                continue
            try:
                doc = json.loads(line.strip())
                if any(resp["content"] is None and resp["reasoning"] is None for resp in doc["responses"]):
                    continue

                doc_id = doc["id"]
                finished_data[doc_id] = doc
            except json.JSONDecodeError:
                continue

        finished_data = list(finished_data.values())

    finished_ids = [doc["id"] for doc in finished_data]
    finished_ids.sort()
    if finished_ids != total_data_ids and not args.skip_inference:
        missing_ids = set(total_data_ids) - set(finished_ids)
        missing_ids = list(missing_ids)
        logger.warning(
            f"Cannot generate final .json file: {len(missing_ids)}/{len(total_data_ids)} samples failed.\n"
            f"First 5 Failed IDs: {missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}\n"
            f"Please re-run the same command to resume and retry failed samples.\n"
            f"If you want to evaluate completed samples first, add --skip_inference to your command."
        )
        return

    final_data = []
    for doc in tqdm(finished_data, desc="KINA scoring"):
        doc_id = doc["id"]
        gt = doc["metadata"]["ground_truth"]

        responses = [resp.get("content") or "" for resp in doc["responses"]]

        score_dict = judge_score(metadata_doc=doc["metadata"], responses=responses)

        final_data.append(
            dict(
                id=doc_id,
                score=score_dict["score"],
                extracted_predictions=score_dict["extracted_predictions"],
                gt=gt
            )
        )

    with open(result_save_path.replace(".jsonl", ".json"), "w") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    asyncio.run(main())
