import argparse
import asyncio
import json
import logging
import os
import random

from os.path import join, dirname
from typing import Dict

import openai
from openai import AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from kina_bench.config import PROJECT_ROOT
from dotenv import find_dotenv, load_dotenv

from kina_bench.utils import load_data, get_messages, judge_score, KINA_METRICS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


async def run_model_by_message(client: AsyncOpenAI, messages, data_id, model, gen_args: Dict, args, timeout: int = 300):
    retry_num, break_flag = 0, False
    responses = None
    while not break_flag:
        try:
            responses = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **gen_args
                ),
                timeout=timeout
            )
            break_flag = True

        except (asyncio.TimeoutError, ConnectionError) as e:
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


async def process_item(client: AsyncOpenAI, doc, args) -> Dict:
    messages = get_messages(doc)

    max_tokens = args.max_tokens
    if args.model_id in ["gpt-4o", "o1"]:
        max_tokens = min(args.max_tokens, 16384)
    if args.model_id in ["o3", "o1", "o4-mini", "gpt-5-mini", "gpt-5", "gpt-5-nano"]:
        gen_args = dict(max_completion_tokens=max_tokens)
    else:
        gen_args = dict(max_tokens=max_tokens)

    gen_args["n"] = args.n_sampling
    if args.reasoning_effort is not None:
        gen_args["reasoning_effort"] = args.reasoning_effort

    if args.think_mode == "think":
        gen_args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    elif args.think_mode == "nothink":
        gen_args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    responses = await run_model_by_message(
        client=client, messages=messages, data_id=doc["id"], model=args.model_id, gen_args=gen_args, args=args,
        timeout=args.timeout
    )

    return dict(
        id=doc["id"],
        metadata=doc,
        request=json.dumps(messages, ensure_ascii=False),
        responses=responses,
    )


async def process_with_semaphore(client, data_item, args, semaphore):
    async with semaphore:
        await asyncio.sleep(random.uniform(0.1, 0.5))
        return await process_item(client, data_item, args)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument(
        '--data_name',
        type=str,
        default="KINA-899",
        help="Dataset JSON basename under data/ (without .json), e.g. KINA-899",
    )
    parser.add_argument('--reasoning_effort', type=str, default=None)
    parser.add_argument('--think_mode', type=str, default="none", choices=["none", "think", "nothink"])
    parser.add_argument('--n_sampling', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=16384)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip_inference', action='store_true')
    parser.add_argument("--n_thread", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for each API request (default: 300)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of data samples for debugging")
    args = parser.parse_args()

    assert args.think_mode in ["none", "think", "nothink"], args.think_mode
    assert args.n_sampling in KINA_METRICS, (
        f"n_sampling={args.n_sampling} is not supported. "
        f"Supported values are: {list(KINA_METRICS.keys())}."
    )

    if "gpt-5" in args.model_id:
        assert args.think_mode == "none", "--think_mode cannot be set for gpt-5 series"
        if args.reasoning_effort is None:
            logger.info(f"reasoning_effort is default to be medium for {args.model_id}")
            args.reasoning_effort = "medium"

    reformat_model_id = args.model_id.replace('/', '--')
    if args.reasoning_effort is not None:
        reformat_model_id += f"-{args.reasoning_effort}"

    if args.think_mode != "none":
        reformat_model_id += f"-{args.think_mode}"

    result_save_path = join(
        PROJECT_ROOT, "results", reformat_model_id, f"n{args.n_sampling}_tokens{args.max_tokens}",
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

    env_ok = load_dotenv(find_dotenv(usecwd=True), override=False)
    if not env_ok:
        logger.warning("No .env found; using existing environment only.")
    else:
        logger.info("Successfully load .env!")

    openai_base_url = os.environ.get("OPENAI_BASE")
    openai_api_key = os.environ.get("OPENAI_KEY")
    client = AsyncOpenAI(base_url=openai_base_url, api_key=openai_api_key)
    logger.info(f"AsyncOpenAI client created by base_url={openai_base_url} and api_key={openai_api_key[:10]}...{openai_api_key[-10:]}")

    # Poll until the inference server is ready.
    # This is useful when running a local server (e.g., SGLang, vLLM) that takes time
    # to initialize, especially with large models or DeepGEMM warmup.
    # For commercial APIs (OpenAI, OpenRouter, etc.), this check passes immediately.
    max_wait, interval = 1200, 10
    for elapsed in range(0, max_wait, interval):
        try:
            page = await client.models.list()
            logger.info(f"Server is ready (waited {elapsed}s)")
            break
        except (openai.APIConnectionError, ConnectionError):
            logger.info(f"Server not ready yet, retrying in {interval}s... ({elapsed}/{max_wait}s)")
            await asyncio.sleep(interval)
    else:
        raise RuntimeError(f"Server did not become ready within {max_wait}s")

    data = load_data(file_path=join(PROJECT_ROOT, "data", f"{args.data_name}.json"))
    total_data_ids = [doc["id"] for doc in data]
    total_data_ids.sort()

    data = [doc for doc in data if doc["id"] not in finish_ids]
    if args.limit is not None:
        remain = max(0, args.limit - len(finish_ids))
        data = data[:remain]
        logger.info(f"Debug mode: limiting to {args.limit} total samples ({len(finish_ids)} finished, {len(data)} remaining)")
    if len(data) > 0 and not args.skip_inference:
        try:
            tasks = []
            semaphore = asyncio.Semaphore(args.n_thread)
            for data_item in data:
                tasks.append(
                    process_with_semaphore(
                        client=client,
                        data_item=data_item,
                        args=args,
                        semaphore=semaphore
                    )
                )

            with open(result_save_path, "w" if len(finish_ids) == 0 else "a") as f:
                for task in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Inference"):
                    ret_dict = await task

                    dump_dict = {
                        "id": ret_dict["id"],
                        "request": ret_dict["request"],
                        "responses": [],
                        "metadata": ret_dict["metadata"],
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                    }

                    if ret_dict["responses"] is not None and getattr(ret_dict["responses"], "choices", None) is not None:
                        for single_response in ret_dict["responses"].choices:
                            try:
                                msg_content = single_response.message.content
                                msg_reasoning = None
                                for key in ["reasoning", "reasoning_content"]:
                                    if hasattr(single_response.message, key):
                                        msg_reasoning = getattr(single_response.message, key)
                                        break
                            except AttributeError:
                                msg_content = None
                                msg_reasoning = None

                            dump_dict["responses"].append(dict(content=msg_content, reasoning=msg_reasoning))

                        dump_dict["total_input_tokens"] += ret_dict["responses"].usage.prompt_tokens
                        dump_dict["total_output_tokens"] += ret_dict["responses"].usage.completion_tokens

                        if len(dump_dict["responses"]) < args.n_sampling:
                            remain_num = args.n_sampling - len(dump_dict["responses"])
                            dump_dict["responses"].extend([dict(content=None, reasoning=None)] * remain_num)

                    else:
                        dump_dict["responses"] = [dict(content=None, reasoning=None)] * args.n_sampling

                    f.write(json.dumps(dump_dict, ensure_ascii=False) + "\n")
                    f.flush()

        finally:
            if client and hasattr(client, 'close'):
                await client.close()

            for task in asyncio.all_tasks(asyncio.get_running_loop()):
                if task is not asyncio.current_task() and not task.done():
                    task.cancel()

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
    for doc in tqdm(finished_data, desc="lighteval evaluation"):
        doc_id = doc["id"]
        gt = doc["metadata"]["ground_truth"]

        responses = []
        for resp in doc["responses"]:
            if resp.get("content") is not None:
                responses.append(resp["content"])
            else:
                responses.append(resp["reasoning"])

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
