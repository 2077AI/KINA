import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from openai import AsyncOpenAI

from kina_bench import run_openai_chat as runner
from kina_bench.utils import judge_score


class FinalAnswerScoringTest(unittest.TestCase):
    def test_only_explicit_final_choices_count(self):
        doc = {"options": {"A": {}, "B": {}}, "ground_truth": "A"}
        result = judge_score(doc, [
            "A valid answer letter cannot be assigned.\n\nAnswer: None of A–J",
            "Answer: A\n\nNo listed option is valid.",
            "**Answer: A**\n",
            "Answer: B",
        ])
        self.assertEqual(result, {"score": 0.25, "extracted_predictions": [[], [], ["A"], ["B"]]})

    def test_answer_must_be_one_listed_choice(self):
        doc = {"options": {"A": {}, "B": {}}, "ground_truth": "A"}
        for response in ("", "Answer: A or B", "Answer: C", "Answer: No valid option"):
            with self.subTest(response=response):
                self.assertEqual(judge_score(doc, [response]), {"score": 0.0, "extracted_predictions": [[]]})
        self.assertEqual(judge_score(doc, ["Answer: $A$"])["score"], 1.0)


class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, request))
        if request.get("stream"):
            count = request["n"]
            chunks = [
                {"choices": [
                    {"index": index, "delta": {"content": "Answer: "}, "finish_reason": None}
                    for index in reversed(range(count))
                ]},
                {"choices": [
                    {"index": index, "delta": {"content": "A" if index == 0 else "B"},
                     "finish_reason": "stop"}
                    for index in range(count)
                    if request["model"] != "incomplete-stream"
                ]},
                {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}},
            ]
            for chunk in chunks:
                chunk.update(id="fake", object="chat.completion.chunk", created=1, model=request["model"])
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            content_type = "text/event-stream"
        else:
            body = json.dumps({
                "id": "fake", "object": "chat.completion", "created": 1, "model": request["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer: A"},
                             "finish_reason": "stop"}],
            })
            content_type = "application/json"
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        if self.server.stall_first_stream and len(self.server.requests) == 1:
            first_event = encoded.split(b"\n\n", 1)[0] + b"\n\n"
            self.wfile.write(first_event)
            self.wfile.flush()
            self.server.release_response.wait(5)
            return
        self.wfile.write(encoded)

    def log_message(self, *_):
        pass


class ChatInferenceTest(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
        self.server.requests = []
        self.server.stall_first_stream = False
        self.server.release_response = threading.Event()
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.release_response.set)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        data_dir = self.root / "data"
        data_dir.mkdir()
        questions = [
            {
                "index": index,
                "question": "What is two plus two?",
                "options": [{"key": "A", "answer": "4"}, {"key": "B", "answer": "5"}],
                "correct_answer": "A",
            }
            for index in range(2)
        ]
        (data_dir / "smoke.json").write_text(json.dumps(questions), encoding="utf-8")

    def run_cli(self, *options, with_key=True):
        argv = ["kina", "--model_id", "gpt-5-example", "--data_name", "smoke", "--limit", "1", *options]
        env = {"OPENAI_BASE": self.base_url}
        if with_key:
            env["OPENAI_KEY"] = "local-test-key"
        with patch.object(runner, "PROJECT_ROOT", str(self.root)), patch.object(sys, "argv", argv):
            with patch.dict(os.environ, env, clear=True), patch.object(runner, "load_dotenv"):
                asyncio.run(runner.main())

    def test_json_defaults_and_offline_scoring(self):
        self.run_cli()
        [(path, request)] = self.server.requests
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(request["model"], "gpt-5-example")
        self.assertEqual(request["max_tokens"], 16384)
        self.assertTrue({"max_completion_tokens", "reasoning_effort", "temperature", "top_p", "stream"}.isdisjoint(request))
        [raw] = self.root.glob("results/*/*/smoke.jsonl")
        self.assertEqual(json.loads(raw.read_text())["total_output_tokens"], 0)
        [scores] = self.root.glob("results/*/*/smoke.json")
        self.assertEqual([item["score"] for item in json.loads(scores.read_text())], [1.0])
        self.run_cli("--skip_inference", with_key=False)
        self.assertEqual(len(self.server.requests), 1)

    def test_stream_preserves_choice_order_and_usage(self):
        self.run_cli("--stream", "--max_completion_tokens", "321", "--timeout", "5", "--n_sampling", "4")
        [(_, request)] = self.server.requests
        self.assertTrue(request["stream"])
        self.assertEqual(request["stream_options"], {"include_usage": True})
        self.assertEqual(request["max_completion_tokens"], 321)
        self.assertNotIn("max_tokens", request)
        [raw] = self.root.glob("results/*/*/smoke.jsonl")
        row = json.loads(raw.read_text())
        self.assertEqual([response["content"] for response in row["responses"]], ["Answer: A", "Answer: B", "Answer: B", "Answer: B"])
        self.assertEqual((row["total_input_tokens"], row["total_output_tokens"]), (10, 4))
        [scores] = self.root.glob("results/*/*/smoke.json")
        self.assertAlmostEqual(json.loads(scores.read_text())[0]["score"], 0.25)

    def test_default_request_rate_spaces_starts_at_high_concurrency(self):
        starts = []
        request_completion = runner.request_completion

        async def record_start(*args, **kwargs):
            starts.append(time.monotonic())
            return await request_completion(*args, **kwargs)

        with patch.object(runner, "request_completion", record_start):
            self.run_cli("--limit", "2", "--n_thread", "20")
        first, second = starts
        self.assertGreaterEqual(second - first, 0.99)
        self.assertEqual(len(self.server.requests), 2)

    def test_incomplete_stream_is_rejected(self):
        async def request():
            async with AsyncOpenAI(base_url=self.base_url, api_key="local-test-key", max_retries=0) as client:
                await runner.request_completion(
                    client, [{"role": "user", "content": "Synthetic question"}],
                    "incomplete-stream", {"stream": True, "n": 1},
                )

        with self.assertRaisesRegex(ConnectionError, "finish reason"):
            asyncio.run(request())

    def test_timeout_covers_reading_the_stream(self):
        self.server.stall_first_stream = True
        self.run_cli("--stream", "--timeout", "1")
        self.assertEqual(len(self.server.requests), 2)
        [scores] = self.root.glob("results/*/*/smoke.json")
        self.assertEqual(json.loads(scores.read_text())[0]["score"], 1.0)

    def test_token_options_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit) as error:
            self.run_cli("--max_tokens", "100", "--max_completion_tokens", "100")
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.server.requests, [])
