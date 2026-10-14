import hashlib
import http.client
import json
import math
import unittest
from pathlib import Path
from unittest import mock

from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    processors,
)
from transformers import PreTrainedTokenizerFast

from dev.tests.server_fixtures import (
    FOREVER,
    CharImageTokenizer,
    CharTokenizer,
    FakeRuntime,
    HarnessTestCase,
    Plan,
    byte_alphabet,
    judgment_body,
    make_frontend,
    png_data_url,
    stays_connected,
)
from server import judgments
from server import server as api

TURNS_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message.role }}\n"
    "{{ message.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def byte_tokenizer(template):
    """A byte-level BPE tokenizer for `template` with Qwen's turn tokens, a
    token per byte between them. Like a model's, it refuses text that holds a
    lone surrogate, and a prompt splits after its last <|im_end|>."""
    vocabulary = {char: byte for byte, char in byte_alphabet().items()}
    backend = Tokenizer(models.BPE(vocabulary, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.ByteLevel(trim_offsets=False)
    backend.add_special_tokens(
        [AddedToken(text, normalized=False) for text in ("<|im_start|>", "<|im_end|>")]
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.chat_template = template
    return tokenizer


class BoundaryCountingTokenizer(CharTokenizer):
    """Counts whole-prompt tokenizations: one to prepare the prompt plus
    one per answer-slot boundary check. An optional clock advances on each
    of them so a test can expire a deadline inside the boundary pass."""

    def __init__(self, clock=None, step=0.0):
        super().__init__()
        self.clock = clock
        self.step = step
        self.prompt = None
        self.prompt_encodes = 0

    def apply_chat_template(self, messages, **kwargs):
        rendered = super().apply_chat_template(messages, **kwargs)
        if kwargs.get("tokenize") is False:
            self.prompt = rendered
        return rendered

    def encode(self, text, **kwargs):
        if self.prompt is not None and text.startswith(self.prompt):
            self.prompt_encodes += 1
            if self.clock is not None:
                self.clock[0] += self.step
        return super().encode(text, **kwargs)


class JudgmentTests(HarnessTestCase):
    def test_judgments_scores_options_without_generation(self):
        runtime = FakeRuntime(Plan(logits=(1.5, -2.25)))
        harness = self.harness(runtime, tokenizer=CharTokenizer(), max_context=8192)
        status, content_type, payload = harness.request(
            "POST", "/v1/judgments", judgment_body()
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(content_type, "application/json")
        response = json.loads(payload)
        self.assertEqual(response["id"], "row-1")
        self.assertEqual(response["option_ids"], ["yes", "no"])
        self.assertEqual(response["option_logits"], [1.5, -2.25])
        probabilities = response["probabilities"]
        self.assertEqual(len(probabilities), 2)
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[0], probabilities[1])
        expected = math.exp(1.5) / (math.exp(1.5) + math.exp(-2.25))
        self.assertAlmostEqual(probabilities[0], expected)
        self.assertEqual(response["prompt_version"], "direct-options-v1")
        self.assertEqual(
            response["usage"],
            {
                "prompt_tokens": response["input_tokens"],
                "completion_tokens": 0,
                "total_tokens": response["input_tokens"],
            },
        )

        request = runtime.requests[0].frame
        # The rendered prompt plus one slot token is exactly the boundary the
        # engine scores; verify it against the tokenizer, not the response.
        prompt_text = harness.tokenizer.decode(request.prompt_tokens)
        self.assertEqual(
            response["prompt_sha256"],
            hashlib.sha256(prompt_text.encode()).hexdigest(),
        )
        self.assertEqual(response["answer_token_ids"], list(request.score_tokens))
        self.assertEqual(
            harness.tokenizer.encode(prompt_text + "A"),
            list(request.prompt_tokens) + [ord("A")],
        )

    def test_judgments_send_the_cache_salt(self):
        # Rows of one salt share cached prompts with each other alone.
        runtime = FakeRuntime(*(Plan(logits=(1.5, -2.25)) for _ in range(4)))
        harness = self.harness(runtime, tokenizer=CharTokenizer(), max_context=8192)
        for salt in ("run-1", "run-1", "run-2", None):
            body = judgment_body() if salt is None else judgment_body(cache_salt=salt)
            status, _, payload = harness.request("POST", "/v1/judgments", body)
            self.assertEqual(status, 200, payload)
        first, again, other, unsalted = (
            request.frame.cache_salt for request in runtime.requests
        )
        self.assertNotEqual(first, 0)
        self.assertEqual(first, again)
        self.assertNotIn(other, (0, first))
        self.assertEqual(unsalted, 0)
        for salt in ("", 7, ["run-1"]):
            status, _, payload = harness.request(
                "POST", "/v1/judgments", judgment_body(cache_salt=salt)
            )
            self.assertEqual(status, 400, payload)
            self.assertIn("cache_salt must be a non-empty string", payload.decode())
        self.assertEqual(len(runtime.requests), 4)

    def test_judgments_rejects_invalid_rows_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=CharTokenizer(), max_context=8192)
        cases = (
            {},
            judgment_body(id=""),
            judgment_body(state={}),
            judgment_body(question=42),
            judgment_body(options=[{"id": "a", "description": "x"}]),
            judgment_body(
                options=[
                    {"id": "a", "description": "x"},
                    {"id": "a", "description": "y"},
                ]
            ),
            judgment_body(
                options=[
                    {"id": "a", "description": "x"},
                    {"id": "b"},
                ]
            ),
            judgment_body(
                options=[{"id": str(i), "description": "x"} for i in range(17)]
            ),
            judgment_body(model="other-model"),
            judgment_body(stream=True),
            judgment_body(priority="urgent"),
        )
        for body in cases:
            with self.subTest(body=body):
                status, _, payload = harness.request("POST", "/v1/judgments", body)
                self.assertIn(status, (400, 404), payload)
        self.assertEqual(runtime.requests, [])

    def test_judgments_rejects_nonfinite_state(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=CharTokenizer(), max_context=8192)
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=3
        )
        connection.request(
            "POST",
            "/v1/judgments",
            '{"id":"r","state":NaN,"question":"q",'
            '"options":[{"id":"a","description":"x"},'
            '{"id":"b","description":"y"}]}',
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 400)
        connection.close()
        self.assertEqual(runtime.requests, [])

    def test_scoring_reads_a_lone_surrogate_as_a_replacement_character(self):
        # JSON spells a lone surrogate, such as half of an emoji a client cut,
        # which the tokenizer refuses. A scoring prompt reads it as U+FFFD, as
        # any request's does, whether the prompt's end encodes on its own or
        # the whole prompt does: a judgment row with one answers as the row
        # with U+FFFD in its place, with the digest of that text, and so does
        # a System One question.
        for template in (TURNS_TEMPLATE, TURNS_TEMPLATE.replace("<|im_end|>", "")):
            split = "<|im_end|>" in template
            with self.subTest(split=split):
                runtime = FakeRuntime(*(Plan(logits=(1.5, -2.25)) for _ in range(3)))
                harness = self.harness(
                    runtime, tokenizer=byte_tokenizer(template), max_context=8192
                )
                digests = []
                for state in ("half an emoji: \ud83d", "half an emoji: \ufffd"):
                    status, _, payload = harness.request(
                        "POST", "/v1/judgments", judgment_body(state=state)
                    )
                    self.assertEqual(status, 200, payload)
                    digests.append(json.loads(payload)["prompt_sha256"])
                tokens = [request.frame.prompt_tokens for request in runtime.requests]
                self.assertEqual(tokens[0], tokens[1])
                prompt = harness.tokenizer.decode(tokens[0])
                self.assertIn("half an emoji: \ufffd", prompt)
                digest = hashlib.sha256(prompt.encode()).hexdigest()
                self.assertEqual(digests, [digest, digest])
                prompt_tokenizer = harness.app.prompt_tokenizer
                self.assertEqual(bool(prompt_tokenizer.split(prompt)[0]), split)
                status, _, payload = harness.request(
                    "POST",
                    "/v1/systemone",
                    {
                        "model": "test-model",
                        "state": "half an emoji: \ud83d",
                        "questions": {
                            "q": {
                                "type": "choice",
                                "criteria": {"yes": "supported", "no": "not supported"},
                            }
                        },
                    },
                )
                self.assertEqual(status, 200, payload)

    def test_judgments_requires_auth_when_configured(self):
        runtime = FakeRuntime(Plan(logits=(0.0, 1.0)))
        harness = self.harness(
            runtime,
            tokenizer=CharTokenizer(),
            max_context=8192,
            api_key="secret",
        )
        status, _, _ = harness.request("POST", "/v1/judgments", judgment_body())
        self.assertEqual(status, 401)
        status, _, payload = harness.request(
            "POST",
            "/v1/judgments",
            judgment_body(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer secret",
            },
        )
        self.assertEqual(status, 200, payload)

    def test_judgments_deadline_cancels_the_score_request(self):
        plan = Plan(logits=(0.0, 1.0), block=True)
        runtime = FakeRuntime(plan)
        harness = self.harness(
            runtime,
            tokenizer=CharTokenizer(),
            max_context=8192,
            timeout=0.05,
        )
        status, _, payload = harness.request("POST", "/v1/judgments", judgment_body())
        self.assertEqual(status, 504, payload)
        self.assertEqual(runtime.cancel_count, 1)

    def test_judgment_deadline_stops_the_slot_boundary_pass(self):
        clock = [100.0]
        tokenizer = BoundaryCountingTokenizer(clock, 0.5)
        app = make_frontend(tokenizer, None, "test-model", 8192, 10, 2, vision=True)
        body = judgment_body(
            options=[
                {"id": f"opt{index}", "description": f"case {index}"}
                for index in range(16)
            ],
            timeout=1.0,
        )
        deadline = app.request_deadline(body, clock[0])
        with (
            mock.patch.object(api.time, "monotonic", side_effect=lambda: clock[0]),
            self.assertRaises(api.APIError) as error,
        ):
            app.prepare_judgment(body, deadline=deadline, disconnected=stays_connected)
        self.assertEqual(
            (error.exception.status, error.exception.code), (504, "request_timeout")
        )
        # The prepare pass and one boundary check; all 17 ran before the fix.
        self.assertEqual(tokenizer.prompt_encodes, 2)

    def test_client_disconnect_stops_the_slot_boundary_pass(self):
        tokenizer = BoundaryCountingTokenizer()
        app = make_frontend(tokenizer, None, "test-model", 8192, 10, 2, vision=True)
        body = judgment_body(
            options=[
                {"id": f"opt{index}", "description": f"case {index}"}
                for index in range(16)
            ]
        )
        # Connected through admission and the first boundary check.
        disconnected = iter((False, False, True))
        with self.assertRaises(ConnectionResetError):
            app.prepare_judgment(
                body, deadline=FOREVER, disconnected=lambda: next(disconnected)
            )
        # The prepare pass and one boundary check, and the slot returned.
        self.assertEqual(tokenizer.prompt_encodes, 2)
        self.assertEqual(app.preparation_active, 0)

    def test_judgment_context_budget_precedes_the_slot_boundary_pass(self):
        tokenizer = BoundaryCountingTokenizer()
        app = make_frontend(tokenizer, None, "test-model", 8, 10, 2, vision=True)
        with self.assertRaises(api.APIError) as error:
            app.prepare_judgment(
                judgment_body(
                    options=[
                        {"id": f"opt{index}", "description": f"case {index}"}
                        for index in range(16)
                    ]
                ),
                deadline=FOREVER,
                disconnected=stays_connected,
            )
        self.assertEqual(
            (error.exception.status, error.exception.code),
            (400, "context_length_exceeded"),
        )
        # Only the prepare pass; the 16 boundary checks ran first before the fix.
        self.assertEqual(tokenizer.prompt_encodes, 1)

    def test_systemone_validates_all_questions_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=CharTokenizer(), max_context=8192)
        for invalid in (
            {"type": []},
            {"type": "noul", "criteria": ["yes"]},
            {"type": "score", "criteria": [None]},
            {"type": "choice", "criteria": {str(i): None for i in range(256)}},
        ):
            with self.subTest(question=invalid):
                status, _, payload = harness.request(
                    "POST",
                    "/v1/systemone",
                    {
                        "model": "test-model",
                        "state": {},
                        "questions": {
                            "valid": {"type": "noul", "criteria": {}},
                            "invalid": invalid,
                        },
                    },
                )
                self.assertEqual(status, 422, payload)
                self.assertTrue(
                    any(
                        "invalid" in error["loc"]
                        for error in json.loads(payload)["detail"]
                    )
                )
        self.assertEqual(runtime.requests, [])

    def test_systemone_singleton_domains_need_no_native_request(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=CharTokenizer())
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": "test-model",
                "state": [],
                "questions": {
                    "choice": {"type": "choice", "criteria": {"only": None}},
                    "score": {
                        "type": "score",
                        "criteria": [{"description": "Only level"}],
                    },
                },
            },
        )
        self.assertEqual(status, 200, payload)
        response = json.loads(payload)
        self.assertEqual(response["answers"]["choice"]["probabilities"], {"only": 1.0})
        self.assertEqual(response["answers"]["score"]["score"], 0.0)
        self.assertEqual(response["usage"], {"input_tokens": 0, "output_tokens": 0})
        self.assertEqual(runtime.requests, [])

    def test_systemone_render_failure_is_a_request_error(self):
        class QuestionRenderFails(CharTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                if messages[0].get("content") == judgments.SYSTEMONE_SYSTEM:
                    raise ValueError("the template cannot render this question")
                return super().apply_chat_template(messages, **kwargs)

        runtime = FakeRuntime()
        harness = self.harness(
            runtime, tokenizer=QuestionRenderFails(), max_context=8192
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": "test-model",
                "state": "Some evidence",
                "questions": {"supported": {"type": "noul"}},
            },
        )
        self.assertEqual(status, 400, payload)
        self.assertEqual(
            json.loads(payload)["error"]["message"],
            "question prompt could not be rendered",
        )
        self.assertEqual(runtime.requests, [])

    def test_systemone_shared_deadline_cancels_only_current_question(self):
        blocked = Plan(logits=(0.0, 1.0), block=True)
        runtime = FakeRuntime(Plan(logits=(1.0, 0.0)), blocked)
        harness = self.harness(
            runtime,
            tokenizer=CharTokenizer(),
            max_context=8192,
            queue_size=1,
            timeout=1,
        )
        judgments.slot_labels(harness.tokenizer)
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": "test-model",
                "state": "Some evidence",
                "questions": {
                    "first": {"type": "noul", "criteria": {}},
                    "blocked": {"type": "noul"},
                    "never_started": {"type": "noul"},
                },
            },
        )
        self.assertEqual(status, 504, payload)
        self.assertEqual(len(runtime.requests), 2)
        self.assertTrue(blocked.cancelled.is_set())
        self.assertEqual(runtime.cancel_count, 1)

    def test_systemone_sends_the_cache_salt(self):
        runtime = FakeRuntime(Plan(logits=(1.0, 0.0)), Plan(logits=(1.0, 0.0)))
        harness = self.harness(runtime, tokenizer=CharTokenizer(), max_context=8192)
        body = {
            "model": "test-model",
            "state": "Some evidence",
            "questions": {"first": {"type": "noul", "criteria": {}}},
        }
        for salt in ("run-1", None):
            request = body if salt is None else body | {"cache_salt": salt}
            status, _, payload = harness.request("POST", "/v1/systemone", request)
            self.assertEqual(status, 200, payload)
        salted, unsalted = (request.frame.cache_salt for request in runtime.requests)
        self.assertNotEqual(salted, 0)
        self.assertEqual(unsalted, 0)
        status, _, payload = harness.request(
            "POST", "/v1/systemone", body | {"cache_salt": ""}
        )
        self.assertEqual(status, 422, payload)
        self.assertEqual(
            [error["loc"] for error in json.loads(payload)["detail"]],
            [["body", "cache_salt"]],
        )
        self.assertEqual(len(runtime.requests), 2)

    def test_systemone_question_count_budget_rejects_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=CharTokenizer(), max_context=8192)
        fitting = {
            f"q{index}": {"type": "choice", "criteria": {"only": None}}
            for index in range(judgments.MAX_SYSTEMONE_QUESTIONS)
        }
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {"model": "test-model", "state": [], "questions": fitting},
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            len(json.loads(payload)["answers"]), judgments.MAX_SYSTEMONE_QUESTIONS
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": "test-model",
                "state": [],
                "questions": {**fitting, "extra": {"type": "noul"}},
            },
        )
        self.assertEqual(status, 422, payload)
        self.assertIn("questions", json.loads(payload)["detail"][0]["loc"])
        self.assertEqual(runtime.requests, [])

    def test_systemone_total_token_budget_rejects_before_inference(self):
        class PaddedPromptTokenizer(BoundaryCountingTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                self.templates.append((messages, kwargs))
                self.prompt = "p" * 600_000
                return self.prompt

        runtime = FakeRuntime()
        tokenizer = PaddedPromptTokenizer()
        harness = self.harness(runtime, tokenizer=tokenizer, max_context=700_000)
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": "test-model",
                "state": "evidence",
                "questions": {"a": {"type": "noul"}, "b": {"type": "noul"}},
            },
        )
        self.assertEqual(status, 422, payload)
        self.assertIn("total prepared", json.loads(payload)["detail"][0]["msg"])
        self.assertEqual(runtime.requests, [])
        # Question a prepares and checks both slot boundaries; question b is
        # rejected on its prepare pass. Before the fix b also ran both checks.
        self.assertEqual(tokenizer.prompt_encodes, 4)

    def systemone_image_harness(self, runtime, **options):
        harness = self.harness(
            runtime, tokenizer=CharImageTokenizer(), max_context=8192, **options
        )
        judgments.slot_labels(harness.tokenizer)
        return harness

    def test_systemone_images_reach_every_scored_question(self):
        runtime = FakeRuntime(Plan(logits=(3.0, 1.0)), Plan(logits=(1.0, 3.0)))
        harness = self.systemone_image_harness(runtime)
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": "test-model",
                "state": "Is the square red?",
                "images": [png_data_url()],
                "cache_salt": "run-1",
                "questions": {"red": {"type": "noul"}, "blue": {"type": "noul"}},
            },
        )
        self.assertEqual(status, 200, payload)
        answers = json.loads(payload)["answers"]
        self.assertGreater(answers["red"]["noul"], 0.5)
        self.assertLess(answers["blue"]["noul"], 0.5)
        self.assertEqual(len(runtime.requests), 2)
        pad = CharImageTokenizer.PAD_ID
        for request in (call.frame for call in runtime.requests):
            (span,) = request.image_spans
            self.assertGreater(span.tokens, 1)
            self.assertEqual(
                list(request.prompt_tokens[span.offset : span.offset + span.tokens]),
                [pad] * span.tokens,
            )
            self.assertEqual(list(request.prompt_tokens).count(pad), span.tokens)
            self.assertTrue(request.image_pixels)
            self.assertNotEqual(request.cache_salt, 0)

    def test_systemone_images_precede_the_evidence(self):
        runtime = FakeRuntime(Plan(logits=(1.0, 0.0)))
        harness = self.systemone_image_harness(runtime)
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": "test-model",
                "state": "evidence",
                "images": [png_data_url()],
                "questions": {"q": {"type": "noul"}},
            },
        )
        self.assertEqual(status, 200, payload)
        user = harness.tokenizer.templates[-1][0][1]["content"]
        self.assertEqual([part["type"] for part in user], ["image_url", "text"])
        self.assertIn("evidence", user[1]["text"])

    def test_systemone_without_images_renders_the_stock_prompt(self):
        runtime = FakeRuntime(Plan(logits=(1.0, 0.0)))
        harness = self.systemone_image_harness(runtime)
        body = {
            "model": "test-model",
            "state": "evidence",
            "questions": {"q": {"type": "noul"}},
        }
        status, _, payload = harness.request("POST", "/v1/systemone", body)
        self.assertEqual(status, 200, payload)
        (call,) = runtime.requests
        state, ((_, spec),), _ = judgments.validate_systemone(body)
        labels = judgments.slot_labels(harness.tokenizer)[: len(spec.labels)]
        messages = judgments.systemone_messages(state, spec, labels)
        self.assertIsInstance(messages[1]["content"], str)
        tokens, _, _ = judgments.encode_prompt(
            harness.app.prompt_tokenizer,
            harness.app.chat_templates.select(None).source,
            messages,
            labels,
            admit=lambda _: None,
            checkpoint=lambda: None,
        )
        self.assertEqual(list(call.frame.prompt_tokens), tokens)
        self.assertEqual(call.frame.image_spans, ())
        self.assertEqual(call.frame.image_pixels, b"")

    def test_systemone_images_rejected_before_inference(self):
        cases = (
            ({"vision": False}, [png_data_url()], "--language-only"),
            ({}, [], "data: URLs"),
            ({}, "data:image/png;base64,AAAA", "data: URLs"),
            ({}, ["https://example.com/a.png"], "data: URLs"),
            ({}, [png_data_url()] * 65, "data: URLs"),
        )
        for options, urls, message in cases:
            with self.subTest(options=options, images=str(urls)[:40]):
                runtime = FakeRuntime()
                harness = self.systemone_image_harness(runtime, **options)
                status, _, payload = harness.request(
                    "POST",
                    "/v1/systemone",
                    {
                        "model": "test-model",
                        "state": "evidence",
                        "images": urls,
                        "questions": {"q": {"type": "noul"}},
                    },
                )
                self.assertEqual(status, 422, payload)
                (error,) = json.loads(payload)["detail"]
                self.assertEqual(error["loc"], ["body", "images"])
                self.assertIn(message, error["msg"])
                self.assertEqual(runtime.requests, [])

    def test_systemone_client_disconnect_stops_image_decoding(self):
        app = make_frontend(
            CharImageTokenizer(), None, "test-model", 8192, 10, 2, vision=True
        )
        body = {
            "model": "test-model",
            "state": "evidence",
            "images": [png_data_url(), png_data_url((30, 30, 200))],
            "questions": {"q": {"type": "noul"}},
        }
        # Connected through the first image.
        polls = iter((False, True))
        prepared = []
        decode = app.images.prepare
        with (
            mock.patch.object(
                app.images,
                "prepare",
                side_effect=lambda *args: (
                    prepared.append(decode(*args)) or prepared[-1]
                ),
            ),
            self.assertRaises(ConnectionResetError),
        ):
            app.prepare_systemone(
                body, deadline=FOREVER, disconnected=lambda: next(polls)
            )
        self.assertEqual(len(prepared), 1)
        self.assertEqual(app.preparation_active, 0)

    def test_systemone_context_limit_counts_image_tokens(self):
        body = {
            "model": "test-model",
            "state": "evidence",
            "questions": {"q": {"type": "noul"}},
        }
        runtime = FakeRuntime(Plan(logits=(1.0, 0.0)))
        harness = self.systemone_image_harness(runtime)
        status, _, payload = harness.request("POST", "/v1/systemone", body)
        self.assertEqual(status, 200, payload)
        text_tokens = len(runtime.requests[0].frame.prompt_tokens)

        runtime = FakeRuntime()
        harness = self.harness(
            runtime,
            tokenizer=CharImageTokenizer(),
            max_context=text_tokens + 16,
        )
        body["images"] = [png_data_url()]
        status, _, payload = harness.request("POST", "/v1/systemone", body)
        self.assertEqual(status, 400, payload)
        self.assertEqual(
            json.loads(payload)["error"]["code"], "context_length_exceeded"
        )
        self.assertEqual(runtime.requests, [])

    def test_systemone_questions_share_one_copy_of_the_pixels(self):
        runtime = FakeRuntime(Plan(logits=(1.0, 0.0)), Plan(logits=(1.0, 0.0)))
        harness = self.systemone_image_harness(runtime)
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": "test-model",
                "state": "evidence",
                "images": [png_data_url((200, 30, 30)), png_data_url((30, 30, 200))],
                "questions": {"a": {"type": "noul"}, "b": {"type": "noul"}},
            },
        )
        self.assertEqual(status, 200, payload)
        first, second = (call.frame for call in runtime.requests)
        self.assertEqual(len(first.image_spans), 2)
        self.assertIs(first.image_pixels, second.image_pixels)

    def test_systemone_total_token_budget_counts_image_tokens(self):
        body = {
            "model": "test-model",
            "state": "evidence",
            "questions": {"q": {"type": "noul"}},
        }
        runtime = FakeRuntime(Plan(logits=(1.0, 0.0)))
        harness = self.systemone_image_harness(runtime)
        status, _, payload = harness.request("POST", "/v1/systemone", body)
        self.assertEqual(status, 200, payload)
        text_tokens = len(runtime.requests[0].frame.prompt_tokens)

        runtime = FakeRuntime()
        harness = self.systemone_image_harness(runtime)
        body["images"] = [png_data_url()]
        with mock.patch.object(
            judgments, "MAX_SYSTEMONE_TOTAL_TOKENS", text_tokens + 1
        ):
            status, _, payload = harness.request("POST", "/v1/systemone", body)
        self.assertEqual(status, 422, payload)
        self.assertIn("total prepared", json.loads(payload)["detail"][0]["msg"])
        self.assertEqual(runtime.requests, [])


# Umpire's tokenizer and Qwen chat template, from the local Hub cache; the
# test below needs the files, not the weights.
UMPIRE_FILES = sorted(
    Path.home().glob(
        ".cache/huggingface/hub/models--ezoushen--umpire-35b-a3b-mlx-4bit"
        "/snapshots/*/tokenizer.json"
    )
)


@unittest.skipUnless(
    UMPIRE_FILES, "Umpire's tokenizer files are not in the local Hub cache"
)
class RealTemplateTests(unittest.TestCase):
    def test_systemone_image_prompt_agrees_with_encode_prompt(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            UMPIRE_FILES[-1].parent, local_files_only=True
        )
        app = make_frontend(tokenizer, None, "test-model", 8192, 10, 2, vision=True)
        pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        body = {
            "model": "test-model",
            "state": "Is the square red?",
            "images": [png_data_url(), png_data_url((30, 30, 200))],
            "questions": {"q": {"type": "noul"}},
        }
        ((_, _, job),) = app.prepare_systemone(
            body, deadline=FOREVER, disconnected=stays_connected
        )
        self.assertEqual(len(job.image_spans), 2)
        for span in job.image_spans:
            self.assertEqual(
                job.prompt_tokens[span.offset : span.offset + span.tokens],
                [pad] * span.tokens,
            )
        # With each image's run narrowed back to its one placeholder, the
        # prompt is what encode_prompt prepares for the same messages.
        narrowed, cursor = [], 0
        for span in job.image_spans:
            narrowed += job.prompt_tokens[cursor : span.offset] + [pad]
            cursor = span.offset + span.tokens
        narrowed += job.prompt_tokens[cursor:]
        state, ((_, spec),), _ = judgments.validate_systemone(body)
        labels = judgments.slot_labels(tokenizer)[: len(spec.labels)]
        expected, _, _ = judgments.encode_prompt(
            app.prompt_tokenizer,
            app.chat_templates.select(None).source,
            judgments.systemone_messages(state, spec, labels, body["images"]),
            labels,
            admit=lambda _: None,
            checkpoint=lambda: None,
        )
        self.assertEqual(narrowed, expected)


if __name__ == "__main__":
    unittest.main()
