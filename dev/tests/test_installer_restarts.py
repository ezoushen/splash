import contextlib
import hashlib
import io
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from dev.benchmarks import prepared
from dev.tests.fixture_files import write_gguf, write_safetensors
from dev.tests.installer_fixtures import (
    DENSE,
    DRAFT_COMMIT,
    MODEL,
    MOE,
    draft_dir,
    fake_hub,
    mlx_target,
    selection,
)
from dev.tools import installer_restarts as restarts
from install import hub, models, upstream


def in_process(fake):
    """The installer as run_installer runs it, in this process, where fake
    stands in for the Hub the environment names."""

    def run(arguments, environment, timeout=None):
        with contextlib.ExitStack() as stack:
            offline = environment.get("HF_HUB_OFFLINE") == "1"
            stack.enter_context(
                mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", offline)
            )
            if "HF_HUB_CACHE" in environment:
                stack.enter_context(
                    mock.patch(
                        "huggingface_hub.constants.HF_HUB_CACHE",
                        environment["HF_HUB_CACHE"],
                    )
                )
            fake.failure = (
                httpx.ConnectError("[Errno 61] Connection refused")
                if environment.get("HF_ENDPOINT") == restarts.UNREACHABLE_HUB
                else None
            )
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(output))
            status = models.main(arguments)
        return status, output.getvalue()

    return run


def safetensors_bytes(data):
    """A safetensors file of one U8 tensor whose data is data."""
    header = b'{"t":{"dtype":"U8","shape":[%d],"data_offsets":[0,%d]}}' % (
        len(data),
        len(data),
    )
    return struct.pack("<Q", len(header)) + header + data


def safetensors_target(root):
    """MODEL's files, with a shard whose tensor data is b"data"."""
    mlx_target(root, DENSE)
    (root / "model.safetensors").write_bytes(safetensors_bytes(b"data"))


def safetensors_draft(root):
    """DENSE's DFlash2 release, with weights whose tensor data is b"draft"."""
    draft_dir(root, DENSE)
    (root / "model.safetensors").write_bytes(safetensors_bytes(b"draft"))


def gguf_bytes(data):
    """A GGUF of one U8 tensor whose data, 64-byte aligned, is data."""
    name = b"t"
    header = (
        b"GGUF"
        + struct.pack("<IQQ", 3, 1, 1)
        + struct.pack("<Q", len(b"general.alignment"))
        + b"general.alignment"
        + struct.pack("<II", 4, 64)
        + struct.pack("<Q", len(name))
        + name
        + struct.pack("<IQIQ", 1, len(data), 0, 0)
    )
    return header + bytes(-len(header) % 64) + data


def prepared_entry(weights, key, component, inputs, tensors=None):
    """A weight cache entry at key that records its component and inputs,
    and the tensor digests it lists when tensors is given, and what
    loaded_entries names of it."""
    directory = weights / key
    directory.mkdir(parents=True)
    (directory / "source").write_text(
        f"{prepared.PROVENANCE}\ncomponent {component}\ninputs {inputs}\n"
        "source /elsewhere\n"
    )
    if tensors is not None:
        (directory / "tensors").write_text("".join(f"{t}\n" for t in tensors))
    digest = hashlib.sha256(key.encode()).hexdigest()
    (directory / "sha256").write_text(digest)
    return {"component": component, "sha256": digest}


def components(family):
    """The prepared files of a text-only installation of family: its
    target's and its draft's."""
    layers = dict(family.signature)["num_hidden_layers"]
    return sorted(
        ["target/embedding.bin", "target/head.bin", "draft/model.bin"]
        + [f"target/layer-{index}.bin" for index in range(layers)]
        + [f"draft/layer-{index}.bin" for index in range(family.draft.layers)]
    )


def source_inputs(data):
    """The inputs a prepared entry records when it is written from one source
    file, of tensor data data."""
    return hashlib.sha256(hashlib.sha256(data).hexdigest().encode()).hexdigest()


class InstallerRestartsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "hub"
        models_root = str(self.root / "models")
        self.arguments = ["--models", models_root, "--model", MODEL, "--language-only"]

    def check(self, run, arguments):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            restarts.check(arguments, run=run, hub_cache=self.cache)
        return output.getvalue()

    def test_every_restart_starts_the_installation_without_a_download(self):
        for revision in (None, "a" * 40):
            with self.subTest(revision=revision):
                fake = fake_hub(self, self.cache)
                arguments = self.arguments + (
                    ["--revision", revision] if revision else []
                )
                output = self.check(in_process(fake), arguments)
                for name in ("offline", "unreachable-Hub", "moved-cache"):
                    self.assertIn(f"{name} restart: PASS", output)

    def test_a_restart_that_installs_another_commit_fails(self):
        fake = fake_hub(self, self.cache)
        run = in_process(fake)

        def run_then_move(arguments, environment, timeout=None):
            result = run(arguments, environment, timeout)
            fake.publish(MODEL, "b" * 40, lambda p: mlx_target(p, DENSE))
            return result

        with self.assertRaisesRegex(restarts.RestartFailure, "moved-cache restart"):
            self.check(run_then_move, self.arguments)

    def test_a_restart_that_writes_to_a_hub_cache_fails(self):
        def blob(environment):
            if environment.get("HF_HUB_OFFLINE") == "1":
                return self.cache / hub.folder_name(MODEL) / "blobs" / "new"
            return None

        def moved(environment):
            if "HF_HUB_CACHE" in environment:
                return Path(environment["HF_HUB_CACHE"]) / "new"
            return None

        for name, where in (("offline", blob), ("moved-cache", moved)):
            with self.subTest(restart=name):
                self.setUp()
                run = in_process(fake_hub(self, self.cache))

                def writing(arguments, environment, timeout=None):
                    result = run(arguments, environment, timeout)
                    if path := where(environment):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("")
                    return result

                with self.assertRaisesRegex(
                    restarts.RestartFailure, f"the {name} restart wrote to a Hub cache"
                ):
                    self.check(writing, self.arguments)

    def test_a_restart_without_the_documented_line_fails(self):
        fake = fake_hub(self, self.cache)
        run = in_process(fake)

        def run_silently(arguments, environment, timeout=None):
            status, output = run(arguments, environment, timeout)
            return status, output.replace("Could not reach the Hub", "")

        with self.assertRaisesRegex(
            restarts.RestartFailure, "unreachable-Hub restart did not print"
        ):
            self.check(run_silently, self.arguments)

    def test_prepared_names_the_entries_of_the_installations_sources(self):
        fake = fake_hub(self, self.cache)
        fake.publish(MODEL, "a" * 40, safetensors_target)
        fake.publish(DENSE.draft.repo, DRAFT_COMMIT, safetensors_draft)
        chosen = selection(self.root)
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            upstream.prepare(chosen)
        weights = self.root / "weights"
        (weights / "verified").mkdir(parents=True)

        def entry(key, component, data=None):
            """An entry of the one tensor of data, by default the tensor data
            of the installation's source the component reads."""
            data = data or (b"draft" if component.startswith("draft/") else b"data")
            tensors = [hashlib.sha256(data).hexdigest()]
            return prepared_entry(weights, key, component, source_inputs(data), tensors)

        names = components(DENSE)
        expected = [entry(f"{i:064x}", name) for i, name in enumerate(names)]
        # Another model's preparation of a component is not this one's.
        entry("f" * 64, "target/layer-0.bin", b"other")
        self.assertEqual(restarts.loaded_entries(chosen.link, weights), expected)
        entry("e" * 64, "target/layer-0.bin")
        with self.assertRaisesRegex(
            restarts.RestartFailure, "holds 2 prepared target/layer-0.bin"
        ):
            restarts.loaded_entries(chosen.link, weights)
        shutil.rmtree(weights / ("e" * 64))
        # The tensors an entry lists decide, whatever its inputs.
        (weights / f"{names.index('target/head.bin'):064x}" / "tensors").write_text(
            f"{'1' * 64}\n"
        )
        with self.assertRaisesRegex(
            restarts.RestartFailure, "holds no prepared target/head.bin"
        ):
            restarts.loaded_entries(chosen.link, weights)

    def test_prepared_joins_entries_on_the_tensors_they_list(self):
        # A base and its fine-tune, each with two tensors in its target
        # shard and in its draft's, share one cache. The fine-tune changed x:
        # its layer 0, which reads y alone, and its draft are the base's.
        weights = self.root / "weights"

        def installation(name, x):
            link = self.root / name
            for directory, tensors in (
                ("target", {"x": ([4], "U8", x), "y": ([4], "U8", b"yyyy")}),
                ("draft", {"d": ([5], "U8", b"draft"), "e": ([4], "U8", b"eeee")}),
            ):
                (link / directory).mkdir(parents=True)
                write_safetensors(link / directory / "model.safetensors", tensors)
            files = {"target/model.safetensors": {}, "draft/model.safetensors": {}}
            (link / "model.json").write_text(
                json.dumps(
                    {"family": DENSE.name, "vision_format": "none", "files": files}
                )
            )
            return link

        def entry(key, component, *data):
            digests = sorted(hashlib.sha256(d).hexdigest() for d in data)
            inputs = hashlib.sha256("".join(digests).encode()).hexdigest()
            return prepared_entry(weights, key, component, inputs, digests)

        base, tuned = installation("base", b"base"), installation("tuned", b"tune")
        expected = {base: [], tuned: []}
        for index, name in enumerate(components(DENSE)):
            if name.startswith("draft/") or name == "target/layer-0.bin":
                data = (b"draft", b"eeee") if name.startswith("draft/") else (b"yyyy",)
                shared = entry(f"{index:064x}", name, *data)
                expected[base].append(shared)
                expected[tuned].append(shared)
            else:
                expected[base].append(entry(f"{index:064x}", name, b"base", b"yyyy"))
                expected[tuned].append(
                    entry(f"{index + 100:064x}", name, b"tune", b"yyyy")
                )
        for link in (base, tuned):
            self.assertEqual(restarts.loaded_entries(link, weights), expected[link])

    def test_prepared_names_the_experts_files_of_an_mlx_moe_target(self):
        # An MLX target prepares each MoE layer's routed experts into a file
        # of their own; a GGUF target's layer images keep them.
        layers = dict(MOE.signature)["num_hidden_layers"]
        for target_format, target, data in (
            ("mlx-affine", "target/model.safetensors", safetensors_bytes(b"abcd")),
            ("gguf", "target/model.gguf", gguf_bytes(b"abcd")),
        ):
            with self.subTest(target_format=target_format):
                link = self.root / target_format
                (link / "target").mkdir(parents=True)
                (link / target).write_bytes(data)
                (link / "draft").mkdir()
                (link / "draft/model.safetensors").write_bytes(
                    safetensors_bytes(b"draft")
                )
                files = {target: {}, "draft/model.safetensors": {}}
                (link / "model.json").write_text(
                    json.dumps(
                        {
                            "family": MOE.name,
                            "target_format": target_format,
                            "vision_format": "none",
                            "files": files,
                        }
                    )
                )
                names = components(MOE)
                if target_format == "mlx-affine":
                    names = sorted(
                        names + [f"target/experts-{i}.bin" for i in range(layers)]
                    )
                weights = link / "weights"
                expected = [
                    prepared_entry(
                        weights,
                        f"{i:064x}",
                        name,
                        source_inputs(
                            b"draft" if name.startswith("draft/") else b"abcd"
                        ),
                    )
                    for i, name in enumerate(names)
                ]
                self.assertEqual(restarts.loaded_entries(link, weights), expected)

    def test_a_gguf_digest_starts_at_the_aligned_tensor_data(self):
        path = self.root / "model.gguf"
        path.write_bytes(gguf_bytes(b"abcd"))
        self.assertEqual(
            restarts.source_digests(path, False)[0], hashlib.sha256(b"abcd").hexdigest()
        )

    def test_an_mmproj_tensor_digest_is_its_bytes_after_the_aligned_table(self):
        # An F32 tensor and a BF16 one, each padded to the 32-byte alignment.
        path = write_gguf(
            self.root / "mmproj.gguf",
            {"general.architecture": "clip"},
            [("a", [2], 0, b"aaaaaaaa"), ("b", [3, 1], 30, b"bbbbbb")],
        )
        data = b"aaaaaaaa" + bytes(24) + b"bbbbbb" + bytes(26)
        self.assertEqual(
            restarts.source_digests(path),
            (
                hashlib.sha256(data).hexdigest(),
                {hashlib.sha256(t).hexdigest() for t in (b"aaaaaaaa", b"bbbbbb")},
            ),
        )

    def test_prepared_reads_a_gguf_source_through_its_assembly_link(self):
        # The Hub cache holds a snapshot's files as blobs named by their
        # hashes; an assembly links them under their own names.
        blob = self.root / "hub/blobs" / ("0" * 64)
        blob.parent.mkdir(parents=True)
        blob.write_bytes(gguf_bytes(b"abcd"))
        draft = self.root / "hub/blobs" / ("1" * 64)
        draft.write_bytes(safetensors_bytes(b"draft"))
        link = self.root / "assembly"
        (link / "target").mkdir(parents=True)
        (link / "target/model.gguf").symlink_to(blob)
        (link / "draft").mkdir()
        (link / "draft/model.safetensors").symlink_to(draft)
        (link / "model.json").write_text(
            json.dumps(
                {
                    "family": DENSE.name,
                    "vision_format": "none",
                    "files": {"target/model.gguf": {}, "draft/model.safetensors": {}},
                }
            )
        )
        weights = self.root / "weights"
        expected = [
            prepared_entry(
                weights,
                f"{i:064x}",
                name,
                source_inputs(b"draft" if name.startswith("draft/") else b"abcd"),
            )
            for i, name in enumerate(components(DENSE))
        ]
        self.assertEqual(restarts.loaded_entries(link, weights), expected)


if __name__ == "__main__":
    unittest.main()
