#!/usr/bin/env python3
"""Restart one installation's installer as a release checks it, and record
the prepared weights it loads (DEVELOPMENT.md, Release check).

`prepare` runs with the Hub, then three restarts must each start the same
installation within RESTART_SECONDS without writing to a Hub cache:
HF_HUB_OFFLINE=1, a Hub that refuses connections, and an empty HF_HUB_CACHE.
`verify --full` then hashes every source file. A legacy package is only
verified. The output names, by component and SHA-256, the prepared-weight
cache entries the installation loads, which a load of it must have written.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.benchmarks import prepared  # noqa: E402
from install import families, gguf, hub, models  # noqa: E402

# The installer's 5 s Hub request (hub.HUB_TIMEOUT) and its own start.
RESTART_SECONDS = 10
UNREACHABLE_HUB = "http://127.0.0.1:9"
# Each run's Hub, whatever the caller's HF_HUB_OFFLINE says.
ONLINE = {"HF_HUB_OFFLINE": "0"}


# The element bytes of the GGML types a vision tower's mmproj stores, which
# preparation reads (VisionLoader.cpp): F32, F16 and BF16.
MMPROJ_ELEMENT_BYTES = {0: 4, 1: 2, 30: 2}


class RestartFailure(RuntimeError):
    pass


def run_installer(arguments, environment, timeout=None):
    """install/models.py's exit status and output with arguments, in this
    environment with environment's variables added."""
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "install/models.py"), *arguments],
            env=os.environ | environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, f"did not exit within {timeout} s"
    return result.returncode, result.stdout


def hub_files(cache: Path, repositories):
    """Every downloaded file and snapshot link of the repositories in a Hub
    cache, with its modification time; the installer's references to them
    are not downloads."""
    return {
        path: path.lstat().st_mtime_ns
        for repository in repositories
        for part in ("blobs", "snapshots")
        for path in (cache / hub.folder_name(repository) / part).glob("**/*")
    }


def installer(run, arguments, command, environment):
    """Run the installer command; fail unless it exits 0."""
    status, output = run([*arguments, *command], environment)
    print(output, end="", flush=True)
    if status != 0:
        raise RestartFailure(f"{' '.join(command)} failed")


def restart(run, arguments, name, environment, expected):
    started = time.monotonic()
    status, output = run([*arguments, "prepare"], environment, RESTART_SECONDS)
    elapsed = time.monotonic() - started
    if status != 0 or elapsed > RESTART_SECONDS:
        raise RestartFailure(f"{name} restart failed after {elapsed:.1f} s:\n{output}")
    for line in expected:
        if line not in output:
            raise RestartFailure(f"{name} restart did not print {line!r}:\n{output}")
    print(f"{name} restart: PASS ({elapsed:.1f} s)", flush=True)


def check(arguments, run=run_installer, hub_cache=None):
    """Run the installer with arguments, the --model and source options of
    one installation, as a release checks it; its selection."""
    options = models.parse_args([*arguments, "link"])
    selection = models.Selection.of(
        options.models,
        options.model,
        revision=options.revision,
        language_only=options.language_only,
        draft_model=options.draft_model,
    )
    if models.installation_kind(selection.link) != models.PACKAGE:
        installer(run, arguments, ["prepare"], ONLINE)
        if hub_cache is None:
            from huggingface_hub import constants

            hub_cache = Path(constants.HF_HUB_CACHE)
        installed = selection.link.resolve()
        sources = models.read_json(installed / "model.json")["sources"]
        target = sources["target"]
        # A local draft directory is no Hub repository.
        repositories = [s["repo"] for s in sources.values() if s["revision"]]
        started = (
            f"Splash model {selection.model} is already installed in {selection.link}"
        )
        # A commit revision starts without asking the Hub.
        fallback = (
            ()
            if models.is_hex_digest(selection.revision, 40)
            else (
                "Could not reach the Hub (",
                f"); using the installed {selection.repo_id}@{target['revision'][:12]}.",
            )
        )
        downloads = hub_files(hub_cache, repositories)
        with tempfile.TemporaryDirectory() as moved:
            for name, environment, expected in (
                ("offline", {"HF_HUB_OFFLINE": "1"}, (started,)),
                (
                    "unreachable-Hub",
                    ONLINE | {"HF_ENDPOINT": UNREACHABLE_HUB},
                    (*fallback, started),
                ),
                ("moved-cache", ONLINE | {"HF_HUB_CACHE": moved}, (started,)),
            ):
                restart(run, arguments, name, environment, expected)
                if selection.link.resolve() != installed:
                    raise RestartFailure(
                        f"the {name} restart relinked {selection.link}"
                    )
                if hub_files(hub_cache, repositories) != downloads or any(
                    Path(moved).iterdir()
                ):
                    raise RestartFailure(f"the {name} restart wrote to a Hub cache")
    installer(run, arguments, ["verify", "--full"], {})
    return selection


def update(file, size, *hashes):
    """Hashes the next size bytes of file, all the rest when size is None."""
    while size is None or size > 0:
        chunk = file.read(1 << 20 if size is None else min(size, 1 << 20))
        if not chunk:
            if size is None:
                return
            raise ValueError(f"truncated source file {file.name}")
        for hasher in hashes:
            hasher.update(chunk)
        if size is not None:
            size -= len(chunk)


def gguf_tensors(path: Path):
    """The (offset in the tensor data, bytes) of each tensor of an mmproj,
    read from its tensor table, which follows its metadata."""
    ranges = []
    with path.open("rb") as file:
        file.seek(8)
        (count,) = struct.unpack("<Q", file.read(8))
        file.seek(gguf.Metadata(path).consumed)
        for _ in range(count):
            (length,) = struct.unpack("<Q", file.read(8))
            name = file.read(length).decode()
            (rank,) = struct.unpack("<I", file.read(4))
            shape = struct.unpack(f"<{rank}Q", file.read(8 * rank))
            kind, offset = struct.unpack("<IQ", file.read(12))
            if kind not in MMPROJ_ELEMENT_BYTES:
                raise ValueError(f"{path}: {name} is not F32, F16 or BF16")
            ranges.append((offset, math.prod(shape) * MMPROJ_ELEMENT_BYTES[kind]))
    return ranges


def source_digests(path: Path, tensors=True):
    """The SHA-256 of a source file's tensor data, after its header, as the
    inputs of an entry that lists no tensors name it (WeightSource::digest),
    and with tensors the set of those of each of its tensors, as an entry's
    tensors file lists them (WeightSource::tensorDigest); one pass. The name
    of path gives the format, so path is an assembly's link, not the Hub
    cache blob it resolves to: a blob is named by its hash."""
    ranges = []
    if path.suffix == ".gguf":
        header = gguf.Metadata(path, tensors=True)
        alignment = header.values.get("general.alignment", 32)
        offset = -(-header.consumed // alignment) * alignment
        if tensors:
            ranges = gguf_tensors(path)
    else:
        with path.open("rb") as file:
            length = int.from_bytes(file.read(8), "little")
            records = json.loads(file.read(length)) if tensors else {}
        offset = 8 + length
        for name, record in records.items():
            if name != "__metadata__":
                begin, end = record["data_offsets"]
                ranges.append((begin, end - begin))
    data, digests = hashlib.sha256(), set()
    with path.open("rb") as file:
        file.seek(offset)
        at = 0
        for begin, size in sorted(ranges):
            if begin < at:
                raise ValueError(f"overlapping tensors in {path}")
            update(file, begin - at, data)
            tensor = hashlib.sha256()
            update(file, size, data, tensor)
            digests.add(tensor.hexdigest())
            at = begin + size
        update(file, None, data)
    return data.hexdigest(), digests


def loaded_entries(link: Path, cache: Path):
    """The (component, SHA-256) of each cache entry the installation at link
    loads. An entry's key hashes the preparation identity and plan too, but
    its source file names its component and its inputs; one entry holds each
    component and inputs (PreparedWeights.cpp evictSuperseded). An entry
    keyed by the content of its source tensors lists their digests, all of
    which a source of the installation holds; another model's source may
    hold them too, as a fine-tune shares its base's unchanged tensors. The
    inputs of any other entry (a GGUF target's, or one of the first
    provenance version) are the digest of the sorted data digests of the
    source files it reads."""
    if models.installation_kind(link) == models.PACKAGE:
        return []  # A package's files are mapped as they are.
    record = models.read_json(link / "model.json")
    family = families.named(record["family"])
    layers = dict(family.signature)["num_hidden_layers"]
    components = {
        "target/embedding.bin",
        "target/head.bin",
        *(f"target/layer-{index}.bin" for index in range(layers)),
        "draft/model.bin",
        *(f"draft/layer-{index}.bin" for index in range(family.draft.layers)),
    }
    # An MLX target prepares each MoE layer's routed experts into a file of
    # their own (AffineTarget.cpp).
    moe = "num_experts" in dict(family.signature)
    if moe and record["target_format"] == "mlx-affine":
        components.update(f"target/experts-{index}.bin" for index in range(layers))
    if record["vision_format"] != "none":
        components.add("vision/model.bin")
    # Each file is hashed once, through one of its links: the vision tower
    # of an MLX model links the target's shards again.
    sources = {}
    for name in record["files"]:
        if name.startswith(("target/", "draft/", "vision/")) and name.endswith(
            (".safetensors", ".gguf")
        ):
            sources.setdefault((link / name).resolve(), name)
    files, tensors = set(), set()
    for name in sources.values():
        # A GGUF target's entries list no tensors.
        data, digests = source_digests(
            link / name, not (name.startswith("target/") and name.endswith(".gguf"))
        )
        files.add(data)
        tensors |= digests
    digests = sorted(files)
    inputs = {
        hashlib.sha256("".join(subset).encode()).hexdigest()
        for size in range(1, len(digests) + 1)
        for subset in itertools.combinations(digests, size)
    }
    found = {}
    for entry in prepared.entries(cache):
        if entry.get("component") in components and (
            tensors.issuperset(entry["tensors"])
            if "tensors" in entry
            else entry.get("inputs") in inputs
        ):
            found.setdefault(entry["component"], []).append(entry)
    for component in sorted(components):
        entries = found.get(component, [])
        if not entries:
            raise RestartFailure(
                f"{cache} holds no prepared {component} of {link}; loading it "
                "prepares it (make test-real)"
            )
        if len(entries) > 1:
            raise RestartFailure(
                f"{cache} holds {len(entries)} prepared {component} of {link}: "
                "give each preparation identity its own SPLASH_WEIGHT_CACHE"
            )
    return [
        {"component": name, "sha256": found[name][0]["sha256"]}
        for name in sorted(components)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments, installer_arguments = parser.parse_known_args(argv)
    try:
        selection = check(installer_arguments)
        record = {
            "model": selection.model,
            "prepared": loaded_entries(selection.link, prepared.cache_root(os.environ)),
        }
    except (RestartFailure, models.ModelError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"installer restarts: PASS ({arguments.output})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
