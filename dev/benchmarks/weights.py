"""The weight images two builds load from one model, compared by their bytes.

A build's engine-tests/weight-digests loads a model as its engine does and
prints the component, size and SHA-256 of every image it holds
(dev/tools/weight_digests.mm). Both builds must load the same images with the
same bytes, except the images an intended change names by a pattern
(--expect-image-change / EXPECT_IMAGE_CHANGE), whose differences are recorded
without failing.

Builds of earlier releases have no weight-digests and loaded a package's
files as they are, so a package's are not compared with theirs. Those that
prepared an assembly's images into a cache, <cache>/<key>/{weights, sha256,
source}, have build/engine/WeightPreparationIdentity.hpp: such a baseline
prepares into a cache of its own, whose images are compared after its rounds.
Once the release baselines of assemblies have weight-digests, delete
IDENTITY_HEADER, PROVENANCE, DIGEST, baseline_environment and prepared, and
compare_builds' environment with the baseline environments that the
regression benchmarks pass it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

WEIGHT_DIGESTS = Path("engine-tests/weight-digests")
IDENTITY_HEADER = Path("engine/WeightPreparationIdentity.hpp")
PROVENANCE = "splash-prepared-weight-v1"
DIGEST = re.compile(r"[0-9a-f]{64}")


def pattern(value: str) -> re.Pattern:
    """value compiled, or the argument error of a malformed pattern."""
    try:
        return re.compile(value)
    except re.error as error:
        raise argparse.ArgumentTypeError(f"invalid pattern {value!r}: {error}")


def add_expect_image_change(parser: argparse.ArgumentParser) -> None:
    """The option, defaulting to EXPECT_IMAGE_CHANGE, of the pattern that
    names the components whose bytes a change means to change; unset, none."""
    parser.add_argument(
        "--expect-image-change",
        metavar="REGEX",
        type=pattern,
        default=os.environ.get("EXPECT_IMAGE_CHANGE") or None,
        help="allow the weight images whose whole component name matches "
        "REGEX to differ or appear in one build only, recorded "
        "(EXPECT_IMAGE_CHANGE=REGEX)",
    )


def loads_in_memory(build: Path) -> bool:
    """Whether a build directory (a checkout's build/) has weight-digests;
    a baseline without it is of an earlier release."""
    return (Path(build) / WEIGHT_DIGESTS).is_file()


def digests(build: Path, package: Path) -> dict:
    """The SHA-256 of each image, by component, that the build directory's
    engine loads from package."""
    tool = Path(build) / WEIGHT_DIGESTS
    result = subprocess.run(
        [str(tool), str(Path(build) / "splash.metallib"), str(package)],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"{tool} exited {result.returncode}: {result.stderr.strip()}"
        )
    return {image["component"]: image["sha256"] for image in json.loads(result.stdout)}


def baseline_environment(directory: Path) -> dict:
    """The variable that gives a baseline of an earlier release a cache of
    its own, directory/baseline-weights, created now."""
    cache = (Path(directory) / "baseline-weights").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    return {"SPLASH_WEIGHT_CACHE": str(cache)}


def prepared(environment: dict, package: Path) -> dict:
    """The SHA-256 of each image, by component, that a baseline of an earlier
    release, started with environment, prepared from package: its cache's
    complete entries whose source is under package."""
    root = str(package)
    images = {}
    for entry in sorted(Path(environment["SPLASH_WEIGHT_CACHE"]).iterdir()):
        if not DIGEST.fullmatch(entry.name):
            continue
        try:
            digest = (entry / "sha256").read_text()
            lines = (entry / "source").read_text().splitlines()
        except (FileNotFoundError, NotADirectoryError):
            continue
        if lines[:1] != [PROVENANCE] or not DIGEST.fullmatch(digest):
            continue
        record = dict(line.partition(" ")[::2] for line in lines[1:])
        source = record.get("source", "")
        if "component" in record and (source == root or source.startswith(root + "/")):
            images[record["component"]] = digest
    return images


def compare(
    baseline: dict, candidate: dict, expected: re.Pattern | None = None
) -> dict:
    """Whether the builds loaded the same images, by component, with the
    same bytes; a difference in a component whose whole name expected
    matches is recorded in expected_changes instead of failing."""
    rows = [
        {
            "component": component,
            "baseline_sha256": baseline.get(component),
            "candidate_sha256": candidate.get(component),
        }
        for component in sorted(baseline.keys() | candidate.keys())
    ]
    failures, changes = [], []
    for row in rows:
        if row["baseline_sha256"] == row["candidate_sha256"]:
            continue
        allowed = expected is not None and expected.fullmatch(row["component"])
        (changes if allowed else failures).append(
            f"{row['component']}: the baseline loaded {row['baseline_sha256'] or 'nothing'}, "
            f"the candidate {row['candidate_sha256'] or 'nothing'}"
        )
    return {
        "images": rows,
        "failures": failures,
        "expected_changes": changes,
        "expect_image_change": expected.pattern if expected else None,
        "pass": not failures,
    }


def compare_builds(
    baseline: Path,
    candidate: Path,
    package: Path,
    environment: dict,
    assembly: bool,
    expected: re.Pattern | None = None,
) -> dict:
    """Compares the images the build directories load from package, the
    model root both were given; environment started the baseline, assembly
    says whether package is one, and expected names the images that may
    change (compare)."""
    if loads_in_memory(baseline):
        images = digests(baseline, package)
    elif not assembly:
        return compare({}, {}, expected)
    elif (Path(baseline) / IDENTITY_HEADER).is_file():
        images = prepared(environment, package)
    else:
        raise RuntimeError(f"the baseline build has no {WEIGHT_DIGESTS}")
    return compare(images, digests(candidate, package), expected)
