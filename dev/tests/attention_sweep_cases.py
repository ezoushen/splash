"""Check benchmark behavior through its CLI, using the production metallib."""
import json
import subprocess
import sys

binary, library = sys.argv[1:3]
selected = sys.argv[3] if len(sys.argv) > 3 else "all"


def run(*options):
    command = [binary, library, "--shapes", "122b,35b", "--histories", "8192",
               "--phases", "verify", "--lanes", "1", "--repeat", "1", *options]
    print("command:", command, flush=True)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    print(result.stderr, end="")
    print(result.stdout, end="", flush=True)
    assert result.returncode == 0, f"CLI exited {result.returncode}"
    return json.loads(result.stdout)["cases"], result.stderr


def s1():
    cases, diagnostics = run("--extent-pages", "512")
    assert len(cases) == 2
    assert all(c["extent_pages"] == 512 for c in cases), "S1: both fixtures must use the extent override"
    assert diagnostics.count("extents of 512 pages") == 2, "S1: extent override must reach both shapes"


def s2():
    cases, _ = run("--histories", "8192,16384", "--rows", "128,512,2048",
                   "--phases", "both", "--lanes", "1,4", "--extent-pages", "512")
    expected = []
    for history in [8192, 16384]:
        for rows in [128, 512, 2048]:
            for shape in ["ornith-1.5-122b", "qwen3.6-35b-a3b"]:
                expected.append((history, "prefill", rows, 1, shape))
        for lanes in [1, 4]:
            for shape in ["ornith-1.5-122b", "qwen3.6-35b-a3b"]:
                expected.append((history, "verify", 8, lanes, shape))
    observed = [(c["history"], c["kind"], c["rows"], c["lanes"], c["shape"]) for c in cases]
    assert observed == expected, "S2: alternate shapes within each history, row count and phase"


def s3():
    for options in [(), ("--compare-metallib", library)]:
        cases, _ = run("--lanes", "1,4", *options)
        assert len(cases) == (8 if options else 4)
        assert all(c["warmup_graphs_per_submission"] > 1 for c in cases), "S3: verify warm-up batches the measured graph"
        assert all(c["warmup_gpu_seconds"] >= 1.0 for c in cases), \
            "S3: every verify arm needs at least one second of GPU warm-up"
        assert all(c["start_epoch"] < c["end_epoch"] for c in cases)
        assert all(len(c["fused_samples_ms"]) == 1 for c in cases)
    cases, _ = run("--phases", "prefill", "--rows", "128")
    assert all(c["warmup_graphs_per_submission"] == 1 for c in cases), "S3: prefill warm-up stays unchanged"
    assert cases[0]["warmup_gpu_seconds"] >= 1.5
    assert 0.1 <= cases[1]["warmup_gpu_seconds"] < 1.0, "S3: retain the later prefill warm-up"
    cases, _ = run("--phases", "prefill", "--rows", "128", "--warmup-ms", "1500")
    assert all(c["warmup_gpu_seconds"] >= 1.5 for c in cases), "S3: configurable warm-up minimum for clock qualification"


def s4():
    cases, _ = run("--phases", "prefill", "--rows", "128,512,2048")
    expected = {(shape, rows) for shape in ["ornith-1.5-122b", "qwen3.6-35b-a3b"]
                for rows in [128, 512, 2048]}
    assert len(cases) == 6 and {(c["shape"], c["rows"]) for c in cases} == expected, \
        "S4: each shape must actually run all selected prefill row counts"
    assert all(c["kv_bytes"] == (8192 + c["rows"]) * 1040 for c in cases), \
        "S4: selected rows must reach the fixture's KV history"


def s5():
    for variant in [0, 1, 2]:
        cases, _ = run("--phases", "prefill", "--rows", "7", "--variant", str(variant))
        assert len(cases) == 4
        for baseline, candidate in zip(cases[::2], cases[1::2]):
            assert baseline["variant"] == 0 and candidate["variant"] == variant
            assert baseline["arm"] == 0 and candidate["arm"] == 1
            assert baseline["bit_equal"] and baseline["max_abs_diff"] == 0
            if variant == 0:
                assert candidate["bit_equal"] and candidate["max_abs_diff"] == 0, "S5: variant 0 against itself"
            elif variant == 1:
                assert baseline["query_stride"] == 32 and candidate["query_stride"] == 64
                assert candidate["bit_equal"] and candidate["max_abs_diff"] == 0, "S5: compare logical output across layouts"
            else:
                assert not candidate["bit_equal"] and candidate["max_abs_diff"] > 0, "S5: perturbed variant must report a difference"


rules = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}
for name, check in rules.items():
    if selected in ("all", name):
        check()
        print(name, "PASS", flush=True)
assert selected == "all" or selected in rules, f"unknown rule {selected}"
