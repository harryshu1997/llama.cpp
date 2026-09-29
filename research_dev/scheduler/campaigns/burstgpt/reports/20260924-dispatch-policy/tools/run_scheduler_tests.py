#!/usr/bin/env python3
"""Run every research_dev/scheduler/tests/test_*.py of one tree in its own process (run_all.py
semantics, S42 spike tests excluded). Usage: run_scheduler_tests.py ROOT OUT_JSON [name-filter...]"""
import json, subprocess, sys, time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
filters = sys.argv[3:]
tests_dir = root / "research_dev/scheduler/tests"
results = {}
import os
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.pathsep.join([str(root), str(Path('/tmp/claude-1761612022/-home-myid-zs89458-Documents-llama-cpp-release/6fd7e39c-bd31-4e92-b800-5784f94be902/scratchpad/evidence-fixes/shared/gguf-py')), os.environ.get("PYTHONPATH", "")])}
for path in sorted(tests_dir.glob("test_*.py")):
    if filters and not any(f in path.name for f in filters):
        continue
    text = path.read_text()
    if "def test_" not in text:
        continue
    if path.parent == tests_dir and "\nfrom ." in text:
        cmd = [sys.executable, "-m", f"research_dev.scheduler.tests.{path.stem}"]
    else:
        cmd = [sys.executable, str(path)]
    started = time.time()
    try:
        done = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=3600,
                              env=ENV)
        code, tail = done.returncode, (done.stdout + done.stderr)[-6000:]
    except subprocess.TimeoutExpired:
        code, tail = "TIMEOUT", ""
    results[path.name] = {"exit": code, "seconds": round(time.time() - started, 1), "tail": tail}
    print(path.name, code, results[path.name]["seconds"], flush=True)
    out.write_text(json.dumps(results, indent=1))
failed = [k for k, v in results.items() if v["exit"] != 0]
print("FAILED:", failed)
