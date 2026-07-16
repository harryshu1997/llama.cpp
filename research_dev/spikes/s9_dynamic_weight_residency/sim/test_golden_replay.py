#!/usr/bin/env python3
"""S9-V0-R1 golden-replay test (with V0 / V0-R preservation).

Compares a fresh R1 run with the complete persisted R1 manifest and checks independent
process replay hashes. The frozen V0-R simulator is replayed from its preserved canonical
config. V0 has no preserved input config, so its evidence is explicitly limited to module
provenance and stored-manifest self-consistency.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import residency_sim as R  # noqa: E402  (R1 simulator)

CFG = os.path.join(HERE, "scenarios", "baseline_sweep.config.json")
GOLD_R1 = os.path.join(SPIKE, "golden", "v0r1", "baseline_sweep.v0r1.manifest.json")
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))


def sha_file(path):
    return "sha256:" + hashlib.sha256(open(path, "rb").read()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def replay_subprocess(cfg_path):
    r = subprocess.run([sys.executable, os.path.join(HERE, "residency_sim.py"), cfg_path],
                       capture_output=True, text=True, cwd=HERE)
    for line in r.stdout.splitlines():
        if line.startswith("replay:"):
            return line.split("replay:")[1].strip()
    return None


gold = json.load(open(GOLD_R1))
gold_replay = gold["deterministic_replay_sha256"]
cfg = json.load(open(CFG))

# 1. two independent processes reproduce the R1 golden replay hash
p1 = replay_subprocess(CFG)
p2 = replay_subprocess(CFG)
check("R1: cross-process replay reproduces the stored golden", p1 == gold_replay and p2 == gold_replay,
      f"proc={p1[:20] if p1 else None} golden={gold_replay[:20]}")
check("R1: cross-process runs agree", p1 == p2)
check("R1: golden replay hash matches its stored goodput_results",
      R.sha_text(R.canonical(gold["goodput_results"])) == gold_replay)
fresh_manifest = R.build_manifest(cfg, CFG, R.run_sweep(cfg))
check("R1: complete fresh manifest matches stored golden", fresh_manifest == gold)

# 2. MUTATION: tampering with the stored golden changes the hash (non-vacuous)
mutated = json.loads(json.dumps(gold["goodput_results"]))
mutated[0]["baselines"][0]["completed_server"] += 1
check("R1: golden mutation is detected (hash changes)",
      R.sha_text(R.canonical(mutated)) != gold_replay)

# 3. PRESERVE V0-R: frozen sim on frozen canonical config reproduces results and identities
V0R = load(os.path.join(SPIKE, "golden", "v0r_historical", "residency_sim_v0r.py"), "residency_sim_v0r")
v0r_sim_path = os.path.join(SPIKE, "golden", "v0r_historical", "residency_sim_v0r.py")
v0r_cfg = json.load(open(os.path.join(SPIKE, "golden", "v0r", "baseline_sweep.v0r.config.json")))
v0r_gold = json.load(open(os.path.join(SPIKE, "golden", "v0r", "baseline_sweep.v0r.manifest.json")))
v0r_results = V0R.run_sweep(v0r_cfg)
v0r_fresh = V0R.sha_text(V0R.canonical(v0r_results))
check("V0-R preserved: frozen sim reproduces the stored V0-R golden",
      v0r_fresh == v0r_gold["deterministic_replay_sha256"] and v0r_results == v0r_gold["goodput_results"],
      v0r_gold["deterministic_replay_sha256"][:20])
check("V0-R frozen simulator hash matches manifest",
      sha_file(v0r_sim_path) == v0r_gold["code_version"]
      == "sha256:6393e227831e0632addc6716d54b79606a632f0f37f0a49895aa6486ebc9b7eb")
check("V0-R canonical config hash matches manifest",
      V0R.sha_text(V0R.canonical(v0r_cfg)) == v0r_gold["config_hash"])

# 4. PRESERVE V0: source config is absent, so do not claim an executable replay.
v0_sim_path = os.path.join(SPIKE, "golden", "v0_historical", "residency_sim_v0.py")
v0_gold = json.load(open(os.path.join(SPIKE, "golden", "v0_historical", "baseline_sweep.v0.manifest.json")))
check("V0 limited evidence: stored manifest is self-consistent",
      V0R.sha_text(V0R.canonical(v0_gold["goodput_results"])) == v0_gold["deterministic_replay_sha256"],
      v0_gold["deterministic_replay_sha256"][:20])
check("V0 limited evidence: frozen simulator hash matches manifest",
      sha_file(v0_sim_path) == v0_gold["code_version"]
      == "sha256:4491827b4d77fc6d513bf7608e90a293dbc518325c3b48eb18253288dd615313")

# 5. provenance of every frozen V0-R module
frozen_hashes = {
    "residency_sim_v0r.py": "sha256:6393e227831e0632addc6716d54b79606a632f0f37f0a49895aa6486ebc9b7eb",
    "bundle_validate_v0r.py": "sha256:b763c3847fdfbeeafa0c2966488e50e6a03d71dd3090ee33f4b24864c2ed8399",
    "s9lib_v0r.py": "sha256:0bf36ea52cf6b383431ff457c460252ed5f38a8546d726ac9764a8a6cbfc26f5",
}
for f, expected in frozen_hashes.items():
    check(f"frozen {f} hash pinned", sha_file(os.path.join(SPIKE, "golden", "v0r_historical", f)) == expected)

n_fail = sum(1 for _, ok in results if not ok)
print(f"\ngolden-replay tests: {len(results)}  failures: {n_fail}")
sys.exit(1 if n_fail else 0)
