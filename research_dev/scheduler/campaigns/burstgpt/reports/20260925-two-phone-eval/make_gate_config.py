"""Derive the two-phone-eval GATE_CONFIG.json from the integration-2 one (tooling, no hand edits).

    python3 make_gate_config.py INT2_GATE_CONFIG.json OUT_GATE_CONFIG.json ROOT

Only `helper_phone` changes (README section 5 of ../20260925-pixel-cpu-gpu/): the staged DVFS-fixed
worker dir, its six sha256 pins, and the three S43 flags on top of the nine qualified S42 variables.
The previous helper block is kept verbatim as `helper_phone_qualified_reference`: it names the worker
the numerical suite qualified, which prepare_campaign_eval.py bridges from by byte identity.
`phone_hash_cache` and `prompt_file` point at copies under ROOT, so nothing is written into the
Pixel agent's directory.
"""

import json
from pathlib import Path
import shutil
import sys

STAGED = "/data/local/tmp/s43-pixel-cpugpu-20260925-v1"
FIXED_SHA256 = {
    "llama-ffn-split-worker": "5d824455d10961fda0fc6369ee8b885195107f68f450198ac2e6d94549174128",
    "libggml.so": "601b8a7c7d14ab951ee85e6680bea3ed428b73ed63fc8a1fcb55d4999760c6f2",
    "libggml-base.so": "ec8396655c0b24bf828702a6e83371fe6bd3f24b57149e3be7d666813ff9f49d",
    "libggml-cpu.so": "b911532d756cad93e74391e86ed4d0e8e6f66773ec0dab79f8aa893021b0589d",
    "libggml-vulkan.so": "892bf36afff1c3b964afe421a25b9b6cbf0e55de949bee093c91038114cd4fa1",
    "QWEN_PACKED.ffn.gguf": "940f5f1f2ce0c68d726713e0b1ec86808334c7ca769feac07cd3fa8581c4eae9",
}
FLAGS = {"S43_PIXEL_UCLAMP_MIN": "1024", "S43_PIXEL_CPU_POLL": "100", "S43_PIXEL_CPU_BATCH_PAIR": "1"}


def derive(config: dict) -> dict:
    old = config["helper_phone"]
    if "helper_phone_qualified_reference" in config or set(FLAGS) & set(old["worker_environment"]):
        raise SystemExit("source config is already derived")
    if old["worker_environment"].get("S42_PIXEL_CPU_PAIR_DOT") != "1" or int(old["max_tokens"]) > 4:
        raise SystemExit("S43_PIXEL_CPU_BATCH_PAIR needs S42_PIXEL_CPU_PAIR_DOT=1 and max_tokens <= 4")
    # the libraries and the shard must be the same bytes as the qualified ones, only their paths move
    old_values = {value for path, value in old["expected_sha256_by_path"].items() if path != old["worker_path"]}
    new_values = {"sha256:" + value for name, value in FIXED_SHA256.items() if name != "llama-ffn-split-worker"}
    if old_values != new_values:
        raise SystemExit("staged libraries or shard differ from the qualified ones")
    helper = dict(old)
    helper.update(
        worker_path=STAGED + "/llama-ffn-split-worker",
        library_directories=[STAGED],
        shard_path=STAGED + "/QWEN_PACKED.ffn.gguf",
        expected_sha256_by_path={STAGED + "/" + name: "sha256:" + value for name, value in FIXED_SHA256.items()},
        worker_environment={**old["worker_environment"], **FLAGS},
        as_root=True,
    )
    derived = dict(config)
    derived["helper_phone"] = helper
    derived["helper_phone_qualified_reference"] = old
    return derived


def main() -> None:
    source, output, root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    config = json.loads(source.read_text())
    derived = derive(config)
    for key, name in (("phone_hash_cache", "PHONE_HASH_CACHE.json"), ("prompt_file", "PROMPT.txt")):
        copy = root / name
        if not copy.exists():
            shutil.copy2(config[key], copy)
        derived[key] = str(copy)
    with output.open("x") as stream:
        json.dump(derived, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("DERIVED", output)


if __name__ == "__main__":
    main()
