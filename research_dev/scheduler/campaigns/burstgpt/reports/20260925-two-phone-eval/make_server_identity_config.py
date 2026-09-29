"""Derive the Pixel-only server identity config for the fixed worker (tooling, no hand edits).

    python3 make_server_identity_config.py PACKED_SERVER_CONFIG2.json GATE_CONFIG.json OUT.json

Source: the Pixel agent's qualified Pixel-only llama-server comparison
(/mnt/storage/s42-pixel10pro-packed-server-20260924-v1/CONFIG2.json, run by the unchanged
qualify_pixel_server.py: desktop / Pixel 8,704 / Pixel 17,408 / desktop, 64 tokens each). Only the
phone worker, its libraries, shard, environment and sha256 pins are replaced by the configured helper
of GATE_CONFIG.json; the server, model, prompt, ports and server environment are unchanged.
"""

import json
from pathlib import Path
import sys


def derive(source: dict, helper: dict) -> dict:
    config = dict(source)
    directory = helper["library_directories"]
    if len(directory) != 1:
        raise SystemExit("expected one library directory")
    pins = helper["expected_sha256_by_path"]
    config.update(
        phone_worker=helper["worker_path"],
        phone_model=helper["shard_path"],
        phone_library_dir=directory[0],
        phone_libraries=sorted(path for path in pins if path.endswith(".so")),
        phone_environment=dict(helper["worker_environment"]),
        phone_backend=helper["backend"],
        phone_root=bool(helper["as_root"]),
        expected_phone_sha256={path: value[7:] for path, value in pins.items()},
    )
    if config["phone"]["layers"] != [il for il in range(64) if int(helper["layer_mask"]) >> il & 1]:
        raise SystemExit("server identity layers differ from the helper's")
    return config


def main() -> None:
    source, gate, output = (Path(value) for value in sys.argv[1:4])
    config = derive(json.loads(source.read_text()), json.loads(gate.read_text())["helper_phone"])
    with output.open("x") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("DERIVED", output)


if __name__ == "__main__":
    main()
