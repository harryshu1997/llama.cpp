"""Compare exact saved tokens and trace energy, including every phone domain."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "20260921-fast-path-trace-v2a"))
from analyze_longdecode_pair import analyze  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.baseline, args.treatment)
    for summary, directory in zip(result["energy"], (args.baseline, args.treatment)):
        raw = json.loads((directory / "RESULT.json").read_text())
        domains = raw["trace_energy"]["fleet_energy_uj_by_domain"]
        phones = {key: value for key, value in domains.items()
                  if key not in {"cpu-package", "gpu-board"}}
        summary["phone_kj_assumed"] = sum(phones.values()) / 1e9
        summary["assumed_phone_kj_by_domain"] = {key: value / 1e9 for key, value in phones.items()}
        summary["fleet_kj"] = sum(domains.values()) / 1e9
        summary["source"] = str(directory / "RESULT.json")
    result["energy_note"] = (
        "Host uses RAPL package plus NVML board energy. All phone domains use assumed power; "
        "fleet_kj is not a measured fleet total. Single development arm per configuration."
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    with args.output.open("x") as stream:
        stream.write(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
