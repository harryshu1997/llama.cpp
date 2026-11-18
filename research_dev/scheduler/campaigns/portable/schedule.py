"""Schedules of the portable path.

A schedule is ``{"name", "gap_s", "phases": [{"name", "requests": [{"id", "arrival_s", "prompt_tokens" |
"prompt_text", "output_tokens", "columns"}]}]}``. Phases run one after another; inside a phase every
request is sent at its arrival time whatever the server is doing (open loop). ``columns`` is the share
the helper computes for that request (0 = the host alone). A request may carry ``prompt_text`` in place
of token ids: the engine tokenises it with the loaded model, so a schedule does not depend on a model.

Builders:

    sweep          one request per leg at a list of shares, host-only legs first, last and in between
                   (the brackets that keep host drift out of the comparison); onboarding a phone
    host_baseline  host-only legs; onboarding a host
    seq            every request served twice back to back, alone and shared, the order alternating
    trace          the requests at their recorded arrival times (scaled), alone or shared
    rows           k simultaneous copies of a probe at each share
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


UNMETERED_PREFIXES = ("warm", "rewarm")
# A neutral probe (about 250 tokens in most tokenizers): decode time does not depend on the words.
PROBE_TEXT = (
    "The following is a plain technical note about how a water treatment plant works. Raw water is drawn "
    "from a river through coarse screens that hold back leaves and debris. It then flows into a basin "
    "where a coagulant is mixed in, so that fine particles stick together and form larger flocs. In the "
    "next basin the water moves slowly and the flocs settle to the bottom, from where the sludge is "
    "removed at regular intervals. The clarified water passes through beds of sand and gravel that hold "
    "back what is left, and the filters are cleaned by sending water through them in the opposite "
    "direction. After filtration a disinfectant is added in a contact tank that gives it enough time to "
    "act. The acidity is adjusted so that the water does not attack the pipes, and the finished water is "
    "pumped to covered reservoirs on high ground, from which it reaches homes by gravity. Operators "
    "measure turbidity, residual disinfectant and flow at every stage and keep a written record. "
    "Describe each stage again in your own words, one stage per paragraph, and say what would go wrong "
    "if that stage were left out."
)


class ScheduleError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScheduleError(message)


def _prompt(prompt) -> dict[str, object]:
    """A request's prompt field: token ids or text."""
    if isinstance(prompt, str):
        _require(bool(prompt), "prompt text is empty")
        return {"prompt_text": prompt}
    tokens = [int(token) for token in prompt]
    _require(bool(tokens), "prompt has no token")
    return {"prompt_tokens": tokens}


def _request(request_id, prompt, output_tokens: int, columns: int, arrival_s: float = 0.0) -> dict[str, object]:
    _require(output_tokens >= 2, "a request needs at least two output tokens")
    return {"id": request_id, "arrival_s": float(arrival_s), **_prompt(prompt), "output_tokens": int(output_tokens),
            "columns": int(columns)}


def _phase(name: str, requests: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {"name": name, "requests": list(requests)}


def sweep(name: str, probe, shares_columns: Sequence[int], *, output_tokens: int = 128, brackets_every: int = 2,
          gap_s: float = 5.0) -> dict[str, object]:
    """Legs at each column count with host-only legs before, after and after every ``brackets_every`` legs."""
    columns = [int(value) for value in shares_columns]
    _require(bool(columns) and all(value > 0 for value in columns), "a sweep needs positive column counts")
    order = [0]
    for index, value in enumerate(columns):
        order.append(value)
        if (index + 1) % brackets_every == 0 or index == len(columns) - 1:
            order.append(0)
    phases = [_phase("warm", [_request("warm", probe, output_tokens, 0)])]
    phases += [_phase(f"leg{index:02d}c{value}", [_request("probe", probe, output_tokens, value)])
               for index, value in enumerate(order)]
    return {"name": name, "gap_s": gap_s, "phases": phases}


def host_baseline(name: str, probe, *, legs: int = 3, output_tokens: int = 128, gap_s: float = 5.0) -> dict[str, object]:
    _require(legs >= 1, "a host baseline needs at least one leg")
    phases = [_phase("warm", [_request("warm", probe, output_tokens, 0)])]
    phases += [_phase(f"leg{index:02d}c0", [_request("probe", probe, output_tokens, 0)]) for index in range(legs)]
    return {"name": name, "gap_s": gap_s, "phases": phases}


def seq(name: str, requests: Sequence[Mapping[str, object]], columns: int, probe, *, warm_tokens: int = 450,
        gap_s: float = 5.0) -> dict[str, object]:
    """Each request alone and shared, back to back; even requests alone first, odd ones shared first."""
    _require(columns > 0 and bool(requests), "seq needs requests and a positive column count")
    phases = [_phase(f"warm{value}", [_request(f"warm{value}", probe, warm_tokens, value)]) for value in (0, columns)]
    for index, row in enumerate(requests):
        pair = [_phase(f"r{index:02d}c{value}", [_request(row["id"], row["prompt"], row["output_tokens"], value)])
                for value in (0, columns)]
        phases += pair if index % 2 == 0 else pair[::-1]
    return {"name": name, "gap_s": gap_s, "phases": phases}


def trace(name: str, requests: Sequence[Mapping[str, object]], columns: int, probe, *, scale: float = 1.0,
          warm_tokens: int = 400, gap_s: float = 5.0) -> dict[str, object]:
    """The requests at their recorded arrivals times ``scale`` (0.5 = twice the rate)."""
    _require(bool(requests) and scale > 0 and columns >= 0, "trace needs requests, a positive scale, columns >= 0")
    first = min(float(row["arrival_s"]) for row in requests)
    rows = [_request(row["id"], row["prompt"], row["output_tokens"], columns,
                     round((float(row["arrival_s"]) - first) * scale, 3))
            for row in sorted(requests, key=lambda row: float(row["arrival_s"]))]
    return {"name": name, "gap_s": gap_s, "phases": [
        _phase("warm", [_request("warm", probe, warm_tokens, columns)]), _phase("trace", rows)]}


def rows(name: str, probe, settings: Sequence[int], *, kmax: int = 4, output_tokens: int = 96,
         gap_s: float = 8.0) -> dict[str, object]:
    phases = [_phase("warm", [_request(0, probe, output_tokens, 0)])]
    phases += [_phase(f"k{k}c{value}", [_request(index, probe, output_tokens, value) for index in range(k)])
               for k in range(1, kmax + 1) for value in settings]
    return {"name": name, "gap_s": gap_s, "phases": phases}


def load_requests(path: Path) -> list[dict[str, object]]:
    """Recorded requests: JSON lines (or a JSON array) of ``{"event_id" | "id", "arrival_us" | "arrival_s",
    "prompt_tokens" | "prompt_text", "output_tokens"}``, in arrival order."""
    text = Path(path).read_text(encoding="utf-8")
    raw = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines()
                                                                  if line.strip()]
    out = []
    for row in raw:
        arrival = float(row["arrival_s"]) if "arrival_s" in row else float(row["arrival_us"]) / 1e6
        prompt = row["prompt_tokens"] if "prompt_tokens" in row else row["prompt_text"]
        out.append({"id": row.get("event_id", row.get("id")), "arrival_s": arrival, "prompt": prompt,
                    "output_tokens": int(row["output_tokens"])})
    _require(bool(out) and all(row["id"] is not None for row in out), "requests need ids")
    return sorted(out, key=lambda row: row["arrival_s"])


def validate(schedule: Mapping[str, object], *, stored_columns: int, column_quantum: int) -> list[dict[str, object]]:
    """Every request of a schedule, checked against what the helper stores (0 = no helper: all alone)."""
    _require(bool(schedule.get("name")) and bool(schedule.get("phases")), "schedule needs a name and phases")
    names = [phase["name"] for phase in schedule["phases"]]
    _require(len(set(names)) == len(names), "schedule phase names are not unique")
    every = [request for phase in schedule["phases"] for request in phase["requests"]]
    _require(bool(every) and all(phase["requests"] for phase in schedule["phases"]), "every phase needs a request")
    for request in every:
        _require(int(request["output_tokens"]) >= 2, "every request needs at least two output tokens")
        _require(bool(request.get("prompt_tokens")) != bool(request.get("prompt_text")),
                 "every request needs prompt tokens or a prompt text")
        columns = int(request.get("columns", 0))
        if not stored_columns:
            _require(columns == 0, "host alone: every request must have columns 0")
        else:
            _require(columns % column_quantum == 0 and 0 <= columns <= stored_columns,
                     f"columns must be multiples of {column_quantum} up to the stored {stored_columns}")
    return every
