"""Helper: dump per-chain beta/delta schedules to pick a stall threshold."""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/s0/data/CAPE/smti_workflow/SMTI_dev")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r3_stall_semantics import build  # noqa: E402
from event_loading import load_event  # noqa: E402
import src.inversion_blackjax as ib  # noqa: E402


def main():
    data = load_event("eq00126")
    inv = build(data)
    inv.min_tempering_increment = 1e-4
    inv.tempering_stall_patience = 8
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        inv._invert_multi_chain(inv._filter_event_by_options(data))
    text = buf.getvalue()
    chains = text.split("Chain ")
    for ci, chunk in enumerate(chains[1:]):
        d = [float(x) for x in re.findall(r"\(d=([0-9.\-]+)\)", chunk)]
        print(f"chain {ci}: n={len(d)} deltas=" + ",".join(f"{x:.4f}" for x in d))


if __name__ == "__main__":
    main()
