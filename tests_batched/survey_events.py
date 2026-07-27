#!/usr/bin/env python
"""Survey production events for polarity / amplitude-ratio observation counts.

Loads each event exactly the way ``m0_spike/prepare_eq00124_arrays.py`` does
(step-4 station filter from ``4_run_inversion.py`` + ``src.data_loader.read_data``)
and reports ``N_pol`` / ``N_ar`` so ``validate_gpu.py`` can pick a heterogeneous
multi-event batch.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        conda run -n smti python tests_batched/survey_events.py --limit 40
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import argparse
import json
import sys
import time

import numpy as np

SMTI_DEV = os.environ.get("SMTI_ROOT", "/s0/data/CAPE/smti_workflow/SMTI_dev")
WORKFLOW = "/s0/data/CAPE/smti_workflow/smti_inversion"
DATA_ROOT = "/s0/data/CAPE/smti_workflow/inversion_data/data"
if SMTI_DEV not in sys.path:
    sys.path.insert(0, SMTI_DEV)

from event_loading import EVENT_IDS_ALL, count_obs, load_event  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ids = EVENT_IDS_ALL[args.start : args.start + args.limit]
    rows = []
    for eid in ids:
        t0 = time.time()
        try:
            data = load_event(eid)
        except Exception as exc:  # noqa: BLE001
            print(f"{eid}: LOAD FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        n_pol, n_ar = count_obs(data)
        dt = time.time() - t0
        rows.append(dict(event_id=eid, n_pol=int(n_pol), n_ar=int(n_ar), load_s=dt))
        print(f"{eid}: N_pol={n_pol:3d} N_ar={n_ar:3d}  ({dt:.1f}s)", flush=True)

    rows.sort(key=lambda r: (r["n_pol"] + r["n_ar"]))
    print("\nsorted by N_pol+N_ar:")
    for r in rows:
        print(f"  {r['event_id']}: N_pol={r['n_pol']:3d} N_ar={r['n_ar']:3d}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
