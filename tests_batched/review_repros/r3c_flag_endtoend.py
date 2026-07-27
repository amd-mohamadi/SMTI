"""R3c: end-to-end run_batched with the r3 stall settings -> event-level flag."""
import contextlib, io, os, sys
import numpy as np
sys.path.insert(0, "/s0/data/CAPE/smti_workflow/SMTI_dev")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r3_stall_semantics import build
from event_loading import load_event

def main():
    from src.inversion_blackjax_batched import run_batched
    data = load_event("eq00126")
    inv = build(data)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = run_batched(inv, [("eq00126", data)], smc_dtype="float64",
                          progress_callback=lambda s: None)
    r = res["eq00126"]
    print("event-level tempering_stalled :", r.tempering_stalled)
    print("gamma shape                   :", r.gamma.shape)
    print("(all 4 chains reached lambda=1.0 in the batched run -- see r3 output)")
main()
