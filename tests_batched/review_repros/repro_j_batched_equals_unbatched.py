"""REPRO J (control): batched == unbatched on CPU with a synthetic event.

Confirms the core is correct, so every finding below is an integration/robustness
issue rather than a sampler-math issue.
"""
import sys, os; sys.path.insert(0,'.')
from stub import make_prep, QUIET
import numpy as np
import src.inversion_blackjax as ib
import src.inversion_blackjax_batched as bx
from repro_e_driver_contract import build
ib.build_location_samples_from_errors = lambda *a, **k: None
prep = make_prep(seed=99, n_pol=8, n_ar=8)
iu = build(60, 2); iu.set_prep(prep)
ru = iu._invert_multi_chain(dict(prep))
ib_ = build(60, 2); ib_.set_prep(prep)
rb = bx.run_batched(ib_, [("ev", dict(prep))], progress_callback=QUIET)["ev"]
for f in ("gamma","delta","kappa","h","sigma","weights","mt6","sigma_amp_ratio"):
    a=np.asarray(getattr(ru,f)); b=np.asarray(getattr(rb,f))
    print(f"{f:18s} max|diff| = {np.max(np.abs(a-b)):.3e}")
