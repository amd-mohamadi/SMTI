"""Verify per-event location-sample seeding + takeoff reflection."""
import sys

import numpy as np

sys.path.insert(0, "/s0/data/CAPE/smti_workflow/SMTI")
from src.data_prep import build_location_samples_from_errors, location_sample_rng

failures = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}: {name}")
    if not cond:
        failures.append(name)


def make_event(uid, takeoff=None):
    n = 8
    rng = np.random.default_rng(0)
    az = rng.uniform(0, 360, n)
    to = np.full(n, takeoff) if takeoff is not None else rng.uniform(20, 160, n)
    return {
        "UID": uid,
        "PPolarity": {
            "Stations": {
                "Name": [f"ST{i:02d}" for i in range(n)],
                "Azimuth": az.copy(),
                "TakeOffAngle": to.copy(),
            },
            "Measured": np.ones((n, 1)),
            "Error": np.full((n, 1), 0.1),
        },
    }


def draw(event, seed=421, n_samples=20):
    return build_location_samples_from_errors(
        event,
        rng=location_sample_rng(seed, event["UID"]),
        n_samples=n_samples,
        azimuth_error=5.0,
        takeoff_error=10.0,
    )


def stack(samples):
    return (
        np.stack([s["Azimuth"] for s in samples]),
        np.stack([s["TakeOffAngle"] for s in samples]),
    )


# --- 1. determinism: same (seed, UID) -> identical samples ------------------
ev = make_event("eq00124_PPolarity_mt")
az1, to1 = stack(draw(ev))
az2, to2 = stack(draw(ev))
check("same seed+UID reproduces identical samples",
      np.array_equal(az1, az2) and np.array_equal(to1, to2))

# --- 2. order-independence: drawing other events in between changes nothing -
_ = draw(make_event("eq00001_PPolarity_mt"))
_ = draw(make_event("eq00002_PPolarity_mt"))
az3, to3 = stack(draw(ev))
check("samples independent of processing order", np.array_equal(az1, az3))

# --- 3. different UID or seed -> different draws ----------------------------
az_other, _ = stack(draw(make_event("eq00125_PPolarity_mt")))
check("different UID gives different draws", not np.array_equal(az1, az_other))
az_seed, _ = stack(draw(ev, seed=422))
check("different base seed gives different draws", not np.array_equal(az1, az_seed))

# --- 4. rng stream is stable regardless of construction site ----------------
r1 = location_sample_rng(421, "eq00124_PPolarity_mt").standard_normal(5)
r2 = location_sample_rng(421, "eq00124_PPolarity_mt").standard_normal(5)
check("location_sample_rng deterministic across instances", np.array_equal(r1, r2))

# --- 5. takeoff reflection: near-pole mean, no boundary atom, in range ------
ev_pole = make_event("eq_pole", takeoff=3.0)
_, to_pole = stack(draw(ev_pole, n_samples=2000))
check("takeoff samples within [0, 180]",
      float(to_pole.min()) >= 0.0 and float(to_pole.max()) <= 180.0)
check("no probability atom at 0 (clip artifact gone)",
      int(np.sum(to_pole == 0.0)) == 0)
# reflection => |N(3,10)| distribution: mean ~ 8.2, not 3
mean_pole = float(to_pole.mean())
check(f"reflected mean near E|N(3,10)|~8.2 (got {mean_pole:.2f})",
      7.0 < mean_pole < 9.5)

ev_pole2 = make_event("eq_pole2", takeoff=177.0)
_, to_pole2 = stack(draw(ev_pole2, n_samples=2000))
check("no atom at 180 either",
      int(np.sum(to_pole2 == 180.0)) == 0
      and float(to_pole2.max()) <= 180.0)

# --- 6. Inversion._location_rng: UID path + fallback ------------------------
from src.inversion_blackjax import InversionBlackJAX

inv = InversionBlackJAX(
    make_event("eq00124_PPolarity_mt"),
    inversion_options=["PPolarity"],
    num_particles=10,
    random_seed=421,
    num_chains=1,
)
a = inv._location_rng(make_event("eq00124_PPolarity_mt")).standard_normal(4)
b = location_sample_rng(421, "eq00124_PPolarity_mt").standard_normal(4)
check("Inversion._location_rng matches module helper (CPU==GPU derivation)",
      np.array_equal(a, b))

no_uid = {"PPolarity": make_event("x")["PPolarity"]}
check("event without UID falls back to shared self.rng",
      inv._location_rng(no_uid) is inv.rng)

print()
if failures:
    print(f"{len(failures)} FAILED")
    sys.exit(1)
print("ALL PASS")
