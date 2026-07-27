"""REPRO K: with num_chains == 1 the batched path seeds differently from forward().

forward()            -> _invert_single_event -> random.PRNGKey(self.random_seed)
prepare_batch()      -> SeedSequence(self.random_seed).generate_state(1)[0]

so ``prepare_batch``'s docstring claim ("a batched chain and its unbatched twin
start from the same particles") does not hold for single-chain runs.
"""
import numpy as np

for seed in (1234, 20240216, 7):
    derived = int(np.random.SeedSequence(seed).generate_state(1)[0])
    print(f"  random_seed={seed:<10d} unbatched PRNGKey seed={seed:<12d} "
          f"batched entry seed={derived:<12d} same={seed == derived}")
print("\nBUG: single-chain batched runs are not reproducible against forward().")
