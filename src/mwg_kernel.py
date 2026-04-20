"""
Metropolis-Within-Gibbs (MWG) kernel for BlackJAX SMC.

Cycles through parameter blocks, updating each block with its own RMH kernel
while holding other blocks fixed. Each block can have independent proposal
scales and MCMC step counts.

Usage within adaptive_tempered_smc / adaptive_persistent_sampling_smc:
    The MWG kernel is wrapped as a single ``mcmc_step_fn`` that BlackJAX
    calls once per rejuvenation. Internally it loops over blocks, applying
    multiple RMH moves per block if desired.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import random

import blackjax.mcmc.random_walk as blackjax_rw

from .blockwise_rmh import RMHBlock


# ---------------------------------------------------------------------------
# MWG info container
# ---------------------------------------------------------------------------

class MWGInfo(NamedTuple):
    """Info returned by the MWG kernel (one SMC rejuvenation step)."""
    acceptance_rate: jnp.ndarray          # overall (mean across blocks)
    is_accepted: jnp.ndarray              # from last block (for BlackJAX compat)
    proposal: jnp.ndarray                 # from last block
    block_acceptance_rates: jnp.ndarray   # (n_blocks,) per-particle mean


# ---------------------------------------------------------------------------
# Per-block proposal builder
# ---------------------------------------------------------------------------

def _build_block_proposal(block: RMHBlock, scale: jnp.ndarray):
    """Build an additive proposal that only perturbs fields in ``block``."""

    def random_step(rng_key, position):
        keys = random.split(rng_key, len(block.fields))
        move = {name: jnp.zeros_like(val) for name, val in position.items()}
        for key, field in zip(keys, block.fields):
            move[field] = scale * random.normal(
                key, shape=position[field].shape, dtype=position[field].dtype
            )
        return move

    return random_step


# ---------------------------------------------------------------------------
# Core MWG kernel
# ---------------------------------------------------------------------------

def build_mwg_kernel(
    blocks: Sequence[RMHBlock],
    inner_steps_per_block: Dict[str, int] | None = None,
):
    """
    Build a Metropolis-Within-Gibbs kernel compatible with BlackJAX SMC.

    Parameters
    ----------
    blocks : sequence of RMHBlock
        Parameter blocks. Each block is updated in turn while others are fixed.
    inner_steps_per_block : dict, optional
        Number of RMH moves per block per SMC rejuvenation call.
        Default is 1 for every block. Set higher for poorly-mixing blocks.

    Returns
    -------
    mwg_step_fn : callable
        ``(rng_key, state, logdensity_fn, **block_scales) -> (state, info)``
        Signature matches what ``blackjax.adaptive_tempered_smc`` expects for
        ``mcmc_step_fn`` when block scale names are passed as keyword args.
    """
    rmh_kernel = blackjax_rw.build_additive_step()

    if inner_steps_per_block is None:
        inner_steps_per_block = {}

    def mwg_step_fn(rng_key, state, logdensity_fn, **block_scales):
        current_state = state
        block_keys = random.split(rng_key, len(blocks))
        block_acc_rates = []
        last_info = None

        for block_key, block in zip(block_keys, blocks):
            n_inner = inner_steps_per_block.get(block.name, 1)
            scale = block_scales[block.name]
            inner_keys = random.split(block_key, n_inner)

            block_acc_sum = jnp.array(0.0)
            for step_idx in range(n_inner):
                step_fn = _build_block_proposal(block, scale)
                current_state, info = rmh_kernel(
                    inner_keys[step_idx], current_state, logdensity_fn, step_fn
                )
                block_acc_sum = block_acc_sum + info.acceptance_rate
                last_info = info

            block_acc_rates.append(block_acc_sum / n_inner)

        overall_acc = jnp.mean(jnp.stack(block_acc_rates))

        return current_state, MWGInfo(
            acceptance_rate=overall_acc,
            is_accepted=last_info.is_accepted,
            proposal=last_info.proposal,
            block_acceptance_rates=jnp.stack(block_acc_rates),
        )

    return mwg_step_fn


# ---------------------------------------------------------------------------
# Adaptive scale update function (for inner_kernel_tuning)
# ---------------------------------------------------------------------------

def build_mwg_parameter_update_fn(
    blocks: Sequence[RMHBlock],
    field_names: Sequence[str],
    initial_stds: Dict[str, float],
    target_acceptance_by_block: Dict[str, float] | None = None,
    adaptation_rate: float = 1.5,
    min_ratio: float = 0.1,
):
    """
    Build a ``mcmc_parameter_update_fn`` for ``inner_kernel_tuning`` that
    adapts per-block proposal scales based on block-level acceptance rates.

    Parameters
    ----------
    blocks, field_names, initial_stds
        Same as in the main inversion code.
    target_acceptance_by_block : dict, optional
        Target acceptance rate per block name. Defaults:
        ``source_type=0.234, mechanism=0.234, noise=0.234``.
        Note: mechanism target is lower (Roberts et al. optimal for d=3).
    adaptation_rate : float
        Controls how aggressively scales adapt. 1.5 is a good default.
    min_ratio : float
        Floor on proposal scale relative to prior width.

    Returns
    -------
    mcmc_parameter_update_fn : callable
    """
    from .blockwise_rmh import build_blockwise_scale_overrides

    # Default targets: use Roberts-Rosenthal optimal rates per dimension
    default_targets = {
        "source_type": 0.234,   # d=2
        "mechanism": 0.234,    # d=3 (the classic 0.234 rate)
        "noise": 0.234,         # d=1
    }
    if target_acceptance_by_block is not None:
        default_targets.update(target_acceptance_by_block)

    def mcmc_parameter_update_fn(rng_key, state, info):
        current_stds = {
            name: jnp.std(state.particles[name])
            for name in field_names
        }

        # Extract per-block acceptance rates from MWG info
        update_info = getattr(info, "update_info", info)
        block_acc_arr = getattr(update_info, "block_acceptance_rates", None)
        if block_acc_arr is None:
            block_acc_arr = getattr(update_info, "block_acceptance_array", None)

        if block_acc_arr is not None:
            block_acceptance_rate = {
                block.name: jnp.mean(block_acc_arr[i])
                for i, block in enumerate(blocks)
            }
        else:
            block_acceptance_rate = None

        new_scales = build_blockwise_scale_overrides(
            blocks=blocks,
            initial_stds=initial_stds,
            current_stds=current_stds,
            block_acceptance_rate=block_acceptance_rate,
            target_acceptance_by_block=default_targets,
            adaptation_rate=adaptation_rate,
            min_ratio=min_ratio,
        )
        return {
            name: jnp.asarray(scale)[None]
            for name, scale in new_scales.items()
        }

    return mcmc_parameter_update_fn


# ---------------------------------------------------------------------------
# Convenience: MWG blocks with extra inner steps for mechanism
# ---------------------------------------------------------------------------

def default_mwg_blocks(dc: bool, has_ar: bool) -> Tuple[RMHBlock, ...]:
    """Same block layout as default_rmh_blocks."""
    blocks = []
    if not dc:
        blocks.append(RMHBlock("source_type", ("gamma", "delta")))
    blocks.append(RMHBlock("mechanism", ("kappa", "h", "sigma")))
    if has_ar:
        blocks.append(RMHBlock("noise", ("sigma_amp_ratio",)))
    return tuple(blocks)


def default_inner_steps(
    mechanism_steps: int = 5,
    source_type_steps: int = 1,
    noise_steps: int = 1,
) -> Dict[str, int]:
    """
    Default inner MCMC steps per block.

    The mechanism block gets more steps because (kappa, h, sigma) are
    poorly constrained and need more moves to mix within each SMC stage.
    """
    return {
        "source_type": source_type_steps,
        "mechanism": mechanism_steps,
        "noise": noise_steps,
    }
