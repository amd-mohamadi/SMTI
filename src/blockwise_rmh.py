from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import random


DEFAULT_BLOCK_TARGET_ACCEPTANCE = {
    "source_type": 0.44,
    "mechanism": 0.30,
    "noise": 0.44,
}


@dataclass(frozen=True)
class RMHBlock:
    name: str
    fields: tuple[str, ...]


def _require_non_empty_block(block: RMHBlock) -> None:
    if not block.fields:
        raise ValueError(f"RMH block '{block.name}' cannot be empty")


def _require_known_fields(
    block: RMHBlock, known_fields: dict[str, float], label: str
) -> None:
    missing_fields = [field for field in block.fields if field not in known_fields]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(
            f"RMH block '{block.name}' references unknown {label} field(s): {missing}"
        )


def default_rmh_blocks(dc: bool, has_ar: bool) -> tuple[RMHBlock, ...]:
    blocks = []
    if not dc:
        blocks.append(RMHBlock("source_type", ("gamma", "delta")))
    blocks.append(RMHBlock("mechanism", ("kappa", "h", "sigma")))
    if has_ar:
        blocks.append(RMHBlock("noise", ("sigma_amp_ratio",)))
    return tuple(blocks)


def build_blockwise_scale_overrides(
    blocks: tuple[RMHBlock, ...],
    initial_stds: dict[str, float],
    current_stds: dict[str, float],
    baseline_scale: float = 1.0,
    min_ratio: float = 0.1,
    block_acceptance_rate: dict[str, float] | None = None,
    target_acceptance_by_block: dict[str, float] | None = None,
    adaptation_rate: float = 1.5,
    min_acceptance_multiplier: float = 0.5,
    max_acceptance_multiplier: float = 4.0,
):
    """Compute per-block proposal scales using optimal RW-MH scaling.

    The proposal std for each block is:
        scale = baseline_scale * (2.38 / sqrt(d)) * mean(current_stds) * acceptance_multiplier

    where d is the block dimension. This tracks the posterior width at each
    SMC stage. ``initial_stds`` provides a floor: the scale never drops below
    ``min_ratio * optimal_factor * mean(initial_stds)``.
    """
    scales = {}
    baseline = jnp.asarray(baseline_scale)
    target_acceptance = dict(DEFAULT_BLOCK_TARGET_ACCEPTANCE)
    if target_acceptance_by_block is not None:
        target_acceptance.update(target_acceptance_by_block)

    for block in blocks:
        _require_non_empty_block(block)
        _require_known_fields(block, initial_stds, "initial_stds")
        _require_known_fields(block, current_stds, "current_stds")

        d = len(block.fields)
        optimal_factor = 2.38 / jnp.sqrt(jnp.asarray(d, dtype=jnp.float64))

        initial_mean = jnp.mean(
            jnp.asarray([initial_stds[field] for field in block.fields])
        )
        current_mean = jnp.mean(
            jnp.asarray([current_stds[field] for field in block.fields])
        )

        # Scale tracks posterior width; floor prevents degenerate proposals
        scale = optimal_factor * current_mean
        floor = min_ratio * optimal_factor * initial_mean
        scale = jnp.maximum(scale, floor)

        acceptance_multiplier = jnp.asarray(1.0)
        if block_acceptance_rate is not None and block.name in block_acceptance_rate:
            target = target_acceptance.get(block.name, 0.30)
            acceptance = jnp.asarray(block_acceptance_rate[block.name])
            log_multiplier = adaptation_rate * (acceptance - target)
            acceptance_multiplier = jnp.exp(log_multiplier)
            acceptance_multiplier = jnp.clip(
                acceptance_multiplier,
                min_acceptance_multiplier,
                max_acceptance_multiplier,
            )
        scales[block.name] = baseline * scale * acceptance_multiplier

    return scales


def build_masked_random_step(block: RMHBlock, scale):
    _require_non_empty_block(block)

    def random_step(rng_key, position):
        missing_fields = [field for field in block.fields if field not in position]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(
                f"RMH block '{block.name}' references unknown position field(s): {missing}"
            )

        keys = random.split(rng_key, len(block.fields))
        move = {name: jnp.zeros_like(value) for name, value in position.items()}
        for key, field in zip(keys, block.fields):
            move[field] = scale * random.normal(
                key,
                shape=position[field].shape,
                dtype=position[field].dtype,
            )
        return move

    return random_step
