"""Machine-readable SuShiE run statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict

from .infer_ss import PreparationOutcome

StatsStatus = Literal[
    "SUCCESS",
    "NON_CONVERGED",
    "EMPTY_INTERSECTION",
    "INSUFFICIENT_VARIANTS",
    "FAILED",
]


class SuShiEStats(BaseModel):
    """Status emitted for one SuShiE application invocation."""

    model_config = ConfigDict(extra="forbid")

    runId: str
    fineMappingLocusSetId: str
    status: StatsStatus
    converged: bool | None = None
    inputVariants: int | None = None
    sharedVariants: int | None = None
    ldValidVariants: int | None = None
    finalVariants: int | None = None
    nComponents: int | None = None
    reason: str | None = None


def stats_from_preparation(
    outcome: PreparationOutcome, *, run_id: str, fine_mapping_locus_set_id: str
) -> SuShiEStats:
    """Convert the strict intersection outcome into the status contract."""

    return SuShiEStats(
        runId=run_id,
        fineMappingLocusSetId=fine_mapping_locus_set_id,
        status=cast(StatsStatus, outcome.status),
        inputVariants=outcome.input_variants,
        sharedVariants=outcome.shared_variants,
        ldValidVariants=outcome.ld_valid_variants,
        finalVariants=outcome.final_variants,
        reason=outcome.reason,
    )


def stats_from_result(
    result: Any,
    *,
    run_id: str,
    fine_mapping_locus_set_id: str,
    min_tol: float = 1e-4,
) -> SuShiEStats:
    """Summarize a reportable or non-converged SuShiE result."""

    log_bf = np.asarray(result.posteriors.log_bf)
    converged = result_converged(result, min_tol)
    return SuShiEStats(
        runId=run_id,
        fineMappingLocusSetId=fine_mapping_locus_set_id,
        status="SUCCESS" if converged else "NON_CONVERGED",
        converged=converged,
        finalVariants=int(log_bf.shape[-1]),
        nComponents=int(log_bf.shape[0]),
        reason=None if converged else "SuShiE fit did not converge",
    )


def result_converged(result: Any, min_tol: float) -> bool:
    """Return whether the final monotone ELBO update met the tolerance."""

    if not bool(result.elbo_increase):
        return False
    elbo = np.asarray(result.elbo, dtype=float).reshape(-1)
    if len(elbo) < 3:
        return False
    delta = elbo[-1] - elbo[-2]
    return bool(np.isfinite(delta) and abs(delta) < min_tol)


def write_stats(output: Path, stats: SuShiEStats) -> None:
    """Write one deterministic JSON status record."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(stats.model_dump(exclude_none=True), sort_keys=True) + "\n"
    )
