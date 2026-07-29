"""Pipeline-facing Typer command for SuShiE summary-statistic fine-mapping."""

# Ruff B008 is incompatible with Typer's documented option declaration style.
# ruff: noqa: B008

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pandas as pd
import typer
from loguru import logger

from .anndata_output import write_anndata
from .infer_ss import infer_sushie_ss
from .models import PipelineInputs, PurityMethod, RunParameters
from .outputs import study_locus_from_result, write_study_locus
from .preparation import PreparedSushie, prepare_inputs
from .stats import SuShiEStats, stats_from_preparation, stats_from_result, write_stats

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    fine_mapping_locus_set: Path = typer.Option(..., exists=True),
    multi_ancestry_pairwise_ld: Path = typer.Option(..., exists=True),
    study_metadata: Path = typer.Option(..., exists=True),
    run_id: str = typer.Option(...),
    fine_mapping_locus_set_id: str = typer.Option(...),
    study_locus_output: Path = typer.Option(...),
    extended_results_output: Path = typer.Option(...),
    stats_output: Path = typer.Option(...),
    rho: float = typer.Option(0.1, min=-1, max=1),
    L: int = typer.Option(10, min=1),
    max_iter: int = typer.Option(500, min=1),
    min_tol: float = typer.Option(1e-4, min=1e-12),
    threshold: float = typer.Option(0.95, min=1e-6, max=0.999999),
    purity: float = typer.Option(0.5, min=1e-6, max=0.999999),
    purity_method: str = typer.Option("weighted"),
    max_select: int = typer.Option(250, min=1),
    min_snps: int = typer.Option(100, min=1),
    no_update: bool = typer.Option(False),
    no_reorder: bool = typer.Option(False),
    seed: int = typer.Option(12345),
) -> None:
    """Run SuShiE for one fine-mapping locus set."""

    inputs = PipelineInputs(
        run_id=run_id,
        fine_mapping_locus_set_id=fine_mapping_locus_set_id,
        fine_mapping_locus_set=fine_mapping_locus_set,
        multi_ancestry_pairwise_ld=multi_ancestry_pairwise_ld,
        study_metadata=study_metadata,
        study_locus_output=study_locus_output,
        extended_results_output=extended_results_output,
        stats_output=stats_output,
    )
    try:
        parameters = RunParameters(
            rho=rho,
            L=L,
            max_iter=max_iter,
            min_tol=min_tol,
            threshold=threshold,
            purity=purity,
            purity_method=cast(PurityMethod, purity_method),
            max_select=max_select,
            min_snps=min_snps,
            no_update=no_update,
            no_reorder=no_reorder,
            seed=seed,
        )
    except ValueError as error:
        _write_failure(inputs, str(error))
        logger.error("Invalid SuShiE parameters: {}", error)
        raise typer.Exit(code=2) from error
    try:
        prepared = prepare_inputs(inputs)
    except (OSError, TypeError, ValueError) as error:
        _write_failure(inputs, str(error))
        logger.error("SuShiE input preparation failed: {}", error)
        raise typer.Exit(code=1) from error

    if prepared.outcome.status != "SUCCESS":
        write_stats(
            stats_output,
            stats_from_preparation(
                prepared.outcome,
                run_id=run_id,
                fine_mapping_locus_set_id=fine_mapping_locus_set_id,
            ),
        )
        logger.warning("SuShiE input is not reportable: {}", prepared.outcome.reason)
        return
    if prepared.outcome.final_variants < min_snps:
        reason = (
            f"Only {prepared.outcome.final_variants} variants remain after the "
            f"shared-variant and LD intersection; at least {min_snps} are required"
        )
        write_stats(
            stats_output,
            SuShiEStats(
                runId=run_id,
                fineMappingLocusSetId=fine_mapping_locus_set_id,
                status="INSUFFICIENT_VARIANTS",
                inputVariants=prepared.outcome.input_variants,
                sharedVariants=prepared.outcome.shared_variants,
                ldValidVariants=prepared.outcome.ld_valid_variants,
                finalVariants=prepared.outcome.final_variants,
                reason=reason,
            ),
        )
        logger.warning(reason)
        return
    if prepared.data is None or prepared.snps is None:
        _write_failure(inputs, "SuShiE preparation returned no numerical data")
        return

    correlations = [parameters.rho] * math.comb(len(prepared.ancestries), 2)
    try:
        run_inference = cast(Any, infer_sushie_ss)
        result = run_inference(
            lds=prepared.data.lds,
            ns=prepared.data.ns,
            zs=prepared.data.zs,
            pi=prepared.data.pi,
            rho=correlations,
            **parameters.model_dump(exclude={"rho"}),
        )
    except (TypeError, ValueError) as error:
        _write_failure(inputs, str(error))
        logger.error("SuShiE inference failed: {}", error)
        raise typer.Exit(code=1) from error

    stats = stats_from_result(
        result,
        run_id=run_id,
        fine_mapping_locus_set_id=fine_mapping_locus_set_id,
        min_tol=parameters.min_tol,
    )
    if not stats.converged:
        write_stats(stats_output, stats)
        logger.warning("SuShiE fit did not converge")
        return
    variants = prepared.variants.rename(
        columns={"chrom": "chromosome", "pos": "position"}
    )
    output = study_locus_from_result(
        result,
        variants,
        run_id=run_id,
        fine_mapping_locus_set_id=fine_mapping_locus_set_id,
        study_id="|".join(prepared.study_ids),
        chromosome=str(variants.iloc[0]["chromosome"]),
        locus_start=int(variants["position"].min()),
        locus_end=int(variants["position"].max()),
    )
    if output.empty:
        _write_failure(inputs, "SuShiE fit has no reportable credible sets")
        return
    try:
        _write_outputs_atomically(
            result=result,
            prepared=prepared,
            output=output,
            parameters=parameters,
            inputs=inputs,
        )
    except (OSError, ValueError) as error:
        _write_failure(inputs, str(error))
        logger.error("SuShiE output writing failed: {}", error)
        raise typer.Exit(code=1) from error
    write_stats(stats_output, stats)


def _write_failure(inputs: PipelineInputs, reason: str) -> None:
    write_stats(
        inputs.stats_output,
        SuShiEStats(
            runId=inputs.run_id,
            fineMappingLocusSetId=inputs.fine_mapping_locus_set_id,
            status="FAILED",
            reason=reason,
        ),
    )


def _write_outputs_atomically(
    *,
    result: Any,
    prepared: PreparedSushie,
    output: pd.DataFrame,
    parameters: RunParameters,
    inputs: PipelineInputs,
) -> None:
    temporary_study_locus = _temporary_path(inputs.study_locus_output)
    temporary_extended_results = _temporary_path(inputs.extended_results_output)
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        write_study_locus(output, temporary_study_locus)
        write_anndata(
            result,
            prepared,
            run_id=inputs.run_id,
            fine_mapping_locus_set_id=inputs.fine_mapping_locus_set_id,
            parameters=parameters.model_dump(),
            output=temporary_extended_results,
        )
        for target in (
            inputs.study_locus_output,
            inputs.extended_results_output,
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = _temporary_path(target)
                target.replace(backup)
                backups.append((target, backup))
        temporary_study_locus.replace(inputs.study_locus_output)
        published.append(inputs.study_locus_output)
        temporary_extended_results.replace(inputs.extended_results_output)
        published.append(inputs.extended_results_output)
    except BaseException:
        for target in published:
            target.unlink(missing_ok=True)
        for target, backup in reversed(backups):
            backup.replace(target)
        raise
    finally:
        temporary_study_locus.unlink(missing_ok=True)
        temporary_extended_results.unlink(missing_ok=True)
        for _, backup in backups:
            backup.unlink(missing_ok=True)


def _temporary_path(output: Path) -> Path:
    return output.with_name(f".{output.name}.{uuid4().hex}.tmp{output.suffix}")
