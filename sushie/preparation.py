"""Prepare pipeline parquet and JSONL inputs for SuShiE summary statistics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, ConfigDict
from scipy.stats import norm

from . import infer_ss, io
from .models import PipelineInputs, StudyMetadata


class PreparedSushie(BaseModel):
    """Aligned strict-intersection inputs and variant metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    snps: pd.DataFrame | None
    data: io.ssData | None
    variants: pd.DataFrame
    study_ids: list[str]
    ancestries: list[str]
    sample_sizes: list[int]
    outcome: infer_ss.PreparationOutcome


def prepare_inputs(inputs: PipelineInputs) -> PreparedSushie:
    """Read inputs, build square LD matrices, and apply strict intersection."""

    metadata = _read_metadata(inputs.study_metadata)
    locus = _read_parquet(inputs.fine_mapping_locus_set)
    _validate_locus(locus, inputs.fine_mapping_locus_set_id, metadata)
    study_variants = _flatten_locus(locus)
    gwas = [_gwas_frame(study_variants[row.studyId]) for row in metadata]
    pairwise_ld = _read_pairwise_ld(inputs.multi_ancestry_pairwise_ld)
    lds = [
        _ld_frame(
            pairwise_ld,
            ancestry=row.ancestry,
            variant_ids=list(study_variants[row.studyId]),
        )
        for row in metadata
    ]
    outcome = infer_ss.prepare_sushie_ss_data_with_stats(
        gwas=gwas,
        lds=lds,
        sample_size=[row.sampleSize for row in metadata],
        gwas_sig=1.0,
    )
    variants = _variant_frame(study_variants, outcome.snps)
    return PreparedSushie(
        snps=outcome.snps,
        data=outcome.data,
        variants=variants,
        study_ids=[row.studyId for row in metadata],
        ancestries=[row.ancestry for row in metadata],
        sample_sizes=[row.sampleSize for row in metadata],
        outcome=outcome,
    )


def _read_parquet(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise ValueError(f"Parquet directory contains no parquet files: {path}")
        return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    return pd.read_parquet(path)


def _read_pairwise_ld(path: Path) -> pl.DataFrame:
    parquet_input: Path | str = (
        str(path / "**" / "*.parquet") if path.is_dir() else path
    )
    required = {"ancestry", "variantIdI", "variantIdJ", "r"}
    try:
        return pl.read_parquet(parquet_input, columns=sorted(required))
    except pl.exceptions.ColumnNotFoundError as error:
        raise ValueError(
            f"MultiAncestryPairwiseLD is missing required columns: {sorted(required)}"
        ) from error


def _read_metadata(path: Path) -> list[StudyMetadata]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    metadata = [StudyMetadata.model_validate(record) for record in records]
    if not metadata:
        raise ValueError("Study metadata must contain at least one row")
    if len({row.studyId for row in metadata}) != len(metadata):
        raise ValueError("Study metadata contains duplicate studyId values")
    if len({row.ancestry for row in metadata}) != len(metadata):
        raise ValueError("Study metadata must contain one study per ancestry")
    return sorted(metadata, key=lambda row: row.studyId)


def _validate_locus(
    locus: pd.DataFrame, locus_set_id: str, metadata: list[StudyMetadata]
) -> None:
    required = {"fineMappingLocusSetId", "studyId", "locus"}
    missing = required - set(locus.columns)
    if missing:
        raise ValueError(f"FineMappingLocusSet is missing columns: {sorted(missing)}")
    if set(locus["fineMappingLocusSetId"].dropna()) != {locus_set_id}:
        raise ValueError("FineMappingLocusSet contains an unexpected locus-set ID")
    if set(locus["studyId"].dropna()) != {row.studyId for row in metadata}:
        raise ValueError("Study metadata and locus studyId values differ")


def _flatten_locus(locus: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    studies: dict[str, dict[str, dict[str, Any]]] = {}
    for _, row in locus[["studyId", "locus"]].iterrows():
        study_id = str(row["studyId"])
        study = studies.setdefault(study_id, {})
        if row["locus"] is None:
            raise ValueError(f"Study locus has no variants: {study_id}")
        for variant in row["locus"]:
            variant_id = str(variant["variantId"])
            if variant_id in study:
                raise ValueError(f"Duplicate variant in study locus: {variant_id}")
            beta = float(variant["beta"])
            se = float(variant["standardError"])
            if not np.isfinite(beta) or not np.isfinite(se) or se <= 0:
                raise ValueError(f"Invalid beta or standardError for {variant_id}")
            chromosome, position, ref, alt = _parse_variant_id(variant_id)
            study[variant_id] = {
                "variantId": variant_id,
                "chrom": chromosome,
                "pos": position,
                "a0": ref,
                "a1": alt,
                "z": beta / se,
                "beta": beta,
                "standardError": se,
                "pValueMantissa": variant.get("pValueMantissa"),
                "pValueExponent": variant.get("pValueExponent"),
            }
    return studies


def _gwas_frame(rows: dict[str, dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows.values()).rename(columns={"variantId": "snp"})


def _ld_frame(
    ld: pl.DataFrame, *, ancestry: str, variant_ids: list[str]
) -> pd.DataFrame:
    selected = ld.filter(pl.col("ancestry") == ancestry).select(
        "variantIdI", "variantIdJ", "r"
    )
    observed = set(
        pl.concat(
            [
                selected.select(pl.col("variantIdI").alias("variantId")),
                selected.select(pl.col("variantIdJ").alias("variantId")),
            ]
        )
        .get_column("variantId")
        .unique()
        .to_list()
    )
    ordered_variants = [
        variant_id for variant_id in variant_ids if variant_id in observed
    ]
    variant_index = pd.Index(ordered_variants)
    matrix = pd.DataFrame(
        np.eye(len(variant_index), dtype=np.float32),
        index=variant_index,
        columns=variant_index,
    )
    if not ordered_variants:
        return matrix

    index_i = pl.DataFrame(
        {"variantIdI": ordered_variants, "indexI": range(len(ordered_variants))}
    )
    index_j = pl.DataFrame(
        {"variantIdJ": ordered_variants, "indexJ": range(len(ordered_variants))}
    )
    indexed = selected.join(index_i, on="variantIdI", how="inner").join(
        index_j, on="variantIdJ", how="inner"
    )
    invalid = indexed.filter(
        ~pl.col("r").is_finite() | (pl.col("r") < -1) | (pl.col("r") > 1)
    )
    if invalid.height:
        row = invalid.row(0, named=True)
        raise ValueError(
            f"Invalid LD value for {row['variantIdI']}/{row['variantIdJ']}: {row['r']}"
        )
    first_indices = indexed.get_column("indexI").to_numpy()
    second_indices = indexed.get_column("indexJ").to_numpy()
    values = indexed.get_column("r").cast(pl.Float32).to_numpy()
    matrix_values = matrix.values
    matrix_values[first_indices, second_indices] = values
    matrix_values[second_indices, first_indices] = values
    return matrix


def _variant_frame(
    studies: dict[str, dict[str, dict[str, Any]]], snps: pd.DataFrame | None
) -> pd.DataFrame:
    if snps is None:
        return pd.DataFrame(columns=pd.Index(["variantId", "chrom", "pos"]))
    first = next(iter(studies.values()))
    rows = []
    for raw_variant_id in snps["snp"]:
        variant_id = str(raw_variant_id)
        contributions = [
            (
                float(study[variant_id]["beta"]),
                float(study[variant_id]["standardError"]),
            )
            for study in studies.values()
        ]
        weights = np.asarray(
            [
                1 / (standard_error * standard_error)
                for _, standard_error in contributions
            ]
        )
        beta = float(
            np.sum(weights * np.asarray([value for value, _ in contributions]))
            / np.sum(weights)
        )
        standard_error = float(np.sqrt(1 / np.sum(weights)))
        z_score = beta / standard_error
        log10_p = (math.log(2) + norm.logsf(abs(z_score))) / math.log(10)
        exponent = math.floor(log10_p)
        source = first[variant_id]
        rows.append(
            {
                **source,
                "beta": beta,
                "standardError": standard_error,
                "zScore": z_score,
                "pValueMantissa": 10 ** (log10_p - exponent),
                "pValueExponent": exponent,
            }
        )
    return pd.DataFrame(rows)


def _parse_variant_id(variant_id: str) -> tuple[str, int, str, str]:
    fields = variant_id.split("_")
    if len(fields) < 4:
        raise ValueError(f"Cannot parse variantId: {variant_id}")
    return fields[0], int(fields[1]), fields[2], fields[3]
