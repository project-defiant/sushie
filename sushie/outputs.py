"""Conversion of SuShiE results into the shared StudyLocus contract."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict
from scipy.special import logsumexp


class StudyLocusVariant(BaseModel):
    """Nested Gentropy-compatible variant in a SuShiE credible set."""

    model_config = ConfigDict(extra="forbid")

    variantId: str
    posteriorProbability: float
    is95CredibleSet: bool
    is99CredibleSet: None = None
    pValueMantissa: float
    pValueExponent: int
    beta: float
    standardError: float
    logBF: float
    r2Overall: None = None


class StudyLocusRecord(BaseModel):
    """Flat Gentropy-compatible parent row for one SuShiE credible set."""

    model_config = ConfigDict(extra="forbid")

    studyLocusId: str
    studyType: None = None
    variantId: str
    chromosome: str
    position: int
    region: None = None
    studyId: str
    beta: float
    zScore: float
    pValueMantissa: float
    pValueExponent: int
    effectAlleleFrequencyFromSource: None = None
    standardError: float
    subStudyDescription: None = None
    qualityControls: list[str]
    finemappingMethod: str
    credibleSetIndex: int
    credibleSetlog10BF: float
    purityMeanR2: None = None
    purityMinR2: float | None
    locusStart: int
    locusEnd: int
    sampleSize: None = None
    ldSet: None = None
    locus: list[StudyLocusVariant]
    confidence: None = None
    isTransQtl: None = None


STUDY_LOCUS_COLUMNS = tuple(StudyLocusRecord.model_fields)
STUDY_LOCUS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("studyLocusId", pa.string(), nullable=False),
        pa.field("studyType", pa.string()),
        pa.field("variantId", pa.string(), nullable=False),
        pa.field("chromosome", pa.string()),
        pa.field("position", pa.int32()),
        pa.field("region", pa.string()),
        pa.field("studyId", pa.string(), nullable=False),
        pa.field("beta", pa.float64()),
        pa.field("zScore", pa.float64()),
        pa.field("pValueMantissa", pa.float32()),
        pa.field("pValueExponent", pa.int32()),
        pa.field("effectAlleleFrequencyFromSource", pa.float32()),
        pa.field("standardError", pa.float64()),
        pa.field("subStudyDescription", pa.string()),
        pa.field(
            "qualityControls",
            pa.list_(pa.field("element", pa.string(), nullable=False)),
        ),
        pa.field("finemappingMethod", pa.string()),
        pa.field("credibleSetIndex", pa.int32()),
        pa.field("credibleSetlog10BF", pa.float64()),
        pa.field("purityMeanR2", pa.float64()),
        pa.field("purityMinR2", pa.float64()),
        pa.field("locusStart", pa.int32()),
        pa.field("locusEnd", pa.int32()),
        pa.field("sampleSize", pa.int32()),
        pa.field(
            "ldSet",
            pa.list_(
                pa.struct(
                    [
                        pa.field("tagVariantId", pa.string()),
                        pa.field("r2Overall", pa.float64()),
                    ]
                )
            ),
        ),
        pa.field(
            "locus",
            pa.list_(
                pa.struct(
                    [
                        pa.field("is95CredibleSet", pa.bool_()),
                        pa.field("is99CredibleSet", pa.bool_()),
                        pa.field("logBF", pa.float64()),
                        pa.field("posteriorProbability", pa.float64()),
                        pa.field("variantId", pa.string()),
                        pa.field("pValueMantissa", pa.float32()),
                        pa.field("pValueExponent", pa.int32()),
                        pa.field("beta", pa.float64()),
                        pa.field("standardError", pa.float64()),
                        pa.field("r2Overall", pa.float64()),
                    ]
                )
            ),
        ),
        pa.field("confidence", pa.string()),
        pa.field("isTransQtl", pa.bool_()),
    ]
)


def study_locus_from_result(
    result: Any,
    variants: pd.DataFrame,
    *,
    run_id: str,
    fine_mapping_locus_set_id: str,
    study_id: str,
    chromosome: str,
    locus_start: int,
    locus_end: int,
) -> pd.DataFrame:
    """Build one StudyLocus row per retained SuShiE credible set.

    ``variants`` must be in the same SNP order used by the SuShiE result and
    contain a ``variantId`` column. The credible-set ``alpha`` is preserved as
    nested ``posteriorProbability``; marginal PIP values remain separate.
    """

    required = {"variantId"}
    missing = required - set(variants.columns)
    if missing:
        raise ValueError(f"Missing variant columns: {sorted(missing)}")

    cs = result.cs.copy()
    log_bf = np.asarray(result.posteriors.log_bf)
    component_log_bf = component_log_bayes_factors(result)
    records: list[StudyLocusRecord] = []
    for component, group in cs.groupby("CSIndex", sort=True):
        component_index = int(component) - 1
        group = group.sort_values(["alpha", "SNPIndex"], ascending=[False, True])
        lead_index = int(group.iloc[0]["SNPIndex"])
        if lead_index < 0 or lead_index >= len(variants):
            raise ValueError(f"SNPIndex {lead_index} is outside the variant table")

        nested: list[StudyLocusVariant] = []
        for _, member in group.iterrows():
            index = int(member["SNPIndex"])
            variant = variants.iloc[index].to_dict()
            nested.append(
                StudyLocusVariant(
                    is95CredibleSet=True,
                    logBF=float(log_bf[component_index, index]),
                    posteriorProbability=float(member["alpha"]),
                    variantId=str(variant["variantId"]),
                    pValueMantissa=float(variant["pValueMantissa"]),
                    pValueExponent=int(variant["pValueExponent"]),
                    beta=float(variant["beta"]),
                    standardError=float(variant["standardError"]),
                )
            )

        lead = variants.iloc[lead_index].to_dict()
        purity = _component_purity(result, component_index)
        stable_key = f"{fine_mapping_locus_set_id}|{lead['variantId']}|SuShiE"
        records.append(
            StudyLocusRecord(
                studyLocusId=hashlib.md5(stable_key.encode()).hexdigest(),
                studyId=study_id,
                variantId=str(lead["variantId"]),
                chromosome=chromosome,
                position=int(lead["position"]),
                beta=float(lead["beta"]),
                zScore=float(lead["zScore"]),
                pValueMantissa=float(lead["pValueMantissa"]),
                pValueExponent=int(lead["pValueExponent"]),
                standardError=float(lead["standardError"]),
                qualityControls=[],
                finemappingMethod="SuShiE",
                credibleSetIndex=component_index,
                credibleSetlog10BF=float(component_log_bf[component_index])
                / math.log(10),
                purityMinR2=purity * purity if purity is not None else None,
                locusStart=locus_start,
                locusEnd=locus_end,
                locus=nested,
            )
        )

    return pd.DataFrame(
        [record.model_dump() for record in records],
        columns=pd.Index(STUDY_LOCUS_COLUMNS),
    )


def _component_purity(result: Any, component_index: int) -> float | None:
    column = f"purity_l{component_index + 1}"
    if not hasattr(result, "alphas") or column not in result.alphas:
        return None
    purity = float(result.alphas.iloc[0][column])
    return purity if math.isfinite(purity) else None


def component_log_bayes_factors(result: Any) -> np.ndarray:
    """Return prior-weighted natural-log Bayes factors for every component."""

    log_bf = np.asarray(result.posteriors.log_bf, dtype=float)
    prior = np.asarray(result.priors.pi, dtype=float).reshape(-1)
    if prior.shape != (log_bf.shape[1],) or np.any(prior <= 0):
        raise ValueError("SuShiE prior probabilities do not match the variants")
    prior = prior / prior.sum()
    return np.asarray(logsumexp(log_bf + np.log(prior), axis=1))


def write_study_locus(output: pd.DataFrame, path: Path) -> None:
    """Write reportable SuShiE credible sets as a flat parquet file."""

    if output.empty:
        raise ValueError("Cannot write StudyLocus output without credible sets")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        output.to_dict(orient="records"), schema=STUDY_LOCUS_ARROW_SCHEMA
    )
    pq.write_table(table, path)
