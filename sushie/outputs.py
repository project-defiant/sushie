"""Conversion of SuShiE results into the shared StudyLocus contract."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

STUDY_LOCUS_COLUMNS = (
    "studyLocusId",
    "studyId",
    "variantId",
    "chromosome",
    "position",
    "beta",
    "sampleSize",
    "pValueMantissa",
    "pValueExponent",
    "effectAlleleFrequencyFromSource",
    "standardError",
    "qualityControls",
    "locusStart",
    "locusEnd",
    "locus",
)


def _optional(row: Mapping[str, Any], name: str) -> Any:
    value = row.get(name)
    return None if pd.isna(value) else value


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
    rows: list[dict[str, Any]] = []
    for component, group in cs.groupby("CSIndex", sort=True):
        component_index = int(component) - 1
        group = group.sort_values(["alpha", "SNPIndex"], ascending=[False, True])
        lead_index = int(group.iloc[0]["SNPIndex"])
        if lead_index < 0 or lead_index >= len(variants):
            raise ValueError(f"SNPIndex {lead_index} is outside the variant table")

        nested: list[dict[str, Any]] = []
        for _, member in group.iterrows():
            index = int(member["SNPIndex"])
            variant = variants.iloc[index].to_dict()
            alpha = float(member["alpha"])
            cumulative = float(member["c_alpha"])
            nested.append(
                {
                    "is95CredibleSet": cumulative >= 0.95,
                    "is99CredibleSet": cumulative >= 0.99,
                    "logBF": float(log_bf[component_index, index]),
                    "posteriorProbability": alpha,
                    "variantId": variant["variantId"],
                    "pValueMantissa": _optional(variant, "pValueMantissa"),
                    "pValueExponent": _optional(variant, "pValueExponent"),
                    "beta": _optional(variant, "beta"),
                    "standardError": _optional(variant, "standardError"),
                    "r2Overall": None,
                }
            )

        lead = variants.iloc[lead_index].to_dict()
        stable_key = f"{run_id}:{fine_mapping_locus_set_id}:sushie:{component_index}:{lead['variantId']}"
        rows.append(
            {
                "studyLocusId": hashlib.md5(stable_key.encode()).hexdigest(),
                "studyId": study_id,
                "variantId": lead["variantId"],
                "chromosome": chromosome,
                "position": _optional(lead, "position"),
                "beta": _optional(lead, "beta"),
                "sampleSize": _optional(lead, "sampleSize"),
                "pValueMantissa": _optional(lead, "pValueMantissa"),
                "pValueExponent": _optional(lead, "pValueExponent"),
                "effectAlleleFrequencyFromSource": _optional(
                    lead, "effectAlleleFrequencyFromSource"
                ),
                "standardError": _optional(lead, "standardError"),
                "qualityControls": [],
                "locusStart": locus_start,
                "locusEnd": locus_end,
                "locus": nested,
            }
        )

    return pd.DataFrame(rows, columns=STUDY_LOCUS_COLUMNS)
