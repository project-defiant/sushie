"""Validated models for the SuShiE pipeline boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PurityMethod = Literal["weighted", "max", "min"]


class StudyMetadata(BaseModel):
    """Metadata required for one ancestry-specific SuShiE input."""

    model_config = ConfigDict(extra="forbid")

    studyId: str = Field(min_length=1)
    ancestry: str = Field(min_length=1)
    sampleSize: int = Field(gt=0)


class PipelineInputs(BaseModel):
    """Input and output paths for one summary-statistic locus-set run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    fine_mapping_locus_set_id: str = Field(min_length=1)
    fine_mapping_locus_set: Path
    multi_ancestry_pairwise_ld: Path
    study_metadata: Path
    study_locus_output: Path
    extended_results_output: Path
    stats_output: Path


class RunParameters(BaseModel):
    """User-configurable SuShiE parameters exposed by the pipeline CLI."""

    model_config = ConfigDict(extra="forbid")

    rho: float = Field(default=0.1, ge=-1, le=1)
    L: int = Field(default=10, gt=0)
    max_iter: int = Field(default=500, gt=0)
    min_tol: float = Field(default=1e-4, gt=0)
    threshold: float = Field(default=0.95, gt=0, lt=1)
    purity: float = Field(default=0.5, gt=0, lt=1)
    purity_method: PurityMethod = "weighted"
    max_select: int = Field(default=250, gt=0)
    min_snps: int = Field(default=100, gt=0)
    no_update: bool = False
    no_reorder: bool = False
    seed: int = 12345
