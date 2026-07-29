"""Write complete SuShiE fits as AnnData objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from .outputs import component_log_bayes_factors
from .preparation import PreparedSushie
from .stats import result_converged


def write_anndata(
    result: Any,
    prepared: PreparedSushie,
    *,
    run_id: str,
    fine_mapping_locus_set_id: str,
    parameters: dict[str, Any],
    output: Path,
) -> None:
    """Persist posterior arrays and application provenance in H5AD."""

    alpha = np.asarray(result.posteriors.alpha, dtype=np.float32)
    n_components, _ = alpha.shape
    obs = pd.DataFrame(
        {
            "componentIndex": np.arange(n_components),
            "componentLogBF": component_log_bayes_factors(result),
        },
        index=pd.Index([f"component_{index}" for index in range(n_components)]),
    )
    variants = prepared.variants
    var = pd.DataFrame(
        {
            "variantId": variants["variantId"].tolist(),
            "chromosome": variants["chrom"].astype(str).tolist(),
            "position": variants["pos"].astype(int).tolist(),
            "priorProbability": np.asarray(result.priors.pi, dtype=np.float32),
            "pipAll": np.asarray(result.pip_all, dtype=np.float32),
            "pipCredibleSets": np.asarray(result.pip_cs, dtype=np.float32),
        },
        index=pd.Index(variants["variantId"].tolist()),
    )
    post_mean = np.asarray(result.posteriors.post_mean, dtype=np.float32)
    layers = {
        f"posteriorMean__{ancestry}": post_mean[:, :, index]
        for index, ancestry in enumerate(prepared.ancestries)
    }
    layers["logBayesFactor"] = np.asarray(result.posteriors.log_bf, dtype=np.float32)
    post_mean_sq = np.asarray(result.posteriors.post_mean_sq, dtype=np.float32)
    for first_index, first_ancestry in enumerate(prepared.ancestries):
        for second_index, second_ancestry in enumerate(prepared.ancestries):
            layers[f"posteriorSecondMoment__{first_ancestry}__{second_ancestry}"] = (
                post_mean_sq[:, :, first_index, second_index]
            )
    adata = ad.AnnData(X=alpha, obs=obs, var=var, layers=layers)
    adata.uns.update(
        {
            "runId": run_id,
            "fineMappingLocusSetId": fine_mapping_locus_set_id,
            "studyIds": prepared.study_ids,
            "ancestries": prepared.ancestries,
            "sampleSizes": prepared.sample_sizes,
            "converged": result_converged(
                result, float(parameters.get("min_tol", 1e-4))
            ),
            "methodParameters": parameters,
            "inputMode": "summary-statistics",
            "priorResidualVariance": np.asarray(
                result.priors.resid_var, dtype=np.float32
            ),
            "priorEffectCovariance": np.asarray(
                result.priors.effect_covar, dtype=np.float32
            ),
            "posteriorWeightedSumCovariance": np.asarray(
                result.posteriors.weighted_sum_covar, dtype=np.float32
            ),
            "posteriorKL": np.asarray(result.posteriors.kl, dtype=np.float32),
            "elbo": np.asarray(result.elbo, dtype=np.float64),
            "componentOrder": np.asarray(result.l_order, dtype=np.int32),
            "credibleSets": result.cs.reset_index(drop=True),
            "fullCredibleSets": result.alphas.reset_index(drop=True),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)
