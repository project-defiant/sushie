from types import SimpleNamespace

import anndata as ad
import jax.numpy as jnp
import pandas as pd

from sushie.anndata_output import write_anndata
from sushie.infer_ss import PreparationOutcome
from sushie.preparation import PreparedSushie


def test_write_anndata_preserves_posteriors_and_provenance(tmp_path) -> None:
    result = SimpleNamespace(
        posteriors=SimpleNamespace(
            alpha=jnp.array([[0.8, 0.2], [0.1, 0.9]]),
            log_bf=jnp.array([[2.0, 1.0], [0.5, 2.5]]),
            post_mean=jnp.array(
                [
                    [[0.08, 0.04], [0.02, 0.01]],
                    [[0.01, 0.02], [0.09, 0.12]],
                ]
            ),
            post_mean_sq=jnp.ones((2, 2, 2, 2)),
            weighted_sum_covar=jnp.ones((2, 2, 2)),
            kl=jnp.array([0.2, 0.3]),
        ),
        pip_all=jnp.array([0.82, 0.92]),
        pip_cs=jnp.array([0.8, 0.9]),
        priors=SimpleNamespace(
            pi=jnp.array([0.5, 0.5]),
            resid_var=jnp.array([[1.0], [0.9]]),
            effect_covar=jnp.array([[0.1, 0.02], [0.02, 0.2]]),
        ),
        cs=pd.DataFrame(
            {
                "CSIndex": [1, 1],
                "SNPIndex": [0, 1],
                "alpha": [0.8, 0.2],
                "c_alpha": [0.8, 1.0],
            }
        ),
        alphas=pd.DataFrame(
            {
                "index": [0, 1],
                "alpha_l1": [0.8, 0.2],
                "kept_l1": [1, 1],
            }
        ),
        elbo=jnp.array([-jnp.inf, 1.0, 1.00001]),
        elbo_increase=True,
        l_order=jnp.array([0, 1]),
    )
    prepared = PreparedSushie(
        snps=None,
        data=None,
        variants=pd.DataFrame(
            {
                "variantId": ["1_100_A_G", "1_110_C_T"],
                "chrom": ["1", "1"],
                "pos": [100, 110],
            }
        ),
        study_ids=["STUDY_A", "STUDY_B"],
        ancestries=["eur", "afr"],
        sample_sizes=[1000, 2000],
        outcome=PreparationOutcome(None, None, "SUCCESS", None, 2, 2, 2, 2),
    )
    output = tmp_path / "fit.h5ad"

    write_anndata(
        result,
        prepared,
        run_id="RUN_A",
        fine_mapping_locus_set_id="LOCUS_A",
        parameters={"rho": 0.1, "L": 2},
        output=output,
    )

    fit = ad.read_h5ad(output)
    assert fit.shape == (2, 2)
    assert fit.obs_names.tolist() == ["component_0", "component_1"]
    assert fit.var_names.tolist() == ["1_100_A_G", "1_110_C_T"]
    assert "posteriorMean__eur" in fit.layers
    assert "posteriorMean__afr" in fit.layers
    assert "posteriorSecondMoment__eur__afr" in fit.layers
    assert "logBayesFactor" in fit.layers
    assert fit.uns["runId"] == "RUN_A"
    assert fit.uns["fineMappingLocusSetId"] == "LOCUS_A"
    assert fit.uns["studyIds"].tolist() == ["STUDY_A", "STUDY_B"]
    assert fit.uns["methodParameters"]["rho"] == 0.1
    assert fit.uns["converged"] is True
    assert fit.uns["credibleSets"].shape == (2, 4)
    assert fit.uns["priorEffectCovariance"].shape == (2, 2)
    assert fit.var["priorProbability"].tolist() == [0.5, 0.5]
