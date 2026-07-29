import json
from types import SimpleNamespace

import jax.numpy as jnp
import pandas as pd
from typer.main import get_command
from typer.testing import CliRunner

from sushie import io
from sushie.infer_ss import PreparationOutcome
from sushie.pipeline_cli import app
from sushie.preparation import PreparedSushie


def test_pipeline_cli_help_exposes_pipeline_contract() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    option_names = {parameter.name for parameter in get_command(app).params}
    assert "fine_mapping_locus_set" in option_names
    assert "multi_ancestry_pairwise_ld" in option_names
    assert "extended_results_output" in option_names


def test_pipeline_cli_reports_empty_intersection_without_outputs(
    tmp_path, monkeypatch
) -> None:
    locus = tmp_path / "locus.parquet"
    ld = tmp_path / "ld.parquet"
    metadata = tmp_path / "metadata.jsonl"
    for path in (locus, ld, metadata):
        path.touch()
    study_locus = tmp_path / "study_locus.parquet"
    extended = tmp_path / "fit.h5ad"
    stats = tmp_path / "stats.json"
    monkeypatch.setattr(
        "sushie.pipeline_cli.prepare_inputs",
        lambda _inputs: PreparedSushie(
            snps=None,
            data=None,
            variants=pd.DataFrame(),
            study_ids=["STUDY_A", "STUDY_B"],
            ancestries=["eur", "afr"],
            sample_sizes=[1000, 2000],
            outcome=PreparationOutcome(
                None,
                None,
                "EMPTY_INTERSECTION",
                "No variants are shared",
                4,
                0,
                0,
                0,
            ),
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "--fine-mapping-locus-set",
            str(locus),
            "--multi-ancestry-pairwise-ld",
            str(ld),
            "--study-metadata",
            str(metadata),
            "--run-id",
            "RUN_A",
            "--fine-mapping-locus-set-id",
            "LOCUS_A",
            "--study-locus-output",
            str(study_locus),
            "--extended-results-output",
            str(extended),
            "--stats-output",
            str(stats),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(stats.read_text())["status"] == "EMPTY_INTERSECTION"
    assert not study_locus.exists()
    assert not extended.exists()


def test_pipeline_cli_reports_too_few_shared_variants_without_inference(
    tmp_path, monkeypatch
) -> None:
    locus = tmp_path / "locus.parquet"
    ld = tmp_path / "ld.parquet"
    metadata = tmp_path / "metadata.jsonl"
    for path in (locus, ld, metadata):
        path.touch()
    stats = tmp_path / "stats.json"
    monkeypatch.setattr(
        "sushie.pipeline_cli.prepare_inputs",
        lambda _inputs: PreparedSushie(
            snps=pd.DataFrame({"snp": ["1_100_A_G"]}),
            data=None,
            variants=pd.DataFrame(),
            study_ids=["STUDY_A", "STUDY_B"],
            ancestries=["eur", "afr"],
            sample_sizes=[1000, 2000],
            outcome=PreparationOutcome(None, None, "SUCCESS", None, 4, 2, 1, 1),
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "--fine-mapping-locus-set",
            str(locus),
            "--multi-ancestry-pairwise-ld",
            str(ld),
            "--study-metadata",
            str(metadata),
            "--run-id",
            "RUN_A",
            "--fine-mapping-locus-set-id",
            "LOCUS_A",
            "--study-locus-output",
            str(tmp_path / "study_locus.parquet"),
            "--extended-results-output",
            str(tmp_path / "fit.h5ad"),
            "--stats-output",
            str(stats),
            "--min-snps",
            "2",
        ],
    )

    assert result.exit_code == 0
    status = json.loads(stats.read_text())
    assert status["status"] == "INSUFFICIENT_VARIANTS"
    assert status["finalVariants"] == 1


def test_pipeline_cli_writes_shared_outputs_for_reportable_fit(
    tmp_path, monkeypatch
) -> None:
    locus = tmp_path / "locus.parquet"
    ld = tmp_path / "ld.parquet"
    metadata = tmp_path / "metadata.jsonl"
    for path in (locus, ld, metadata):
        path.touch()
    study_locus = tmp_path / "study_locus.parquet"
    extended = tmp_path / "fit.h5ad"
    stats = tmp_path / "stats.json"
    variants = pd.DataFrame(
        {
            "variantId": ["1_100_A_G", "1_110_C_T"],
            "chrom": ["1", "1"],
            "pos": [100, 110],
            "beta": [0.25, 0.15],
            "standardError": [0.05, 0.05],
            "zScore": [5.0, 3.0],
            "pValueMantissa": [5.7330, 2.6998],
            "pValueExponent": [-7, -3],
        }
    )
    prepared = PreparedSushie(
        snps=pd.DataFrame({"snp": variants["variantId"]}),
        data=io.ssData(
            zs=[jnp.array([2.0, 1.0]), jnp.array([3.0, 2.0])],
            lds=[jnp.eye(2), jnp.eye(2)],
            ns=jnp.array([[1000], [2000]]),
            pi=None,
        ),
        variants=variants,
        study_ids=["STUDY_A", "STUDY_B"],
        ancestries=["eur", "afr"],
        sample_sizes=[1000, 2000],
        outcome=PreparationOutcome(None, None, "SUCCESS", None, 4, 2, 2, 2),
    )
    fit = SimpleNamespace(
        cs=pd.DataFrame(
            {
                "CSIndex": [1, 1],
                "SNPIndex": [0, 1],
                "alpha": [0.8, 0.2],
                "c_alpha": [0.8, 1.0],
            }
        ),
        alphas=pd.DataFrame({"purity_l1": [0.8, 0.8]}),
        posteriors=SimpleNamespace(
            alpha=jnp.array([[0.8, 0.2]]),
            log_bf=jnp.array([[2.0, 1.0]]),
            post_mean=jnp.array([[[0.08, 0.04], [0.02, 0.01]]]),
            post_mean_sq=jnp.ones((1, 2, 2, 2)),
            weighted_sum_covar=jnp.ones((1, 2, 2)),
            kl=jnp.array([0.2]),
        ),
        priors=SimpleNamespace(
            pi=jnp.array([0.5, 0.5]),
            resid_var=jnp.array([[1.0], [0.9]]),
            effect_covar=jnp.array([[0.1, 0.02], [0.02, 0.2]]),
        ),
        pip_all=jnp.array([0.8, 0.2]),
        pip_cs=jnp.array([0.8, 0.2]),
        elbo=jnp.array([-jnp.inf, 1.0, 1.00001]),
        elbo_increase=True,
        l_order=jnp.array([0]),
    )
    monkeypatch.setattr("sushie.pipeline_cli.prepare_inputs", lambda _inputs: prepared)
    monkeypatch.setattr("sushie.pipeline_cli.infer_sushie_ss", lambda **_kwargs: fit)

    result = CliRunner().invoke(
        app,
        [
            "--fine-mapping-locus-set",
            str(locus),
            "--multi-ancestry-pairwise-ld",
            str(ld),
            "--study-metadata",
            str(metadata),
            "--run-id",
            "RUN_A",
            "--fine-mapping-locus-set-id",
            "LOCUS_A",
            "--study-locus-output",
            str(study_locus),
            "--extended-results-output",
            str(extended),
            "--stats-output",
            str(stats),
            "--min-snps",
            "2",
            "--L",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(stats.read_text())["status"] == "SUCCESS"
    assert pd.read_parquet(study_locus).loc[0, "finemappingMethod"] == "SuShiE"
    assert extended.is_file()
