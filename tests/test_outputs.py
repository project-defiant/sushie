from types import SimpleNamespace

import jax.numpy as jnp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sushie.outputs import study_locus_from_result, write_study_locus


def test_study_locus_output_uses_component_alpha_and_stable_lead_id(tmp_path) -> None:
    result = SimpleNamespace(
        cs=pd.DataFrame(
            {
                "CSIndex": [1, 1],
                "SNPIndex": [0, 1],
                "alpha": [0.8, 0.2],
                "c_alpha": [0.8, 1.0],
                "pip_all": [0.8, 0.2],
                "pip_cs": [0.8, 0.2],
            }
        ),
        posteriors=SimpleNamespace(log_bf=jnp.array([[2.0, 1.0]])),
        priors=SimpleNamespace(pi=jnp.array([0.5, 0.5])),
        alphas=pd.DataFrame({"purity_l1": [0.8, 0.8]}),
    )
    variants = pd.DataFrame(
        {
            "variantId": ["1_100_A_G", "1_110_C_T"],
            "chromosome": ["1", "1"],
            "position": [100, 110],
            "beta": [0.2, 0.1],
            "zScore": [4.0, 2.0],
            "pValueMantissa": [6.3342, 4.5500],
            "pValueExponent": [-5, -2],
            "standardError": [0.05, 0.05],
        }
    )

    output = study_locus_from_result(
        result,
        variants,
        run_id="RUN_A",
        fine_mapping_locus_set_id="LOCUS_A",
        study_id="STUDY_A",
        chromosome="1",
        locus_start=90,
        locus_end=120,
    )

    assert list(output.columns) == [
        "studyLocusId",
        "studyType",
        "variantId",
        "chromosome",
        "position",
        "region",
        "studyId",
        "beta",
        "zScore",
        "pValueMantissa",
        "pValueExponent",
        "effectAlleleFrequencyFromSource",
        "standardError",
        "subStudyDescription",
        "qualityControls",
        "finemappingMethod",
        "credibleSetIndex",
        "credibleSetlog10BF",
        "purityMeanR2",
        "purityMinR2",
        "locusStart",
        "locusEnd",
        "sampleSize",
        "ldSet",
        "locus",
        "confidence",
        "isTransQtl",
    ]
    assert len(output) == 1
    assert output.loc[0, "variantId"] == "1_100_A_G"
    assert output.loc[0, "locus"][0]["posteriorProbability"] == 0.8
    assert output.loc[0, "locus"][0]["is95CredibleSet"] is True
    assert output.loc[0, "locus"][1]["is95CredibleSet"] is True
    assert output.loc[0, "locus"][0]["logBF"] == 2.0
    assert output.loc[0, "finemappingMethod"] == "SuShiE"
    assert output.loc[0, "credibleSetIndex"] == 0
    assert output.loc[0, "purityMinR2"] == pytest.approx(0.64)
    assert output.loc[0, "studyLocusId"] == "a2a17b81b0f2ebc561800cd67ce5c8c9"

    path = tmp_path / "study_locus.parquet"
    write_study_locus(output, path)
    schema = pq.read_schema(path)
    assert schema.field("sampleSize").type == pa.int32()
    assert schema.field("effectAlleleFrequencyFromSource").type == pa.float32()
    assert pa.types.is_list(schema.field("ldSet").type)
    assert (
        schema.field("locus").type.value_type.field("pValueMantissa").type
        == pa.float32()
    )
