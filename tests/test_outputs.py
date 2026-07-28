from types import SimpleNamespace

import jax.numpy as jnp
import pandas as pd

from sushie.outputs import study_locus_from_result


def test_study_locus_output_uses_component_alpha_and_stable_lead_id() -> None:
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
    )
    variants = pd.DataFrame(
        {
            "variantId": ["1_100_A_G", "1_110_C_T"],
            "chromosome": ["1", "1"],
            "position": [100, 110],
            "beta": [0.2, 0.1],
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
    ]
    assert len(output) == 1
    assert output.loc[0, "variantId"] == "1_100_A_G"
    assert output.loc[0, "locus"][0]["posteriorProbability"] == 0.8
    assert output.loc[0, "locus"][0]["is95CredibleSet"] is False
    assert output.loc[0, "locus"][1]["is95CredibleSet"] is True
    assert output.loc[0, "locus"][0]["logBF"] == 2.0
    assert output.loc[0, "studyLocusId"] == "d583d81ee1af2211371ede4c6528fc90"
