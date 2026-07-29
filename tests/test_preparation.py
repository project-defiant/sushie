import json

import numpy as np
import pandas as pd

from sushie.models import PipelineInputs
from sushie.preparation import prepare_inputs


def test_prepare_inputs_builds_strict_shared_summary_statistics(tmp_path) -> None:
    locus = tmp_path / "locus.parquet"
    pd.DataFrame(
        {
            "fineMappingLocusSetId": ["LOCUS_A", "LOCUS_A"],
            "studyId": ["STUDY_A", "STUDY_B"],
            "locus": [
                [
                    {
                        "variantId": "1_100_A_G",
                        "beta": 0.2,
                        "standardError": 0.1,
                    },
                    {
                        "variantId": "1_110_C_T",
                        "beta": 0.1,
                        "standardError": 0.1,
                    },
                ],
                [
                    {
                        "variantId": "1_100_A_G",
                        "beta": 0.3,
                        "standardError": 0.1,
                    },
                    {
                        "variantId": "1_110_C_T",
                        "beta": 0.2,
                        "standardError": 0.1,
                    },
                ],
            ],
        }
    ).to_parquet(locus)
    ld = tmp_path / "ld.parquet"
    pd.DataFrame(
        {
            "ancestry": ["eur", "eur", "afr", "afr"],
            "variantIdI": ["1_100_A_G", "1_100_A_G", "1_100_A_G", "1_100_A_G"],
            "variantIdJ": ["1_100_A_G", "1_110_C_T", "1_100_A_G", "1_110_C_T"],
            "r": [1.0, 0.2, 1.0, 0.3],
        }
    ).to_parquet(ld)
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"studyId": "STUDY_A", "ancestry": "eur", "sampleSize": 1000},
                {"studyId": "STUDY_B", "ancestry": "afr", "sampleSize": 2000},
            ]
        )
        + "\n"
    )

    prepared = prepare_inputs(
        PipelineInputs(
            run_id="RUN_A",
            fine_mapping_locus_set_id="LOCUS_A",
            fine_mapping_locus_set=locus,
            multi_ancestry_pairwise_ld=ld,
            study_metadata=metadata,
            study_locus_output=tmp_path / "study.parquet",
            extended_results_output=tmp_path / "fit.h5ad",
            stats_output=tmp_path / "stats.json",
        )
    )

    assert prepared.outcome.status == "SUCCESS"
    assert prepared.snps["snp"].tolist() == ["1_100_A_G", "1_110_C_T"]
    np.testing.assert_allclose(
        [z.tolist() for z in prepared.data.zs], [[2.0, 1.0], [3.0, 2.0]]
    )
    np.testing.assert_allclose(prepared.data.lds[1], [[1.0, 0.3], [0.3, 1.0]])
    np.testing.assert_allclose(prepared.variants["beta"], [0.25, 0.15])
    np.testing.assert_allclose(prepared.variants["standardError"], [2**-0.5 * 0.1] * 2)


def test_prepare_inputs_removes_variants_missing_ld_in_any_ancestry(tmp_path) -> None:
    locus = tmp_path / "locus.parquet"
    pd.DataFrame(
        {
            "fineMappingLocusSetId": ["LOCUS_A", "LOCUS_A"],
            "studyId": ["STUDY_A", "STUDY_B"],
            "locus": [
                [
                    {
                        "variantId": variant_id,
                        "beta": beta,
                        "standardError": 0.1,
                    }
                    for variant_id, beta in [
                        ("1_100_A_G", 0.2),
                        ("1_110_C_T", 0.1),
                    ]
                ],
                [
                    {
                        "variantId": variant_id,
                        "beta": beta,
                        "standardError": 0.1,
                    }
                    for variant_id, beta in [
                        ("1_100_A_G", 0.3),
                        ("1_110_C_T", 0.2),
                    ]
                ],
            ],
        }
    ).to_parquet(locus)
    ld = tmp_path / "ld.parquet"
    pd.DataFrame(
        {
            "ancestry": ["eur", "eur", "afr"],
            "variantIdI": ["1_100_A_G", "1_100_A_G", "1_100_A_G"],
            "variantIdJ": ["1_100_A_G", "1_110_C_T", "1_100_A_G"],
            "r": [1.0, 0.2, 1.0],
        }
    ).to_parquet(ld)
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"studyId": "STUDY_A", "ancestry": "eur", "sampleSize": 1000},
                {"studyId": "STUDY_B", "ancestry": "afr", "sampleSize": 2000},
            ]
        )
        + "\n"
    )

    prepared = prepare_inputs(
        PipelineInputs(
            run_id="RUN_A",
            fine_mapping_locus_set_id="LOCUS_A",
            fine_mapping_locus_set=locus,
            multi_ancestry_pairwise_ld=ld,
            study_metadata=metadata,
            study_locus_output=tmp_path / "study.parquet",
            extended_results_output=tmp_path / "fit.h5ad",
            stats_output=tmp_path / "stats.json",
        )
    )

    assert prepared.outcome.status == "SUCCESS"
    assert prepared.outcome.shared_variants == 2
    assert prepared.outcome.ld_valid_variants == 1
    assert prepared.outcome.final_variants == 1
