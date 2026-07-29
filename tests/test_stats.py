from types import SimpleNamespace

import jax.numpy as jnp

from sushie.stats import stats_from_result, write_stats


def test_stats_report_non_converged_result(tmp_path) -> None:
    result = SimpleNamespace(
        posteriors=SimpleNamespace(log_bf=jnp.zeros((2, 5))),
        elbo_increase=False,
    )

    stats = stats_from_result(
        result, run_id="RUN_A", fine_mapping_locus_set_id="LOCUS_A"
    )
    output = tmp_path / "stats.json"
    write_stats(output, stats)

    assert stats.status == "NON_CONVERGED"
    assert stats.finalVariants == 5
    assert stats.nComponents == 2
    assert '"reason": "SuShiE fit did not converge"' in output.read_text()
