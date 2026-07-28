from pathlib import Path

import pandas as pd
import pytest

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnla
import jax.random as rdm
import jax.scipy.linalg as jsla

import sushie

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("N,K", [(50, 2), (100, 1)])
def test_infer_sushie_simple(N: int, K: int, seed: int = 0):
    key = rdm.PRNGKey(seed)

    P = 4
    L = 2

    key, g_key, b_key, s_key, y_key = rdm.split(key, 5)

    h2g = 0.1

    g_covar_block = jnp.array([[1.0, 0.8], [0.8, 1.0]])
    g_covar = jsla.block_diag(g_covar_block, g_covar_block)
    L_g = jnla.cholesky(g_covar)

    X = rdm.normal(g_key, shape=(K, N, P))
    X = jnp.einsum("knp,pj->knj", X, L_g)

    snps = jnp.array([0, 2])
    beta = jnp.ones((L, K))

    G = jnp.einsum("knl,lk->kn", X[:, :, snps], beta)

    s2gs = jnp.std(G, axis=-1)
    s2es = ((1 / h2g) - 1) * s2gs
    y = G + rdm.normal(y_key, shape=(K, N)) * jnp.sqrt(s2es[:, jnp.newaxis])

    Xs = []
    ys = []
    for k in range(K):
        Xs.append(X[k, :, :])
        ys.append(y[k, :])

    # this really is just sanity check that it doesn't crash...
    res = sushie.infer.infer_sushie(Xs, ys, L=L, min_snps=L)

    assert res is not None


@pytest.mark.parametrize("N,P,K,L", [(50, 100, 2, 2), (100, 50, 3, 2)])
def test_infer_sushie(N: int, P: int, K: int, L: int, seed: int = 0):
    key = rdm.PRNGKey(seed)

    key, g_key, b_key, s_key, y_key = rdm.split(key, 5)

    h2g = 0.1
    rho = 0.8 * h2g
    covar = (
        jnp.diag(h2g * jnp.ones(K))
        + rho * jnp.ones((K, K))
        - jnp.diag(rho * jnp.ones(K))
    )

    X = rdm.normal(g_key, shape=(K, N, P))
    snps = rdm.choice(s_key, P, shape=(L,), replace=False)
    beta = rdm.multivariate_normal(b_key, mean=jnp.zeros(K), cov=covar, shape=(L,))

    G = jnp.einsum("knl,lk->kn", X[:, :, snps], beta)

    s2gs = jnp.std(G, axis=-1)
    s2es = ((1 / h2g) - 1) * s2gs
    y = G + rdm.normal(y_key, shape=(K, N)) * jnp.sqrt(s2es[:, jnp.newaxis])

    Xs = []
    ys = []
    for k in range(K):
        Xs.append(X[k, :, :])
        ys.append(y[k, :])

    # this really is just sanity check that it doesn't crash...
    res = sushie.infer.infer_sushie(Xs, ys, L=L, min_snps=L)
    assert res is not None


def _write_gwas(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _write_ld(path: Path, columns: list[str], values: list[list[float]]) -> None:
    pd.DataFrame(values, columns=columns).to_csv(path, sep="\t", index=False)


def test_process_raw_ss_uses_only_variants_shared_by_all_studies_and_ld_panels(
    tmp_path: Path,
) -> None:
    gwas_1 = tmp_path / "study_1.tsv"
    gwas_2 = tmp_path / "study_2.tsv"
    ld_1 = tmp_path / "ld_1.tsv"
    ld_2 = tmp_path / "ld_2.tsv"

    _write_gwas(
        gwas_1,
        [
            {"chrom": 1, "snp": "rs1", "pos": 101, "a1": "A", "a0": "G", "z": 1.0},
            {"chrom": 1, "snp": "rs2", "pos": 102, "a1": "C", "a0": "T", "z": 2.0},
            {"chrom": 1, "snp": "rs3", "pos": 103, "a1": "G", "a0": "A", "z": 3.0},
        ],
    )
    _write_gwas(
        gwas_2,
        [
            {"chrom": 1, "snp": "rs2", "pos": 102, "a1": "C", "a0": "T", "z": 4.0},
            {"chrom": 1, "snp": "rs3", "pos": 103, "a1": "G", "a0": "A", "z": 5.0},
            {"chrom": 1, "snp": "rs4", "pos": 104, "a1": "T", "a0": "C", "z": 6.0},
        ],
    )
    _write_ld(
        ld_1,
        ["rs1", "rs2", "rs3"],
        [
            [1.0, 0.1, 0.0],
            [0.1, 1.0, 0.2],
            [0.0, 0.2, 1.0],
        ],
    )
    _write_ld(
        ld_2,
        ["rs2", "rs3", "rs4"],
        [
            [1.0, 0.3, 0.0],
            [0.3, 1.0, 0.4],
            [0.0, 0.4, 1.0],
        ],
    )

    snps, data = sushie.infer_ss.prepare_sushie_ss_data(
        gwas=[
            sushie.io.read_gwas(
                str(gwas_1), ["chrom", "snp", "pos", "a1", "a0", "z"], None, 0, 10_000
            ),
            sushie.io.read_gwas(
                str(gwas_2), ["chrom", "snp", "pos", "a1", "a0", "z"], None, 0, 10_000
            ),
        ],
        lds=[sushie.io.read_ld(str(ld_1)), sushie.io.read_ld(str(ld_2))],
        sample_size=[100, 120],
        pi=pd.DataFrame(),
    )

    assert snps["snp"].tolist() == ["rs2", "rs3"]
    assert snps["pi"].tolist() == pytest.approx([0.5, 0.5])
    assert [z.tolist() for z in data.zs] == [[2.0, 3.0], [4.0, 5.0]]
    assert [ld.tolist() for ld in data.lds] == [
        [[1.0, 0.2], [0.2, 1.0]],
        [[1.0, 0.3], [0.3, 1.0]],
    ]


def test_prepare_sushie_ss_data_reports_empty_intersection_without_raising(
    tmp_path: Path,
) -> None:
    gwas = pd.DataFrame(
        [{"chrom": 1, "snp": "rs1", "pos": 101, "a1": "A", "a0": "G", "z": 2.0}]
    )
    ld = pd.DataFrame([[1.0]], columns=["rs1"])

    outcome = sushie.infer_ss.prepare_sushie_ss_data_with_stats(
        gwas=[gwas, gwas.assign(snp="rs2")],
        lds=[ld, ld.rename(columns={"rs1": "rs2"})],
        sample_size=[100, 120],
    )

    assert outcome.status == "EMPTY_INTERSECTION"
    assert outcome.reason
    assert outcome.final_variants == 0
    assert outcome.snps is None
    assert outcome.data is None
