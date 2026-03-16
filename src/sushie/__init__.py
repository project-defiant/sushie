"""Main entrypoint for the package."""

import logging
from logging import LogRecord
from pathlib import Path
from typing import Annotated, Literal
from argparse import Namespace
import typer
from sushie.cli import parameter_check, parameter_check_ss
from sushie.option_store import OptionStore
from enum import Enum
from typer import Context


class RunMode(Enum):
    FROM_SUMMARY_STATISTICS = 0
    FROM_GENOTYPES = 1


def _filter_absl_cpu_warning(record: LogRecord) -> bool:
    """Filter absl cpu warning.

    Args:
        record (LogRecord): logging record

    Returns:
        bool: if we should filter the record out.

    https://github.com/pyro-ppl/numpyro/blob/master/numpyro/__init__.py
    Copyright Pyro devs
    """
    return not record.getMessage().startswith("No GPU/TPU found, falling back to CPU.")


logging.getLogger("absl").addFilter(_filter_absl_cpu_warning)

main = typer.Typer()
option_store = OptionStore()


@main.command()
def version() -> None:
    """Get the version and exit."""
    import sys
    from importlib.metadata import version

    print(version("sushie"))

    sys.exit(0)


@main.command(
    help="Perform SNP fine-mapping of gene expression on individual genotype and pheotype using SuShiE."
)
def finemap(
    summary: Annotated[bool, option_store.get("summary")] = False,
    pheno: Annotated[list[Path] | None, option_store.get("pheno")] = None,
    gwas: Annotated[list[Path] | None, option_store.get("gwas")] = None,
    parquet: Annotated[bool | None, option_store.get("parquet")] = None,
    plink: Annotated[list[Path] | None, option_store.get("plink")] = None,
    vcf: Annotated[list[Path] | None, option_store.get("vcf")] = None,
    bgen: Annotated[list[Path] | None, option_store.get("bgen")] = None,
    ancestry_index: Annotated[Path | None, option_store.get("ancestry_index")] = None,
    keep: Annotated[Path | None, option_store.get("keep")] = None,
    covar: Annotated[list[Path] | None, option_store.get("covar")] = None,
    ld: Annotated[list[Path] | None, option_store.get("ld")] = None,
    chrom: Annotated[int | None, option_store.get("chrom")] = None,
    start: Annotated[int | None, option_store.get("start")] = None,
    end: Annotated[int | None, option_store.get("end")] = None,
    sample_size: Annotated[list[int] | None, option_store.get("sample_size")] = None,
    gwas_header: Annotated[list[str], option_store.get("gwas_header")] = [
        "chrom",
        "snp",
        "pos",
        "a1",
        "a0",
        "z",
    ],
    gwas_sig: Annotated[float, option_store.get("gwas_sig")] = 1.0,
    gwas_sig_type: Annotated[
        Literal["at-least", "all"], option_store.get("gwas_sig_type")
    ] = "at-least",
    L: Annotated[int, option_store.get("L")] = 10,
    pi: Annotated[str, option_store.get("pi")] = "uniform",
    resid_var: Annotated[list[float] | None, option_store.get("resid_var")] = None,
    effect_var: Annotated[list[float] | None, option_store.get("effect_var")] = None,
    rho: Annotated[list[float] | None, option_store.get("rho")] = None,
    no_scale: Annotated[bool, option_store.get("no_scale")] = False,
    no_regress: Annotated[bool, option_store.get("no_regress")] = False,
    no_update: Annotated[bool, option_store.get("no_update")] = False,
    max_iter: Annotated[int, option_store.get("max_iter")] = 500,
    min_tol: Annotated[float, option_store.get("min_tol")] = 1e-3,
    threshold: Annotated[float, option_store.get("threshold")] = 0.95,
    purity: Annotated[float, option_store.get("purity")] = 0.5,
    purity_method: Annotated[
        Literal["weighted", "max", "min"], option_store.get("purity_method")
    ] = "weighted",
    ld_adjust: Annotated[float, option_store.get("ld_adjust")] = 0.0,
    max_select: Annotated[int, option_store.get("max_select")] = 250,
    min_snps: Annotated[int, option_store.get("min_snps")] = 100,
    maf: Annotated[float, option_store.get("maf")] = 0.01,
    rint: Annotated[bool, option_store.get("rint")] = False,
    no_reorder: Annotated[bool, option_store.get("no_reorder")] = False,
    keep_ambiguous: Annotated[bool, option_store.get("keep_ambiguous")] = False,
    meta: Annotated[bool, option_store.get("meta")] = False,
    mega: Annotated[bool, option_store.get("mega")] = False,
    her: Annotated[bool, option_store.get("her")] = False,
    cv: Annotated[bool, option_store.get("cv")] = False,
    cv_num: Annotated[int, option_store.get("cv_num")] = 5,
    seed: Annotated[int, option_store.get("seed")] = 12345,
    alphas: Annotated[bool, option_store.get("alphas")] = False,
    numpy: Annotated[bool, option_store.get("numpy")] = False,
    trait: Annotated[str, option_store.get("trait")] = "Trait",
    quiet: Annotated[bool, option_store.get("quiet")] = False,
    verbose: Annotated[bool, option_store.get("verbose")] = False,
    compress: Annotated[bool, option_store.get("compress")] = False,
    platform: Annotated[
        Literal["cpu", "gpu", "tpu"], option_store.get("platform")
    ] = "cpu",
    jax_precision: Annotated[Literal[32, 64], option_store.get("jax_precision")] = 64,
    output: Annotated[str, option_store.get("output")] = "sushie_finemap",
) -> None:
    """Run the Typer finemap command."""
    logging.info("Running finemap")
    ctx = typer.get_current_context()

    # ctx.params contains all CLI parameters
    args = Namespace(**ctx.params)
    arg_proc = ArgumentPreprocessor(summary)


class ArgumentPreprocessor:
    """Argument preprocessor."""

    def __init__(self, summary) -> RunMode:
        """Initialize argument preprocessor based on summary."""
        if summary:
            self.run_mode = RunMode.FROM_SUMMARY_STATISTICS
        else:
            self.run_mode = RunMode.FROM_GENOTYPES
