"""Main entrypoint for the package."""

import logging
from argparse import Namespace
from enum import Enum
from logging import LogRecord
from pathlib import Path
from typing import Annotated, Literal

import typer
from typer import Context

from sushie.helpers import run_finemap
from sushie.option_store import OptionStore


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
    ctx: Context = None,  # type: ignore
) -> None:
    """Run the Typer finemap command."""
    logging.info("Running finemap")
    # Get context from function parameter injection
    # Context is automatically injected by Typer when used as function parameter
    if ctx is not None:
        args = Namespace(**ctx.params)
    else:
        # Fallback: construct args from arguments directly
        args = Namespace(
            summary=summary,
            pheno=pheno,
            gwas=gwas,
            parquet=parquet,
            plink=plink,
            vcf=vcf,
            bgen=bgen,
            ancestry_index=ancestry_index,
            keep=keep,
            covar=covar,
            ld=ld,
            chrom=chrom,
            start=start,
            end=end,
            sample_size=sample_size,
            gwas_header=gwas_header,
            gwas_sig=gwas_sig,
            gwas_sig_type=gwas_sig_type,
            L=L,
            pi=pi,
            resid_var=resid_var,
            effect_var=effect_var,
            rho=rho,
            no_scale=no_scale,
            no_regress=no_regress,
            no_update=no_update,
            max_iter=max_iter,
            min_tol=min_tol,
            threshold=threshold,
            purity=purity,
            purity_method=purity_method,
            ld_adjust=ld_adjust,
            max_select=max_select,
            min_snps=min_snps,
            maf=maf,
            rint=rint,
            no_reorder=no_reorder,
            keep_ambiguous=keep_ambiguous,
            meta=meta,
            mega=mega,
            her=her,
            cv=cv,
            cv_num=cv_num,
            seed=seed,
            alphas=alphas,
            numpy=numpy,
            trait=trait,
            quiet=quiet,
            verbose=verbose,
            compress=compress,
            platform=platform,
            jax_precision=jax_precision,
            output=output,
        )
    # Convert empty list/tuple to None for optional list parameters
    if args.covar is not None and len(args.covar) == 0:
        args.covar = None
    if args.ld is not None and len(args.ld) == 0:
        args.ld = None
    if args.plink is not None and len(args.plink) == 0:
        args.plink = None
    if args.vcf is not None and len(args.vcf) == 0:
        args.vcf = None
    if args.bgen is not None and len(args.bgen) == 0:
        args.bgen = None
    if args.gwas is not None and len(args.gwas) == 0:
        args.gwas = None
    if args.pheno is not None and len(args.pheno) == 0:
        args.pheno = None
    if args.sample_size is not None and len(args.sample_size) == 0:
        args.sample_size = None
    if args.resid_var is not None and len(args.resid_var) == 0:
        args.resid_var = None
    if args.effect_var is not None and len(args.effect_var) == 0:
        args.effect_var = None
    if args.rho is not None and len(args.rho) == 0:
        args.rho = None
    # Convert Path objects to strings for compatibility with underlying libraries
    if args.plink is not None:
        args.plink = [str(p) for p in args.plink]
    if args.vcf is not None:
        args.vcf = [str(v) for v in args.vcf]
    if args.bgen is not None:
        args.bgen = [str(b) for b in args.bgen]
    if args.gwas is not None:
        args.gwas = [str(g) for g in args.gwas]
    if args.ld is not None:
        args.ld = [str(l) for l in args.ld]
    if args.pheno is not None:
        args.pheno = [str(p) for p in args.pheno]
    if args.covar is not None:
        args.covar = [str(c) for c in args.covar]
    if args.keep is not None:
        args.keep = str(args.keep)
    if args.ancestry_index is not None:
        args.ancestry_index = str(args.ancestry_index)
    if args.pi != "uniform":
        args.pi = str(args.pi)

    run_finemap(args)
