import argparse
import logging
import os
import sys
import warnings
from collections.abc import Callable
from importlib import metadata
from typing import List, Tuple

import pandas as pd

from . import io, log

# Filter ABSL and JAX warnings that clutter output (must be before JAX import)
warnings.filterwarnings("ignore", module="absl")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
import jax.numpy as jnp  # noqa: E402


def parameter_check(
    args: argparse.Namespace,
) -> tuple[int, pd.DataFrame, list[str], pd.DataFrame, list[str], Callable]:
    """The function to process raw phenotype, genotype, covariates data across ancestries fine-mapping.

    Args:
        args: The command line parameter input.

    Returns:
        :py:obj:`Tuple[int, pd.DataFrame, List[str], Callable]`:
            A tuple of
                #. an integer to indicate how many ancestries,
                #. a DataFrame that contains ancestry index (can be none),
                #. a list that contains subject ID that fine-mapping performs on.
                #. a DataFrame that contains prior probability for each SNP to be causal.
                #. a list of genotype data paths (:py:obj:`List[str]`),
                #. genotype read-in function (:py:obj:`Callable`).
    """
    if args.pheno is None:
        raise ValueError(
            "No phenotype file specified. Specify --summary if summary-level fine-mapping is wanted."
        )

    if args.ancestry_index is not None:
        log.logger.debug("Read in ancestry index file.")
        ancestry_index = pd.read_csv(args.ancestry_index[0], header=None, sep="\t")
        old_pt = ancestry_index.shape[0]
        ancestry_index = ancestry_index.drop_duplicates()

        if old_pt != ancestry_index.shape[0]:
            log.logger.debug(
                f"Index file has {old_pt - ancestry_index.shape[0]} duplicated subjects."
            )

        if ancestry_index[0].duplicated().sum() != 0:
            raise ValueError(
                "The ancestry index file contains subjects with multiple ancestry index. Check the source."
            )

        n_pop = len(ancestry_index[1].unique())
        index_check = jnp.all(
            jnp.array(ancestry_index[1].unique()).sort() == (jnp.arange(n_pop) + 1)
        )

        if not index_check:
            raise ValueError(
                "The ancestry index doesn't start from 1 continuously to the total number of ancestry."
                + f" Check {args.ancestry_index}."
            )

        if len(args.pheno) > 1:
            raise ValueError(
                "Multiple phenotype files are detected. Expectation is one when --ancestry-index is specified."
            )

        log.logger.debug(
            "Detect ancestry index file, so it expects to have single phenotype, genotype, and covariates files"
        )

    else:
        ancestry_index = pd.DataFrame()
        n_pop = len(args.pheno)

    name_ancestry = "ancestry" if n_pop == 1 else "ancestries"

    log.logger.info(f"Detect phenotypes for {args.trait} for {n_pop} {name_ancestry}.")

    n_geno = (
        int(args.plink is not None)
        + int(args.vcf is not None)
        + int(args.bgen is not None)
    )

    if n_geno > 1:
        log.logger.info(
            f"Detect {n_geno} genotypes, will only use one type of genotypes in the order of 'plink, vcf, and bgen'"
        )

    # decide genotype data
    if args.plink is not None:
        if args.ancestry_index is not None:
            if len(args.plink) > 1:
                raise ValueError(
                    "Multiple plink files are detected. Expectation is one when --ancestry-index is specified."
                )
        elif len(args.plink) != n_pop:
            raise ValueError(
                "The numbers of ancestries in plink geno and pheno data does not match. Check the source."
            )

        log.logger.info(
            f"Detect genotype data in plink format for {n_pop} {name_ancestry}."
        )
        geno_path = args.plink
        geno_func = io.read_triplet
    elif args.vcf is not None:
        if args.ancestry_index is not None:
            if len(args.vcf) > 1:
                raise ValueError(
                    "Multiple vcf files are detected. Expectation is one when --ancestry-index is specified."
                )
        elif len(args.vcf) != n_pop:
            raise ValueError(
                "The numbers of ancestries in vcf geno and pheno data does not match. Check the source."
            )
        log.logger.info(
            f"Detect genotype data in vcf format for {n_pop} {name_ancestry}."
        )
        geno_path = args.vcf
        geno_func = io.read_vcf
    elif args.bgen is not None:
        if args.ancestry_index is not None:
            if len(args.bgen) > 1:
                raise ValueError(
                    "Multiple bgen files are detected. Expectation is one when --ancestry-index is specified."
                )
        elif len(args.bgen) != n_pop:
            raise ValueError(
                "The numbers of ancestries in bgen geno and pheno data does not match. Check the source."
            )

        log.logger.info(
            f"Detect genotype data in bgen format for {n_pop} {name_ancestry}."
        )
        geno_path = args.bgen
        geno_func = io.read_bgen
    else:
        raise ValueError(
            "No genotype data specified in either plink, vcf, or bgen format. Check the source."
        )

    if args.covar is not None:
        if args.ancestry_index is not None:
            if len(args.covar) > 1:
                raise ValueError(
                    "Multiple covariates files are detected. Expectation is one when --ancestry-index is specified."
                )
        else:
            print(f"DEBUG: covar={args.covar}, type={type(args.covar)}")
            if args.covar is not None and len(args.covar) != n_pop:
                raise ValueError(
                    "The number of covariates data does not match geno data."
                )
        log.logger.info("Detect covariates data.")
    else:
        log.logger.info("No covariates detected for this analysis.")

    keep_subject = []
    if args.keep is not None:
        log.logger.info(
            "Detect keep subject file. The inference only performs on the subjects listed in the file."
        )
        df_keep = pd.read_csv(args.keep[0], header=None, sep="\t")[[0]]
        if df_keep.shape[0] == 0:
            raise ValueError(
                "No subjects are listed in the keep subject file. Check the source."
            )
        old_pt = df_keep.shape[0]
        df_keep = df_keep.drop_duplicates()

        if old_pt != df_keep.shape[0]:
            log.logger.debug(
                f"The keep subject file has {old_pt - df_keep.shape[0]} duplicated subjects."
            )
        keep_subject = df_keep[0].values.tolist()

    if args.pi != "uniform":
        log.logger.info(
            "Detect file that contains prior weights for each SNP to be causal."
        )
        pi = pd.read_csv(args.pi, header=None, sep="\t")
        if pi.shape[0] == 0:
            raise ValueError(
                "No prior weights are listed in the prior file. Check the source."
            )

        if pi.shape[1] < 2:
            raise ValueError(
                "The prior file has less than 2 columns. It has to be at least two columns."
                + " The first column is the SNP ID, and the second column the prior probability."
            )

        if pi.shape[1] > 2:
            log.logger.debug(
                "The prior file has more than 2 columns. Will only use the first two columns."
            )

        pi = pi.iloc[:, 0:2]
        pi.columns = ["snp", "pi"]
    else:
        pi = pd.DataFrame()

    if args.seed <= 0:
        raise ValueError(
            "The seed specified for randomization must be greater than 0. Choose a positive integer using --seed."
        )

    if args.cv:
        if args.cv_num <= 1:
            raise ValueError(
                "The number of folds in cross validation must be greater than 1."
                + " Update with --cv-num.",
            )

    if args.maf <= 0 or args.maf > 0.5:
        raise ValueError(
            "The minor allele frequency (MAF) has to be between 0 (exclusive) and 0.5 (inclusive)."
            + " Choose a valid frequency using --maf."
        )

    if (args.meta or args.mega) and n_pop == 1:
        log.logger.debug(
            "The number of ancestry is 1, but --meta or --mega is specified. Will skip meta or mega SuSiE."
        )

    if args.chrom is None and args.start is None and args.end is None:
        log.logger.debug(
            "No region is specified. Will use all SNPs available in the data."
        )

    elif args.chrom is not None and args.start is not None and args.end is not None:
        if args.start <= 0:
            raise ValueError(
                "The start position for the region must be greater than 0. Update with --start."
            )

        if args.end <= 0:
            raise ValueError(
                "The end position for the region must be greater than 0. Update with --end."
            )

        if args.end <= args.start:
            raise ValueError(
                "The end position for the region must be greater than --start. Update with"
                + " --start or --end."
            )

        log.logger.info(
            f"Detect region (chrom{args.chrom}:{args.start}:{args.end}) to be fine-mapped."
            + " Will only use SNPs within the region."
        )
    else:
        raise ValueError(
            "The region is not specified correctly. Please provide --chrom, --start, and --end together,"
            + " or omit all of them."
        )

    log.logger.debug("Finish parameter check for individual-level fine-mapping.")

    return n_pop, ancestry_index, keep_subject, pi, geno_path, geno_func


def parameter_check_ss(
    args: argparse.Namespace,
) -> Tuple[int, pd.DataFrame, List[str], Callable, bool]:
    """The function to process raw phenotype, genotype, covariates data across ancestries
        for summary-level fine-mapping.

    Args:
        args: The command line parameter input.

    Returns:
        :py:obj:`Tuple[int, pd.DataFrame, List[str], Callable]`:
            A tuple of
                #. an integer to indicate how many ancestries,
                #. a DataFrame that contains prior probability for each SNP to be causal.
                #. a list of genotype data paths (:py:obj:`List[str]`),
                #. genotype read-in function (:py:obj:`Callable`).
                #. a boolean to indicate whether the genotype data is in LD format.

    """
    if args.gwas is None:
        raise ValueError("No GWAS summary statistics file specified. Check the source.")
    else:
        n_pop = len(args.gwas)

    name_ancestry = "ancestry" if n_pop == 1 else "ancestries"

    log.logger.info(f"Detect GWAS files for {args.trait} for {n_pop} {name_ancestry}.")

    n_geno = (
        int(args.plink is not None)
        + int(args.vcf is not None)
        + int(args.bgen is not None)
        + int(args.ld is not None)
    )

    if n_geno > 1:
        log.logger.info(
            f"Detect {n_geno} genotype or LD files,"
            + " will only use one type of file in the order of 'plink, vcf, bgen, and ld'",
        )

    # decide genotype data
    ld_file = False
    if args.plink is not None:
        if len(args.plink) != n_pop:
            raise ValueError(
                "The numbers of ancestries in plink geno and GWAS data does not match. Check the source."
            )

        log.logger.info(
            f"Detect genotype data in plink format for {n_pop} {name_ancestry}."
        )
        geno_path = args.plink
        geno_func = io.read_triplet
    elif args.vcf is not None:
        if len(args.vcf) != n_pop:
            raise ValueError(
                "The numbers of ancestries in vcf geno and GWAS data does not match. Check the source."
            )
        log.logger.info(
            f"Detect genotype data in vcf format for {n_pop} {name_ancestry}."
        )
        geno_path = args.vcf
        geno_func = io.read_vcf
    elif args.bgen is not None:
        if len(args.bgen) != n_pop:
            raise ValueError(
                "The numbers of ancestries in bgen geno and GWAS data does not match. Check the source."
            )

        log.logger.info(
            f"Detect genotype data in bgen format for {n_pop} {name_ancestry}."
        )
        geno_path = args.bgen
        geno_func = io.read_bgen
    elif args.ld is not None:
        if len(args.ld) != n_pop:
            raise ValueError(
                "The numbers of ancestries in ld geno and pheno data does not match. Check the source."
            )

        log.logger.info(f"Detect LD data in tsv files for {n_pop} {name_ancestry}.")
        geno_path = args.ld
        geno_func = io.read_ld
        ld_file = True
    else:
        raise ValueError(
            "No genotype/LD data specified in either plink, vcf, bgen, or LD files. Check the source."
        )

    if args.sample_size is not None:
        if len(args.sample_size) != n_pop:
            raise ValueError(
                "The numbers of ancestries in sample size and GWAS data does not match. Check the source"
                + " and update --sample-size."
            )

        if not all(sample_size > 0 for sample_size in args.sample_size):
            raise ValueError(
                "The sample size specified for summary-level fine-mapping is invalid. Choose a positive integer"
                + " using --sample-size."
            )

        log.logger.info(f"Detect sample sizes for {n_pop} {name_ancestry}.")
    else:
        raise ValueError(
            "No sample size specified for summary-level fine-mapping. Check the source."
        )

    if args.pi != "uniform":
        log.logger.info(
            "Detect file that contains prior weights for each SNP to be causal."
        )
        pi = pd.read_csv(args.pi, header=None, sep="\t")

        # remove dupliate rows
        pi = pi.drop_duplicates(subset=pi.columns[0])

        if pi.shape[0] == 0:
            raise ValueError(
                "No prior weights are listed in the prior file. Check the source."
            )

        if pi.shape[1] < 2:
            raise ValueError(
                "The prior file has less than 2 columns. It has to be at least two columns."
                + " The first column is the SNP ID, and the second column the prior probability."
            )

        if pi.shape[1] > 2:
            log.logger.debug(
                "The prior file has more than 2 columns. Will only use the first two columns."
            )

        pi = pi.iloc[:, 0:2]
        pi.columns = ["snp", "pi"]
    else:
        pi = pd.DataFrame()

    if args.seed <= 0:
        raise ValueError(
            "The seed specified for randomization is invalid. Choose a positive integer using --seed."
        )

    if args.maf <= 0 or args.maf > 0.5:
        raise ValueError(
            "The minor allele frequency (MAF) has to be between 0 (exclusive) and 0.5 (inclusive)."
            + " Choose a valid frequency using --maf."
        )

    if args.meta and n_pop == 1:
        log.logger.debug(
            "The number of ancestry is 1, but --meta is specified. Will skip meta or mega SuSiE."
        )

    if args.chrom is None and args.start is None and args.end is None:
        log.logger.debug(
            "No region is specified. Will use all SNPs available in the data."
        )
    elif args.chrom is not None and args.start is not None and args.end is not None:
        if args.start <= 0:
            raise ValueError(
                "The start position for the region must be greater than 0. Update with --start."
            )

        if args.end <= 0:
            raise ValueError(
                "The end position for the region must be greater than 0. Update with --end."
            )

        if args.end <= args.start:
            raise ValueError(
                "The end position for the region must be greater than --start. Update with"
                + " --start or --end."
            )

        log.logger.info(
            f"Detect region (chrom{args.chrom}:{args.start}:{args.end}) to be fine-mapped."
            + " Will only use SNPs within the region."
        )
    else:
        raise ValueError(
            "The region is not specified correctly. Please provide --chrom, --start, and --end together,"
            + " or omit all of them."
        )

    if args.gwas_sig <= 0 or args.gwas_sig > 1:
        raise ValueError(
            "The significance threshold for P-values in GWAS summary statistics must be greater than 0"
            + " and less than or equal to 1. Choose a valid number using --gwas-sig."
        )

    if args.ld_adjust > 0.1 or args.ld_adjust < 0:
        raise ValueError(
            "The LD adjustment parameter has to be greater than 0 and less than 0.1."
            + " Choose a valid number using --ld-adjust."
        )

    if args.cv:
        log.logger.debug(
            "Cross-validation is not supported for summary-level fine-mapping. This flag (--cv) will be ignored."
        )

    if args.mega:
        log.logger.debug(
            "Mega SuShiE is not supported for summary-level fine-mapping. This flag (--mega) will be ignored."
        )

    if args.her:
        log.logger.debug(
            "Heritability estimation is not supported for summary-level fine-mapping."
            + " This flag (--her) will be ignored."
        )

    log.logger.debug("Finish parameter check for summary-level fine-mapping.")

    return n_pop, pi, geno_path, geno_func, ld_file


def _main(argsv):
    # setup main parser
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subp = argp.add_subparsers(
        help="Subcommands: finemap to perform gene expression fine-mapping using SuShiE"
    )

    finemap = build_finemap_parser(subp)
    finemap.set_defaults(func=run_finemap)

    # parse arguments
    args = argp.parse_args(argsv)

    cmd_str = _get_command_string(argsv)

    version = metadata.version("sushie")

    masthead = "===================================" + os.linesep
    masthead += f"             SuShiE v{version}             " + os.linesep
    masthead += "===================================" + os.linesep

    # setup logging
    log_format = "[%(asctime)s - %(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    if args.verbose:
        log.logger.setLevel(logging.DEBUG)
    else:
        log.logger.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt=log_format, datefmt=date_format)
    log.logger.propagate = False

    # write to stdout unless quiet is set
    if not args.quiet:
        sys.stdout.write(masthead)
        sys.stdout.write(cmd_str)
        sys.stdout.write("Starting log..." + os.linesep)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(fmt)
        log.logger.addHandler(stdout_handler)

    # setup log file, but write PLINK-style command first
    disk_log_stream = open(f"{args.output}.log", "w")
    disk_log_stream.write(masthead)
    disk_log_stream.write(cmd_str)
    disk_log_stream.write("Starting log..." + os.linesep)

    disk_handler = logging.StreamHandler(disk_log_stream)
    disk_handler.setFormatter(fmt)
    log.logger.addHandler(disk_handler)

    # launch finemap
    args.func(args)

    return 0


def run_cli():
    return _main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
