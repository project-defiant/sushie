"""CLI option store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer


@dataclass
class Plink1Files:
    """Container for the three files behind a plink 1 prefix."""

    bed: Path
    bim: Path
    fam: Path

    @classmethod
    def from_prefix_path(cls, path_prefix: Path) -> Plink1Files:
        """Construct plink file paths from a path prefix."""
        return cls(
            bed=path_prefix.parent / f"{path_prefix.name}.bed",
            bim=path_prefix.parent / f"{path_prefix.name}.bim",
            fam=path_prefix.parent / f"{path_prefix.name}.fam",
        )

    def exist(self) -> bool:
        """Check whether all expected plink files exist."""
        return (
            self.bed.exists()
            and self.bed.is_file()
            and self.bim.exists()
            and self.bim.is_file()
            and self.fam.exists()
            and self.fam.is_file()
        )


def _ensure_plink_files(path_prefixes: list[Path] | None) -> list[Path] | None:
    """Validate that each provided plink prefix has bed/bim/fam files."""
    if path_prefixes is None:
        return None

    for path_prefix in path_prefixes:
        if not Plink1Files.from_prefix_path(path_prefix).exist():
            raise FileNotFoundError(
                f"Not all plink files found for prefix: {path_prefix}"
            )
    return path_prefixes


def _help(*parts: str) -> str:
    """Join help fragments and remove embedded quote characters."""
    text = " ".join(part.strip() for part in parts if part)
    text = " ".join(text.split())
    return text.replace('"', "").replace("'", "").replace("`", "")


class OptionStore:
    """Option store for the CLI."""

    def __init__(self) -> None:
        """Initialize all Typer options used by the finemap command."""
        self.options = {
            "summary": typer.Option(
                help=_help(
                    """
                    Indicator whether to run fine-mapping using summary statistics.

                    If True, the software will need GWAS files as input data by specifying --gwas
                    and need LD matrix by specifying either --ld or one of the --plink, --vcf, or --bgen.

                    If False (default), the software will need phenotype data by specifying --pheno,
                    and genotype data by specifying either --plink, --vcf, or --bgen.)}
                    """
                )
            ),
            "pheno": typer.Option(
                help=_help(
                    """
                    Phenotype data.

                    It has to be a gzip compressed tsv file that contains at least two columns where the first column is subject ID and the second column is the continuous phenotypic value.

                    It is okay to have additional columns, but only the first two columns will be used.
                    No headers. Use space to separate ancestries if more than two.
                    For individual-level data fine-mapping, SuShiE currently only fine-maps on continuous data.
                    """
                )
            ),
            "gwas": typer.Option(
                help=_help(
                    """
                    GWAS data.

                    It has to be a tsv file that contains at least six columns [chromosome, snpId, position, effectAllele, otherAllele, z-score].

                    Users can can specify column names using --gwas-header flag.
                    The chromsome number needs to be integer number instead of chr1 or chrom1.
                    By default, the software assumes the header names are chrom, snp, pos, a1, a0, z.
                    It can be a compressed file (e.g., tsv.gz).,
                    It is okay to have additional columns, but only the mentioned columns will be used.
                    Use space to separate ancestries if more than two.,
                    """
                )
            ),
            "parquet": typer.Option(
                help=_help("""Provide GWAS data in parquet file.""")
            ),
            "plink": typer.Option(
                help=_help(
                    """
                    Genotype data in plink 1 format.
                    The plink triplet (bed, bim, and fam) should be in the same folder with the same prefix.

                    Use space to separate ancestries if more than two.
                    Keep the same ancestry order as phenotypes.
                    SuShiE currently does not take plink 2 format.
                    Data has to only contain bialleic variant.
                    If used in summary-level fine-mapping, the SNP ID has to match the GWAS data in --gwas.
                    The software will flip the alleles if the counting allele in GWAS data is different from the plink data.
                    """
                ),
                callback=_ensure_plink_files,
            ),
            "vcf": typer.Option(
                help=_help(
                    """
                    Genotype data in vcf format.
                    Use space to separate ancestries if more than two.
                    Keep the same ancestry order as phenotypes. The software will count RFE allele.
                    If gt_types is UNKNOWN, it will be coded as NA, and be imputed by allele frequency.
                    Data has to only contain bialleic variant.
                    If used in summary-level fine-mapping, the SNP ID has to match the GWAS data in --gwas.
                    The software will flip the alleles if the counting allele in GWAS data is different from the plink data.
                    """
                )
            ),
            "bgen": typer.Option(
                help=_help(
                    """
                    Genotype data in bgen 1.3 format.
                    Use space to separate ancestries if more than two.
                    Keep the same ancestry order as phenotypes.
                    Data has to only contain bialleic variant.
                    If used in summary-level fine-mapping, the SNP ID has to match the GWAS data in --gwas.
                    The software will flip the alleles if the counting allele in GWAS data is different from the plink data.
                    """
                )
            ),
            "ancestry_index": typer.Option(
                help=_help(
                    """
                    Single file that contains subject ID and their ancestry index.
                    It has to be a tsv file that contains at least two columns where the,
                    first column is the subject ID and the second column is the ancestry index,
                    starting from 1 (e.g., 1, 2, 3 etc.). It can be a compressed file (e.g., tsv.gz).
                    Only the first two columns will be used. No headers.
                    If this file is specified, it assumes that all the phenotypes across ancestries are in one single file,
                    and same thing for genotypes and covariates data.
                    It will produce errors if multiple phenotype, genotype, and covariates are specified.
                    """
                )
            ),
            "keep": typer.Option(
                help=_help(
                    """
                    Single file that contains subject ID across all ancestries that are used for fine-mapping.
                    It has to be a tsv file that contains at least one columns where the,
                    first column is the subject ID. It can be a compressed file (e.g., tsv.gz). No headers.
                    If this file is specified, all phenotype, genotype, and covariates data will be filtered down to, the subjects listed in it.
                    """
                )
            ),
            "covar": typer.Option(
                help=_help(
                    """
                    Covariates that will be accounted in the fine-mapping.
                    It has to be a tsv file that contains at least two columns where the, first column is the subject ID. It can be a compressed file (e.g., tsv.gz),
                    All the columns will be counted. Use space to separate ancestries if more than two.
                    Keep the same ancestry order as phenotypes.
                    Pre-converting the categorical covariates into dummy variables is required. No headers.
                    If your categorical covariates have n levels, make sure the dummy variables have n-1 columns.
                    """
                )
            ),
            "ld": typer.Option(
                help=_help(
                    """
                    LD files that will be used in the fine-mapping.
                    Keep the same ancestry order as GWAS files.
                    It has to be a tsv or comparessed file (e.g., tsv.gz).
                    The header has to be the SNP name matching the GWAS data in --gwas.
                    It can have less or more SNPs than the GWAS data, and the software will find the overlap SNPs.
                    Users must ensure that the LD and GWAS z statistics are computed using the same counting alleles.
                    """
                )
            ),
            "chrom": typer.Option(
                min=1,
                max=22,
                help=_help(
                    "Chromsome number to subset SNPs in the fine-mapping. Default is None."
                    " Value has to be an integer number between 1 and 22.",
                    " The SNP position information from the first ancestry will be used for filtering.",
                    " If this flag is specified, --start and --end must also be provided.",
                ),
            ),
            "start": typer.Option(
                help=_help(
                    "Base-pair start position to subset SNPs in the fine-mapping. Default is None."
                    " Value has to be a positive integer number.",
                    " The SNP position information from the first ancestry will be used for filtering.",
                    " If this flag is specified, --chrom and --end must also be provided.",
                )
            ),
            "end": typer.Option(
                help=_help(
                    "Base-pair end position to subset SNPs in the fine-mapping. Default is None."
                    " Value has to be a positive integer number and larger than --start.",
                    " The SNP position information from the first ancestry will be used for filtering.",
                    " If this flag is specified, --chrom and --start must also be provided.",
                )
            ),
            "sample_size": typer.Option(
                help=_help(
                    "GWAS sample size of each ancestry. Default is None."
                    " Values have to be positive integer. Use space to separate ancestries if more than two.",
                    " The order has to be the same as the GWAS data in --gwas.",
                )
            ),
            "gwas_header": typer.Option(
                help=_help(
                    "GWAS file header names. Default is [chrom, snp, pos, a1, a0, z].",
                    " Users can specify the header names for the GWAS data in this order.",
                )
            ),
            "gwas_sig": typer.Option(
                help=_help(
                    "The significance threshold for SNPs to be included in the fine-mapping.",
                    " Default is 1.0. Only SNPs with P value less than this threshold will be included.",
                    " It has to be a float number between 0 and 1.",
                )
            ),
            "gwas_sig_type": typer.Option(
                help=_help(
                    "The cases how to include significant SNPs in the fine-mapping across ancestries.",
                    " If it is at-least, the software will include SNPs that are significant in at least one ancestry.",
                    " If it is all, the software will include SNPs that are significant in all ancestries.",
                    " Default is at-least.",
                    " The significant threshold is specified by --gwas-sig.",
                )
            ),
            "L": typer.Option(
                "--L",
                help=_help(
                    "Integer number of shared effects pre-specified.",
                    " Default is 10. Larger number may cause slow inference.",
                ),
            ),
            "pi": typer.Option(
                help=_help(
                    "Prior probability for each SNP to be causal.",
                    " Default is uniform (i.e., 1/p where p is the number of SNPs in the region.",
                    " It is the fixed across all ancestries.",
                    " Alternatively, users can specify the file path that contains the prior weights for each SNP.",
                    " The weights have to be positive value.",
                    " The weights will be normalized to sum to 1 before inference.",
                    " The file has to be a tsv file that contains two columns where the",
                    " first column is the SNP ID and the second column is the prior weights.",
                    " Additional columns will be ignored.",
                    " For SNPs do not have prior weights in the file, it will be assigned the average value of the rest.",
                    " It can be a compressed file (e.g., tsv.gz). No headers.",
                )
            ),
            "resid_var": typer.Option(
                help=_help(
                    "Specify the prior for the residual variance for ancestries. Default is 1e-3 for each ancestry.",
                    " Values have to be positive. Use space to separate ancestries if more than two.",
                )
            ),
            "effect_var": typer.Option(
                help=_help(
                    "Specify the prior for the causal effect size variance for ancestries. Default is 1e-3 for each ancestry.",
                    " Values have to be positive. Use space to separate ancestries if more than two.",
                    " If --no-update is specified and --rho is not, specifying this parameter will",
                    " only keep effect_var as prior through optimizations and update rho.",
                    " If --effect-var, --rho, and --no-update all three are specified, both --effect-var and --rho",
                    " will be fixed as prior through optimizations.",
                    " If --no-update is specified, but neither --effect-var nor --rho,",
                    " both --effect-var and --rho will be fixed as default prior value through optimizations.",
                )
            ),
            "rho": typer.Option(
                help=_help(
                    "Specify the prior for the effect correlation for ancestries. Default is 0.1 for each pair of ancestries.",
                    " Use space to separate ancestries if more than two. Each rho has to be a float number between -1 and 1.",
                    " If there are N > 2 ancestries, X = choose(N, 2) is required.",
                    " The rho order has to be rho(1,2), ..., rho(1, N), rho(2,3), ..., rho(N-1. N).",
                    " If --no-update is specified and --effect-var is not, specifying this parameter will",
                    " only fix rho as prior through optimizations and update effect-var.",
                    " If --effect-var, --rho, and --no-update all three are specified, both --effect-var and --rho",
                    " will be fixed as prior through optimizations.",
                    " If --no-update is specified, but neither --effect-var nor --rho,",
                    " both --effect-var and --rho will be fixed as default prior value through optimizations.",
                )
            ),
            "no_scale": typer.Option(
                help=_help(
                    "Indicator to scale the genotype and phenotype data by standard deviation.",
                    " Default is False (to scale)."
                    " Specify --no-scale will store True value, and may cause different inference.",
                )
            ),
            "no_regress": typer.Option(
                help=_help(
                    "Indicator to regress the covariates on each SNP. Default is False (to regress).",
                    " Specify --no-regress will store True value.",
                    " It may slightly slow the inference, but can be more accurate.",
                )
            ),
            "no_update": typer.Option(
                help=_help(
                    "Indicator to update effect covariance prior before running single effect regression.",
                    " Default is False (to update).",
                    " Specify --no-update will store True value. The updating algorithm is similar to EM algorithm",
                    " that computes the prior covariance conditioned on other parameters.",
                    " See the manuscript for more information.",
                )
            ),
            "max_iter": typer.Option(
                help=_help(
                    "Maximum iterations for the optimization. Default is 500.",
                    " Larger number may slow the inference while smaller may cause different inference.",
                )
            ),
            "min_tol": typer.Option(
                help=_help(
                    "Minimum tolerance for the convergence. Default is 1e-3.",
                    " Smaller number may slow the inference while larger may cause different inference.",
                )
            ),
            "threshold": typer.Option(
                help=_help(
                    "Specify the PIP threshold for SNPs to be included in the credible sets. Default is 0.95.",
                    " It has to be a float number between 0 and 1.",
                )
            ),
            "purity": typer.Option(
                help=_help(
                    "Specify the purity threshold for credible sets to be output. Default is 0.5.",
                    " It has to be a float number between 0 and 1.",
                )
            ),
            "purity_method": typer.Option(
                help=_help(
                    "Specify the method to compute purity across ancestries.",
                    " Users choose weighted, max, or min.",
                    " weighted is the sum of the purity of each ancestry weighted by the sample size.",
                    " max is the maximum purity value across ancestries.",
                    " min is the minimum purity value across ancestries.",
                    " Default is weighted.",
                )
            ),
            "ld_adjust": typer.Option(
                help=_help(
                    "The adjusting number to LD diagonal to ensure the positive definiteness.",
                    " It has to be positive integer number between 0 and 0.1. Default is 0.",
                )
            ),
            "max_select": typer.Option(
                help=_help(
                    "The maximum selected number of SNPs to calculate the purity. Default is 250.",
                    " It has to be positive integer number. A larger number can unnecessarily spend much memory.",
                )
            ),
            "min_snps": typer.Option(
                help=_help(
                    "The minimum number of SNPs to fine-map. Default is 100.",
                    " It has to be positive integer number.",
                )
            ),
            "maf": typer.Option(
                help=_help(
                    "Threshold for minor allele frequency (MAF) to filter out SNPs for each ancestry.",
                    " It has to be a float between 0 (exclusive) and 0.5 (inclusive).",
                )
            ),
            "rint": typer.Option(
                help=_help(
                    "Indicator to perform rank inverse normalization transformation (rint) for each phenotype data.",
                    " Default is False (do not transform).",
                    " Specify --rint will store True value.",
                    " We suggest users to do this QC during data preparation.",
                )
            ),
            "no_reorder": typer.Option(
                help=_help(
                    "Indicator to re-order single effects based on Frobenius norm of effect size covariance prior."
                    " Default is False (to re-order).",
                    " Specify --no-reorder will store True value.",
                )
            ),
            "keep_ambiguous": typer.Option(
                help=_help(
                    "Indicator to keep ambiguous SNPs (i.e., A/T, T/A, C/G, or G/C pairs) from the genotypes.",
                    " Recommend to remove these SNPs if each ancestry data is from different studies",
                    " or plan to use the inference results for downstream analysis with other datasets."
                    " Default is False (not to keep).",
                    " Specify --keep-ambiguous will store True value.",
                )
            ),
            "meta": typer.Option(
                help=_help(
                    "Indicator to perform single-ancestry SuShiE followed by meta analysis of the results.",
                    " Default is False. Specify --meta will store True value and increase running time.",
                    " Specifying one ancestry in phenotype and genotype parameter will ignore --meta.",
                )
            ),
            "mega": typer.Option(
                help=_help(
                    "Indicator to perform mega SuShiE that run single-ancestry SuShiE on",
                    " genotype and phenotype data that is row-wise stacked across ancestries.",
                    " After row-binding phenotype data, it will perform rank inverse normalization transformation.",
                    " Default is False. Specify --mega will store True value and increase running time.",
                    " Specifying one ancestry in phenotype and genotype parameter will ignore --mega.",
                )
            ),
            "her": typer.Option(
                help=_help(
                    "Indicator to perform heritability (h2g) analysis using limix. Default is False.",
                    " Specify --her will store True value and increase running time.",
                    " It estimates h2g with two definitions. One is with variance of fixed terms (original limix definition),",
                    " and the other is without variance of fixed terms (gcta definition).",
                    " It also estimates these two definitions h2g using using all genotypes,",
                    " and using only SNPs in the credible sets.",
                )
            ),
            "cv": typer.Option(
                help=_help(
                    "Indicator to perform cross validation (CV) and output CV results (adjusted r-squared and its p-value)",
                    " for future FUSION pipeline. Default is False. ",
                    " Specify --cv will store True value and increase running time.",
                )
            ),
            "cv_num": typer.Option(
                help=_help(
                    "The number of fold cross validation. Default is 5.",
                    " It has to be a positive integer number. Larger number may cause longer running time.",
                )
            ),
            "seed": typer.Option(
                help=_help(
                    "The seed for randomization. It can be used to cut data sets in cross validation. ",
                    " It can also be used to randomly select SNPs in the credible sets to calculate the purity."
                    " Default is 12345. It has to be positive integer number.",
                )
            ),
            "alphas": typer.Option(
                help=_help(
                    "Indicator to output all the cs (alphas) results before pruning for purity",
                    " including PIPs, alphas, whether in cs, across all L.",
                    " Default is False. Specify --alphas will store True value and increase running time.",
                )
            ),
            "numpy": typer.Option(
                help=_help(
                    "Indicator to output all the results in *.npy file.",
                    " Default is False. Specify --numpy will store True value and increase running time.",
                    " *.npy file contains all the inference results including credible sets, pips, priors and posteriors",
                    " for your own post-hoc analysis.",
                )
            ),
            "trait": typer.Option(
                help=_help(
                    "Trait, tissue, gene name of the phenotype for better indexing in post-hoc analysis. Default is Trait.",
                )
            ),
            "quiet": typer.Option(
                help=_help(
                    "Indicator to not print message to console. Default is False. Specify --quiet will store True value.",
                )
            ),
            "verbose": typer.Option(
                help=_help(
                    "Indicator to include debug information in the log. Default is False.",
                    " Specify --verbose will store True value.",
                )
            ),
            "compress": typer.Option(
                help=_help(
                    "Indicator to compress all output tsv files in tsv.gz.",
                    " Default is False. Specify --compress will store True value to save disk space.",
                    " This command will not compress *.npy files.",
                )
            ),
            "platform": typer.Option(
                help=_help(
                    "Indicator for the JAX platform. It has to be cpu, gpu, or tpu. Default is cpu.",
                )
            ),
            "jax_precision": typer.Option(
                help=_help(
                    "Indicator for the JAX precision: 64-bit or 32-bit.",
                    " Default is 64-bit. Choose 32-bit may cause elbo decreases warning.",
                )
            ),
            "output": typer.Option(
                help=_help(
                    "Prefix for output files. Default is sushie_finemap.",
                )
            ),
        }

    def get(self, name: str):
        """Get a stored option by name."""
        option = self.options.get(name)
        if option is None:
            raise ValueError(f"Option {name} not found in OptionStore.")
        return option
