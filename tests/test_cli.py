"""Tests for the sushie command-line interface."""

import subprocess

from typer.testing import CliRunner

from sushie import main

runner = CliRunner()


def test_cli_version() -> None:
    """Test sushie version command."""
    result = runner.invoke(main, ["version"])
    assert result.exit_code == 0
    assert len(result.stdout.strip()) > 0


def test_cli_help() -> None:
    """Test sushie help message."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "finemap" in result.stdout


def test_finemap_help() -> None:
    """Test finemap subcommand help message."""
    result = runner.invoke(main, ["finemap", "--help"])
    assert result.exit_code == 0
    assert "--pheno" in result.stdout
    assert "--plink" in result.stdout


def test_subprocess_version() -> None:
    """Test sushie version via subprocess."""
    result = subprocess.run(
        ["uv", "run", "sushie", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_subprocess_help() -> None:
    """Test sushie finemap help via subprocess."""
    result = subprocess.run(
        ["uv", "run", "sushie", "finemap", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--pheno" in result.stdout


