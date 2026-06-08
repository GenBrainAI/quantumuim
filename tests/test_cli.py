"""Tests for the quantumium CLI."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from quantumium.cli import main


class TestInfo:
    def test_info_returns_zero(self) -> None:
        assert main(["info"]) == 0

    def test_info_shows_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["info"])
        out = capsys.readouterr().out
        assert "quantumium" in out
        assert "Python" in out

    def test_info_shows_backends(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["info"])
        out = capsys.readouterr().out
        assert "Backends:" in out

    def test_info_shows_features(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["info"])
        out = capsys.readouterr().out
        assert "Features:" in out


class TestCheck:
    def test_check_passes(self) -> None:
        assert main(["check"]) == 0

    def test_check_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["check"])
        out = capsys.readouterr().out
        assert "PASS" in out

    def test_check_timing(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["check"])
        out = capsys.readouterr().out
        assert "s)" in out


class TestBenchmark:
    def test_benchmark_returns_zero(self) -> None:
        assert main(["benchmark", "--max-qubits", "8"]) == 0

    def test_benchmark_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["benchmark", "--max-qubits", "8"])
        out = capsys.readouterr().out
        assert "Benchmark" in out
        assert "Circuit creation" in out
        assert "GHZ" in out


class TestRun:
    def test_run_valid_qasm(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        qasm = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];

            h q[0];
            cx q[0],q[1];
        """)
        f = tmp_path / "bell.qasm"
        f.write_text(qasm)
        rc = main(["run", str(f)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "2 qubits" in out
        assert "Measurement results" in out

    def test_run_missing_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["run", "/nonexistent/file.qasm"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "not found" in err

    def test_run_bad_qasm(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / "bad.qasm"
        f.write_text("not valid qasm at all\n")
        rc = main(["run", str(f)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Error" in err

    def test_run_custom_shots(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        qasm = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];

            x q[0];
        """)
        f = tmp_path / "x.qasm"
        f.write_text(qasm)
        rc = main(["run", str(f), "--shots", "500"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "500 shots" in out


class TestCircuit:
    def test_bell_circuit(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["circuit", "H(0); CNOT(0,1)"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "2 qubits" in out
        assert "|00>" in out
        assert "|11>" in out

    def test_circuit_with_measure(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["circuit", "H(0); CNOT(0,1); measure_all()"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Measurement results" in out

    def test_circuit_unknown_gate(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["circuit", "FAKEGATE(0)"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "unknown gate" in err

    def test_circuit_bad_expression(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["circuit", "notvalid"])
        assert rc == 1


class TestHelp:
    def test_no_command_shows_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "quantumium" in out.lower()

    def test_version_flag(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_invalid_command(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent"])
        assert exc_info.value.code == 2


class TestModuleInvocation:
    def test_python_m_quantumium(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "quantumium", "info"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "quantumium" in result.stdout

    def test_python_m_quantumium_check(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "quantumium", "check"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_python_m_quantumium_version(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "quantumium", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "quantumium" in result.stdout
