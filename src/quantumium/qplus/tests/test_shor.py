"""Tests for Q+ Shor's algorithm — the full-stack integration."""

from __future__ import annotations

import pytest

from quantumium.qplus.shor import ShorQPlus, QplusResult


class TestShorQPlusInit:
    def test_default_strategy(self) -> None:
        shor = ShorQPlus()
        assert shor.strategy == "auto"

    def test_explicit_strategies(self) -> None:
        for s in ("auto", "semiclassical", "recycled", "distributed"):
            shor = ShorQPlus(strategy=s)
            assert shor.strategy == s

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            ShorQPlus(strategy="magic")


class TestShorQPlusTrivial:
    def test_factor_even(self) -> None:
        shor = ShorQPlus()
        result = shor.factor(6)
        assert result.success
        assert result.factors is not None
        p, q = result.factors
        assert p * q == 6

    def test_factor_4(self) -> None:
        shor = ShorQPlus()
        result = shor.factor(4)
        assert result.success
        assert result.factors == (2, 2)

    def test_factor_too_small(self) -> None:
        shor = ShorQPlus()
        with pytest.raises(ValueError, match="N must be >= 4"):
            shor.factor(3)

    def test_trivial_gcd(self) -> None:
        shor = ShorQPlus()
        result = shor.factor(15, a=3)
        assert result.success
        assert result.factors is not None
        p, q = result.factors
        assert p * q == 15
        assert "trivial_gcd" in result.layers_used


class TestShorQPlusQuantum:
    """Tests that use actual quantum order finding."""

    def test_factor_15(self) -> None:
        shor = ShorQPlus(strategy="semiclassical")
        successes = 0
        for _ in range(20):
            result = shor.factor(15, a=7, max_attempts=1)
            if result.success and result.factors is not None:
                p, q = result.factors
                if p * q == 15:
                    successes += 1
        assert successes >= 3, f"Only {successes}/20 succeeded"

    def test_factor_15_recycled(self) -> None:
        shor = ShorQPlus(strategy="recycled")
        successes = 0
        for _ in range(20):
            result = shor.factor(15, a=7, max_attempts=1)
            if result.success and result.factors is not None:
                p, q = result.factors
                if p * q == 15:
                    successes += 1
        assert successes >= 3, f"Only {successes}/20 succeeded"

    def test_factor_21(self) -> None:
        shor = ShorQPlus(strategy="semiclassical")
        successes = 0
        for _ in range(20):
            result = shor.factor(21, a=2)
            if result.success and result.factors is not None:
                p, q = result.factors
                if p * q == 21:
                    successes += 1
        assert successes >= 2, f"Only {successes}/20 succeeded for 21"

    def test_factor_35(self) -> None:
        shor = ShorQPlus(strategy="semiclassical")
        successes = 0
        for _ in range(20):
            result = shor.factor(35, a=2)
            if result.success and result.factors is not None:
                p, q = result.factors
                if p * q == 35:
                    successes += 1
        assert successes >= 2, f"Only {successes}/20 succeeded for 35"

    def test_factor_auto_strategy(self) -> None:
        shor = ShorQPlus(strategy="auto")
        result = shor.factor(15, a=3)
        assert result.success


class TestShorQPlusQubitSavings:
    def test_qubit_savings_15(self) -> None:
        shor = ShorQPlus(strategy="semiclassical")
        result = shor.factor(15, a=7)
        if result.success and result.qubits_used > 0:
            assert result.qubit_savings > 0
            assert result.qubits_used < result.qubits_standard

    def test_qubit_comparison(self) -> None:
        comp = ShorQPlus.qubit_comparison(15)
        assert comp["standard_shor"] > comp["semiclassical_qft"]
        assert comp["qplus_recycled"] == comp["semiclassical_qft"]
        assert comp["n_bits"] == 4

    def test_qubit_comparison_21(self) -> None:
        comp = ShorQPlus.qubit_comparison(21)
        assert comp["n_bits"] == 5
        assert comp["standard_shor"] == 16  # 2*5+1+5
        assert comp["semiclassical_qft"] == 7  # 5+2

    def test_qubit_comparison_scaling(self) -> None:
        for N in [15, 21, 35, 77, 143, 323]:
            comp = ShorQPlus.qubit_comparison(N)
            assert comp["semiclassical_qft"] < comp["standard_shor"]
            ratio = comp["semiclassical_qft"] / comp["standard_shor"]
            assert ratio < 0.6  # always at least 40% savings


class TestQplusResult:
    def test_result_structure(self) -> None:
        result = QplusResult(
            N=15,
            factors=(3, 5),
            success=True,
            qubits_used=6,
            qubits_standard=13,
            qubit_savings=7,
            order=4,
            base=7,
            phase=0.25,
            strategy="semiclassical",
            attempts=1,
            layers_used=["compress"],
        )
        assert result.N == 15
        assert result.factors == (3, 5)
        assert result.qubit_savings == 7

    def test_result_defaults(self) -> None:
        result = QplusResult(
            N=15,
            factors=None,
            success=False,
            qubits_used=6,
            qubits_standard=13,
            qubit_savings=7,
        )
        assert result.order is None
        assert result.layers_used == []


class TestShorQPlusMultipleNumbers:
    """Test factoring across multiple semiprimes."""

    @pytest.mark.parametrize(
        "N,expected_factors",
        [
            (15, {(3, 5), (5, 3)}),
            (21, {(3, 7), (7, 3)}),
        ],
    )
    def test_factor_semiprimes_reliable(
        self, N: int, expected_factors: set[tuple[int, int]]
    ) -> None:
        shor = ShorQPlus(strategy="recycled")
        successes = 0
        for _ in range(30):
            result = shor.factor(N, max_attempts=10)
            if result.success and result.factors is not None:
                if result.factors in expected_factors:
                    successes += 1
        assert successes >= 3, f"Only {successes}/30 for N={N}"

    def test_factor_77(self) -> None:
        shor = ShorQPlus(strategy="semiclassical")
        successes = 0
        for _ in range(20):
            result = shor.factor(77, max_attempts=15)
            if result.success and result.factors is not None:
                p, q = result.factors
                if p * q == 77:
                    successes += 1
        assert successes >= 1, f"Only {successes}/20 for N=77"

    def test_factor_143(self) -> None:
        shor = ShorQPlus(strategy="semiclassical")
        successes = 0
        for _ in range(20):
            result = shor.factor(143, max_attempts=15)
            if result.success and result.factors is not None:
                p, q = result.factors
                if p * q == 143:
                    successes += 1
        assert successes >= 1, f"Only {successes}/20 for N=143"
