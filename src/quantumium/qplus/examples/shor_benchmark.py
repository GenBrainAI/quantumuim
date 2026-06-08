#!/usr/bin/env python3
"""Shor Q+ Benchmark — factor semiprimes with fewer qubits than standard Shor.

Demonstrates that ShorQPlus.factor() successfully factors semiprimes
(15, 21, 35, 77, 143) using n+2 qubits instead of 3n+1 qubits.
"""

from __future__ import annotations

import time

from quantumium.qplus.shor import ShorQPlus


SEMIPRIMES: list[tuple[int, int, int]] = [
    (15, 3, 5),
    (21, 3, 7),
    (35, 5, 7),
    (77, 7, 11),
    (143, 11, 13),
]


def benchmark_factoring() -> None:
    print("SHOR Q+ BENCHMARK")
    print("=" * 70)
    print(
        f"{'N':>5} {'n_bits':>6} {'Std Qubits':>10} {'Q+ Qubits':>10} "
        f"{'Savings':>8} {'Result':>12} {'Time':>8}"
    )
    print("-" * 70)

    shor = ShorQPlus(strategy="semiclassical")

    total_successes = 0
    for N, p_expected, q_expected in SEMIPRIMES:
        comp = ShorQPlus.qubit_comparison(N)
        n_bits = comp["n_bits"]
        std_qubits = comp["standard_shor"]
        qplus_qubits = comp["semiclassical_qft"]
        savings = std_qubits - qplus_qubits

        t0 = time.time()

        success = False
        result = shor.factor(N, max_attempts=15)
        for attempt in range(30):
            result = shor.factor(N, max_attempts=15)
            if result.success and result.factors is not None:
                p, q = sorted(result.factors)
                if p * q == N and p > 1 and q > 1:
                    success = True
                    break

        elapsed = time.time() - t0

        if success and result.factors is not None:
            p, q = sorted(result.factors)
            status = f"{p} x {q}"
            total_successes += 1
        else:
            status = "FAILED"

        print(
            f"{N:>5} {n_bits:>6} {std_qubits:>10} {qplus_qubits:>10} "
            f"{savings:>7}  {status:>12} {elapsed:>7.2f}s"
        )

    print("-" * 70)
    print(f"Results: {total_successes}/{len(SEMIPRIMES)} semiprimes factored successfully")
    print()

    print("QUBIT SAVINGS ANALYSIS")
    print("=" * 70)
    for N, _, _ in SEMIPRIMES:
        comp = ShorQPlus.qubit_comparison(N)
        pct = (1 - comp["semiclassical_qft"] / comp["standard_shor"]) * 100
        print(
            f"  N={N:>4}: {comp['standard_shor']:>3} -> {comp['semiclassical_qft']:>3} qubits "
            f"({pct:.0f}% reduction)"
        )

    print()
    print("Q+ LAYER CONTRIBUTIONS")
    print("=" * 70)
    print("  Layer 1 (Compress): Semiclassical QFT reduces counting register n -> 1 qubit")
    print("  Layer 2 (Recycle):  Measure-reset-reuse enables single counting qubit")
    print("  Layer 3 (Cut/Dist): Circuit cutting enables multi-QPU distribution")
    print(
        f"\n  Benchmark: factor {SEMIPRIMES[-1][0]} ({comp['n_bits']}-bit) "
        f"using {comp['semiclassical_qft']} qubits instead of {comp['standard_shor']}"
    )


def main() -> None:
    benchmark_factoring()


if __name__ == "__main__":
    main()
