"""Benchmarking suite for Quantumium circuit construction and simulation.

Measures wall-clock performance for circuit building and statevector
simulation across standard test circuits of varying sizes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT, X, Z, S, T, Rz, controlled


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    circuit_name: str
    num_qubits: int
    gate_count: int
    construction_ms: float
    simulation_ms: float
    shots_per_sec: float


class BenchmarkSuite:
    """Run construction and simulation benchmarks on standard circuits."""

    def __init__(self, repeats: int = 5, shots: int = 100):
        self._repeats = repeats
        self._shots = shots

    def _build_bell(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        return qc

    def _build_ghz(self, n: int) -> QuantumCircuit:
        qc = QuantumCircuit(n)
        qc.apply(H, 0)
        for i in range(n - 1):
            qc.apply(CNOT, 0, i + 1)
        return qc

    def _build_qft(self, n: int) -> QuantumCircuit:
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.apply(H, i)
            for j in range(i + 1, n):
                angle = np.pi / (2 ** (j - i))
                crz = controlled(Rz(angle))
                qc.apply(crz, j, i)
        return qc

    def _build_random_clifford(self, n: int) -> QuantumCircuit:
        qc = QuantumCircuit(n)
        gates_1q = [H, S, X, Z]
        for _ in range(n * 3):
            for q in range(n):
                gate = gates_1q[q % len(gates_1q)]
                qc.apply(gate, q)
            if n >= 2:
                for q in range(0, n - 1, 2):
                    qc.apply(CNOT, q, q + 1)
        return qc

    def _time_construction(self, builder, *args) -> tuple[float, QuantumCircuit]:
        times = []
        qc = None
        for _ in range(self._repeats):
            t0 = time.perf_counter()
            qc = builder(*args)
            times.append(time.perf_counter() - t0)
        median_s = sorted(times)[len(times) // 2]
        return median_s * 1000, qc

    def _time_simulation(self, qc: QuantumCircuit) -> tuple[float, float]:
        sim_repeats = min(self._repeats, 2) if qc.num_qubits >= 6 else self._repeats
        times = []
        for _ in range(sim_repeats):
            t0 = time.perf_counter()
            qc.measure_all(shots=self._shots)
            times.append(time.perf_counter() - t0)
        median_s = sorted(times)[len(times) // 2]
        shots_per_sec = self._shots / median_s if median_s > 0 else float('inf')
        return median_s * 1000, shots_per_sec

    def run_all(self) -> list[BenchmarkResult]:
        results = []
        circuits = [
            ("Bell", self._build_bell, []),
            ("GHZ-4", self._build_ghz, [4]),
            ("GHZ-6", self._build_ghz, [6]),
            ("GHZ-8", self._build_ghz, [8]),
            ("QFT-4", self._build_qft, [4]),
            ("QFT-6", self._build_qft, [6]),
            ("Clifford-4", self._build_random_clifford, [4]),
            ("Clifford-6", self._build_random_clifford, [6]),
        ]

        for name, builder, args in circuits:
            construction_ms, qc = self._time_construction(builder, *args)
            simulation_ms, shots_per_sec = self._time_simulation(qc)
            results.append(BenchmarkResult(
                circuit_name=name,
                num_qubits=qc.num_qubits,
                gate_count=len(qc.instructions),
                construction_ms=construction_ms,
                simulation_ms=simulation_ms,
                shots_per_sec=shots_per_sec,
            ))

        return results

    def print_report(self, results: list[BenchmarkResult]) -> None:
        print(f"{'Circuit':<14} {'Qubits':>6} {'Gates':>6} "
              f"{'Build(ms)':>10} {'Sim(ms)':>10} {'Shots/s':>10}")
        print("-" * 62)
        for r in results:
            print(f"{r.circuit_name:<14} {r.num_qubits:>6} {r.gate_count:>6} "
                  f"{r.construction_ms:>10.2f} {r.simulation_ms:>10.2f} "
                  f"{r.shots_per_sec:>10.0f}")


def main():
    print("Quantumium Benchmark Suite")
    print("=" * 62)
    suite = BenchmarkSuite()
    results = suite.run_all()
    suite.print_report(results)


if __name__ == "__main__":
    main()
