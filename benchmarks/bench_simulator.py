"""Performance benchmark suite for Quantumium SDK.

Covers statevector simulation, density matrix simulation, MPS simulation,
VQE/QAOA optimization, and Q+ factoring. Outputs JSON results for tracking.

Usage:
    python benchmarks/bench_simulator.py [--iterations N] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from dataclasses import dataclass, field

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, CNOT, Rz, Rx, Ry, controlled
from quantumium.core.noise import NoiseModel, depolarizing


@dataclass
class BenchResult:
    name: str
    category: str
    num_qubits: int
    circuit_depth: int
    gate_count: int
    wall_time_ms: float
    peak_memory_kb: float
    iterations: int
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return sum(self.times_ms) / len(self.times_ms) if self.times_ms else self.wall_time_ms

    @property
    def std_ms(self) -> float:
        if len(self.times_ms) < 2:
            return 0.0
        mean = self.mean_ms
        return (sum((t - mean) ** 2 for t in self.times_ms) / (len(self.times_ms) - 1)) ** 0.5


def _peak_memory_kb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _time_fn(fn, iterations: int) -> tuple[list[float], float]:
    times = []
    mem_before = _peak_memory_kb()
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    mem_after = _peak_memory_kb()
    return times, mem_after - mem_before


def _circuit_depth(qc: QuantumCircuit) -> int:
    if not qc.instructions:
        return 0
    layers: dict[int, int] = {}
    for inst in qc.instructions:
        max_layer = max((layers.get(q, 0) for q in inst.target_qubits), default=0)
        for q in inst.target_qubits:
            layers[q] = max_layer + 1
    return max(layers.values()) if layers else 0


def _build_bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.apply(H, 0)
    qc.apply(CNOT, 0, 1)
    return qc


def _build_ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.apply(H, 0)
    for i in range(n - 1):
        qc.apply(CNOT, 0, i + 1)
    return qc


def _build_qft(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.apply(H, i)
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            qc.apply(controlled(Rz(angle)), j, i)
    return qc


def _build_random(n: int, depth: int, seed: int = 42) -> QuantumCircuit:
    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(n)
    gates_1q = [H, X, Rx(np.pi / 4), Ry(np.pi / 3), Rz(np.pi / 6)]
    for _ in range(depth):
        for q in range(n):
            qc.apply(gates_1q[rng.randint(len(gates_1q))], q)
        for q in range(0, n - 1, 2):
            qc.apply(CNOT, q, q + 1)
        for q in range(1, n - 1, 2):
            qc.apply(CNOT, q, q + 1)
    return qc


def _run_bench(name: str, category: str, nq: int, builder, run_fn, iterations: int) -> BenchResult:
    qc = builder()
    gc = len(qc.instructions)
    depth = _circuit_depth(qc)
    times, mem_delta = _time_fn(lambda: run_fn(qc), iterations)
    result = BenchResult(
        name=name,
        category=category,
        num_qubits=nq,
        circuit_depth=depth,
        gate_count=gc,
        wall_time_ms=sum(times) / len(times),
        peak_memory_kb=mem_delta,
        iterations=iterations,
        times_ms=times,
    )
    print(f"  {name}: {result.mean_ms:.2f} ms (±{result.std_ms:.2f})")
    return result


# --- Statevector benchmarks (tensor contraction — O(2^n) per gate) ---


def bench_statevector(iterations: int) -> list[BenchResult]:
    results = []
    configs = [
        ("Bell (2q)", 2, _build_bell),
        ("GHZ-3", 3, lambda: _build_ghz(3)),
        ("GHZ-5", 5, lambda: _build_ghz(5)),
        ("GHZ-7", 7, lambda: _build_ghz(7)),
        ("GHZ-8", 8, lambda: _build_ghz(8)),
        ("GHZ-10", 10, lambda: _build_ghz(10)),
        ("GHZ-12", 12, lambda: _build_ghz(12)),
        ("GHZ-15", 15, lambda: _build_ghz(15)),
        ("GHZ-20", 20, lambda: _build_ghz(20)),
        ("QFT-3", 3, lambda: _build_qft(3)),
        ("QFT-5", 5, lambda: _build_qft(5)),
        ("QFT-7", 7, lambda: _build_qft(7)),
        ("QFT-10", 10, lambda: _build_qft(10)),
        ("QFT-12", 12, lambda: _build_qft(12)),
        ("QFT-15", 15, lambda: _build_qft(15)),
        ("Random-5x5", 5, lambda: _build_random(5, 5)),
        ("Random-7x3", 7, lambda: _build_random(7, 3)),
        ("Random-8x3", 8, lambda: _build_random(8, 3)),
        ("Random-10x5", 10, lambda: _build_random(10, 5)),
        ("Random-12x3", 12, lambda: _build_random(12, 3)),
        ("Random-15x3", 15, lambda: _build_random(15, 3)),
        ("Random-20x3", 20, lambda: _build_random(20, 3)),
    ]
    for name, nq, builder in configs:
        results.append(
            _run_bench(name, "statevector", nq, builder, lambda qc: qc.statevector(), iterations)
        )
    return results


# --- Density matrix benchmarks (tensor contraction with Kraus channels) ---


def bench_density_matrix(iterations: int) -> list[BenchResult]:
    results = []
    noise = NoiseModel()
    noise.add_gate_noise("X", depolarizing(0.01))
    noise.add_gate_noise("H", depolarizing(0.01))
    noise.add_gate_noise("CNOT", depolarizing(0.02))

    configs = [
        ("Bell-noisy (2q)", 2, _build_bell),
        ("GHZ-3-noisy", 3, lambda: _build_ghz(3)),
        ("GHZ-5-noisy", 5, lambda: _build_ghz(5)),
        ("GHZ-7-noisy", 7, lambda: _build_ghz(7)),
        ("GHZ-10-noisy", 10, lambda: _build_ghz(10)),
        ("Random-3x5-noisy", 3, lambda: _build_random(3, 5)),
        ("Random-5x3-noisy", 5, lambda: _build_random(5, 3)),
        ("Random-7x3-noisy", 7, lambda: _build_random(7, 3)),
        ("Random-10x3-noisy", 10, lambda: _build_random(10, 3)),
    ]
    for name, nq, builder in configs:
        results.append(
            _run_bench(
                name, "density_matrix", nq, builder, lambda qc: qc.density_matrix(noise), iterations
            )
        )
    return results


# --- MPS benchmarks (scales beyond statevector limits) ---


def bench_mps(iterations: int) -> list[BenchResult]:
    from quantumium.tensor_networks.mps import mps_zero_state
    from quantumium.tensor_networks.operations import apply_single_qubit_gate, apply_two_qubit_gate

    results = []
    H_mat = H.matrix
    CNOT_mat = CNOT.matrix

    configs = [
        ("MPS-GHZ-10 (χ=8)", 10, 8),
        ("MPS-GHZ-15 (χ=8)", 15, 8),
        ("MPS-GHZ-20 (χ=8)", 20, 8),
        ("MPS-GHZ-20 (χ=16)", 20, 16),
        ("MPS-GHZ-30 (χ=8)", 30, 8),
    ]

    for name, nq, chi in configs:

        def run_mps(n=nq, bond=chi):
            mps = mps_zero_state(n, max_bond_dim=bond)
            apply_single_qubit_gate(mps, H_mat, 0)
            for i in range(n - 1):
                apply_two_qubit_gate(mps, CNOT_mat, i, i + 1)
            return mps

        times, mem_delta = _time_fn(run_mps, iterations)
        result = BenchResult(
            name=name,
            category="mps",
            num_qubits=nq,
            circuit_depth=nq,
            gate_count=nq,
            wall_time_ms=sum(times) / len(times),
            peak_memory_kb=mem_delta,
            iterations=iterations,
            times_ms=times,
        )
        results.append(result)
        print(f"  {name}: {result.mean_ms:.2f} ms (±{result.std_ms:.2f})")

    return results


# --- VQE/QAOA benchmarks ---


def bench_optimization(iterations: int) -> list[BenchResult]:
    from quantumium.algorithms.vqe import vqe
    from quantumium.algorithms.qaoa import qaoa_maxcut
    from quantumium.applications.chemistry import _build_hamiltonian

    results = []

    H2_coeffs = {
        "II": -0.8105,
        "ZI": +0.1721,
        "IZ": -0.2257,
        "ZZ": +0.1709,
        "XX": +0.0453,
        "YY": +0.0453,
    }
    ham = _build_hamiltonian(H2_coeffs, 2)

    def run_vqe():
        return vqe(num_qubits=2, hamiltonian=ham, ansatz_depth=1, max_iter=30, num_starts=1)

    times, mem_delta = _time_fn(run_vqe, iterations)
    results.append(
        BenchResult(
            name="VQE-H2 (2q, depth=1, 30 iter)",
            category="optimization",
            num_qubits=2,
            circuit_depth=1,
            gate_count=0,
            wall_time_ms=sum(times) / len(times),
            peak_memory_kb=mem_delta,
            iterations=iterations,
            times_ms=times,
        )
    )
    print(f"  VQE-H2: {results[-1].mean_ms:.2f} ms (±{results[-1].std_ms:.2f})")

    edges_4 = [(0, 1), (1, 2), (2, 3), (3, 0)]

    def run_qaoa_4():
        return qaoa_maxcut(4, edges_4, p=1, max_iter=20)

    times, mem_delta = _time_fn(run_qaoa_4, iterations)
    results.append(
        BenchResult(
            name="QAOA-MaxCut-4 (p=1, 20 iter)",
            category="optimization",
            num_qubits=4,
            circuit_depth=1,
            gate_count=0,
            wall_time_ms=sum(times) / len(times),
            peak_memory_kb=mem_delta,
            iterations=iterations,
            times_ms=times,
        )
    )
    print(f"  QAOA-MaxCut-4: {results[-1].mean_ms:.2f} ms (±{results[-1].std_ms:.2f})")

    edges_6 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]

    def run_qaoa_6():
        return qaoa_maxcut(6, edges_6, p=1, max_iter=20)

    times, mem_delta = _time_fn(run_qaoa_6, iterations)
    results.append(
        BenchResult(
            name="QAOA-MaxCut-6 (p=1, 20 iter)",
            category="optimization",
            num_qubits=6,
            circuit_depth=1,
            gate_count=0,
            wall_time_ms=sum(times) / len(times),
            peak_memory_kb=mem_delta,
            iterations=iterations,
            times_ms=times,
        )
    )
    print(f"  QAOA-MaxCut-6: {results[-1].mean_ms:.2f} ms (±{results[-1].std_ms:.2f})")

    return results


# --- Q+ Shor benchmarks ---


def bench_qplus(iterations: int) -> list[BenchResult]:
    from quantumium.qplus.shor import ShorQPlus

    results = []
    semiprimes = [15, 21, 35, 77, 143]

    for N in semiprimes:
        sqp = ShorQPlus()

        def run_factor(n=N):
            return sqp.factor(n, max_attempts=10)

        times, mem_delta = _time_fn(run_factor, iterations)
        result = BenchResult(
            name=f"Q+-Shor N={N}",
            category="qplus",
            num_qubits=N.bit_length() + 2,
            circuit_depth=0,
            gate_count=0,
            wall_time_ms=sum(times) / len(times),
            peak_memory_kb=mem_delta,
            iterations=iterations,
            times_ms=times,
        )
        results.append(result)
        print(f"  Q+-Shor N={N}: {result.mean_ms:.2f} ms (±{result.std_ms:.2f})")

    return results


def run_all(iterations: int = 3) -> dict:
    print(f"\n{'=' * 60}")
    print("Quantumium SDK Performance Benchmark Suite")
    print(f"{'=' * 60}")
    print(f"Iterations per benchmark: {iterations}")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy: {np.__version__}")
    print()

    all_results: list[BenchResult] = []

    print("--- Statevector Simulation (max 8q, full matrix) ---")
    all_results.extend(bench_statevector(iterations))

    print("\n--- Density Matrix Simulation (max 7q, noisy) ---")
    all_results.extend(bench_density_matrix(iterations))

    print("\n--- MPS Simulation (10-30q, bounded bond dim) ---")
    all_results.extend(bench_mps(iterations))

    print("\n--- VQE / QAOA Optimization ---")
    all_results.extend(bench_optimization(iterations))

    print("\n--- Q+ Shor Factoring ---")
    all_results.extend(bench_qplus(iterations))

    output = {
        "metadata": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "iterations": iterations,
            "note": "Statevector uses full O(4^n) matrix multiply; MPS scales beyond.",
        },
        "results": [
            {
                "name": r.name,
                "category": r.category,
                "num_qubits": r.num_qubits,
                "circuit_depth": r.circuit_depth,
                "gate_count": r.gate_count,
                "mean_ms": round(r.mean_ms, 3),
                "std_ms": round(r.std_ms, 3),
                "peak_memory_delta_kb": r.peak_memory_kb,
                "times_ms": [round(t, 3) for t in r.times_ms],
            }
            for r in all_results
        ],
    }

    print(f"\n{'=' * 60}")
    print(f"Total benchmarks: {len(all_results)}")
    print(f"{'=' * 60}\n")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantumium SDK Benchmark Suite")
    parser.add_argument(
        "--iterations", "-n", type=int, default=3, help="Iterations per benchmark (default: 3)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="benchmarks/baseline_results.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    output = run_all(args.iterations)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
