"""Quantumium command-line interface."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np

import quantumium


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantumium",
        description="Quantumium — quantum computing SDK for Python",
    )
    parser.add_argument(
        "--version", action="version", version=f"quantumium {quantumium.__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Show version, Python version, and installed backends")

    sub.add_parser("check", help="Run a quick self-test (Bell state)")

    bench = sub.add_parser("benchmark", help="Run a condensed benchmark suite")
    bench.add_argument(
        "--max-qubits",
        type=int,
        default=16,
        help="Maximum qubit count for simulation benchmark (default: 16)",
    )

    run_p = sub.add_parser("run", help="Parse and simulate an OpenQASM 2.0 file")
    run_p.add_argument("file", type=str, help="Path to a .qasm file")
    run_p.add_argument(
        "--shots", type=int, default=1024, help="Number of measurement shots (default: 1024)"
    )

    circ = sub.add_parser("circuit", help="Build and simulate a circuit from an expression")
    circ.add_argument(
        "expression",
        type=str,
        help='Circuit expression, e.g. "H(0); CNOT(0,1); measure_all()"',
    )
    circ.add_argument(
        "--shots", type=int, default=1024, help="Number of measurement shots (default: 1024)"
    )

    return parser


def _cmd_info() -> int:
    print(f"quantumium {quantumium.__version__}")
    print(f"Python     {sys.version}")
    print()

    backends: list[tuple[str, str]] = [
        ("qiskit", "qiskit"),
        ("cirq", "cirq"),
        ("braket", "braket.circuits"),
    ]
    print("Backends:")
    for label, mod in backends:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "installed")
            print(f"  {label:10s} {ver}")
        except ImportError:
            print(f"  {label:10s} not installed")

    features: list[tuple[str, str]] = [
        ("visualization", "quantumium.visualization"),
        ("interop", "quantumium.interop"),
        ("notebooks", "jupyter"),
    ]
    print()
    print("Features:")
    for label, mod in features:
        try:
            importlib.import_module(mod)
            print(f"  {label:10s} available")
        except ImportError:
            print(f"  {label:10s} not available")

    print()
    print(f"NumPy      {np.__version__}")

    try:
        import scipy  # type: ignore[import-untyped]

        print(f"SciPy      {scipy.__version__}")
    except ImportError:
        pass

    try:
        import matplotlib

        print(f"Matplotlib {matplotlib.__version__}")
    except ImportError:
        pass

    return 0


def _cmd_check() -> int:
    from quantumium.core.circuit import QuantumCircuit
    from quantumium.core.gate import H, CNOT

    print("Running self-test...")
    t0 = time.perf_counter()

    qc = QuantumCircuit(2)
    qc.apply(H, 0)
    qc.apply(CNOT, 0, 1)
    sv = qc.statevector()

    expected_00 = 1.0 / np.sqrt(2)
    expected_11 = 1.0 / np.sqrt(2)

    ok = True
    if not np.isclose(abs(sv[0]), expected_00, atol=1e-10):
        print(f"  FAIL: |00> amplitude = {sv[0]}, expected ~{expected_00}")
        ok = False
    if not np.isclose(abs(sv[3]), expected_11, atol=1e-10):
        print(f"  FAIL: |11> amplitude = {sv[3]}, expected ~{expected_11}")
        ok = False
    if not np.isclose(abs(sv[1]), 0, atol=1e-10) or not np.isclose(abs(sv[2]), 0, atol=1e-10):
        print(f"  FAIL: |01>,|10> not zero: {sv[1]}, {sv[2]}")
        ok = False

    counts = qc.measure_all(shots=1000)
    total = sum(counts.values())
    if total != 1000:
        print(f"  FAIL: total shots = {total}, expected 1000")
        ok = False

    for key in counts:
        if key not in ("00", "11"):
            print(f"  FAIL: unexpected outcome '{key}'")
            ok = False

    elapsed = time.perf_counter() - t0

    if ok:
        print(f"  PASS  ({elapsed:.3f}s)")
        return 0
    else:
        print(f"  FAIL  ({elapsed:.3f}s)")
        return 1


def _cmd_benchmark(max_qubits: int) -> int:
    from quantumium.core.circuit import QuantumCircuit
    from quantumium.core.gate import H, CNOT, X

    print("Quantumium Benchmark Suite")
    print("=" * 60)

    results: list[tuple[str, str, str]] = []

    # 1. Circuit creation
    t0 = time.perf_counter()
    for _ in range(1000):
        qc = QuantumCircuit(5)
        qc.apply(H, 0)
        for i in range(4):
            qc.apply(CNOT, i, i + 1)
    elapsed = time.perf_counter() - t0
    results.append(("Circuit creation (1000x 5q)", f"{elapsed:.4f}s", f"{1000 / elapsed:.0f}/s"))

    # 2. Simulation at various qubit counts
    for n in [4, 8, 12, 16]:
        if n > max_qubits:
            break
        qc = QuantumCircuit(n)
        qc.apply(H, 0)
        for i in range(n - 1):
            qc.apply(CNOT, i, i + 1)
        t0 = time.perf_counter()
        qc.statevector()
        elapsed = time.perf_counter() - t0
        results.append((f"Simulate {n}-qubit GHZ", f"{elapsed:.4f}s", f"2^{n} = {2**n} amps"))

    # 3. Measurement sampling
    qc = QuantumCircuit(10)
    qc.apply(H, 0)
    for i in range(9):
        qc.apply(CNOT, i, i + 1)
    t0 = time.perf_counter()
    qc.measure_all(shots=10000)
    elapsed = time.perf_counter() - t0
    results.append(("Measure 10q x 10k shots", f"{elapsed:.4f}s", ""))

    # 4. Q+ factoring N=15
    try:
        from quantumium.qplus.shor import ShorQPlus

        shor = ShorQPlus(strategy="semiclassical")
        t0 = time.perf_counter()
        result = shor.factor(15)
        elapsed = time.perf_counter() - t0
        factors_str = f"{result.factors}" if result.factors else "failed"
        results.append(("Q+ factor N=15", f"{elapsed:.4f}s", factors_str))
    except Exception as exc:
        results.append(("Q+ factor N=15", "error", str(exc)[:30]))

    # 5. Gate application throughput
    qc = QuantumCircuit(1)
    t0 = time.perf_counter()
    for _ in range(5000):
        qc.apply(X, 0)
    elapsed = time.perf_counter() - t0
    results.append(("Gate apply (5000x single)", f"{elapsed:.4f}s", f"{5000 / elapsed:.0f}/s"))

    # Print table
    print()
    print(f"{'Benchmark':<35} {'Time':>10} {'Detail':>15}")
    print("-" * 60)
    for name, t, detail in results:
        print(f"{name:<35} {t:>10} {detail:>15}")
    print()

    return 0


def _cmd_run(file: str, shots: int) -> int:
    from quantumium.transpiler.qasm import from_qasm

    path = Path(file)
    if not path.exists():
        print(f"Error: file not found: {file}", file=sys.stderr)
        return 1
    if not path.suffix == ".qasm":
        print(f"Warning: file does not have .qasm extension: {file}", file=sys.stderr)

    qasm_str = path.read_text()

    try:
        qc = from_qasm(qasm_str)
    except ValueError as exc:
        print(f"Error parsing QASM: {exc}", file=sys.stderr)
        return 1

    print(f"Circuit: {qc.num_qubits} qubits, {len(qc.instructions)} gates")
    print()

    sv = qc.statevector()
    probs = np.abs(sv) ** 2

    print("Statevector probabilities (non-zero):")
    for i, p in enumerate(probs):
        if p > 1e-10:
            bitstring = format(i, f"0{qc.num_qubits}b")
            print(f"  |{bitstring}> : {p:.6f}")

    print()
    counts = qc.measure_all(shots=shots)
    print(f"Measurement results ({shots} shots):")
    for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  |{bitstring}> : {count:5d}  ({count / shots:.4f})")

    return 0


def _cmd_circuit(expression: str, shots: int) -> int:
    from quantumium.core.circuit import QuantumCircuit
    from quantumium.core import gate as _gate_mod

    parts = [s.strip() for s in expression.split(";") if s.strip()]

    max_qubit = 0
    parsed: list[tuple[str, list[int]]] = []
    has_measure = False

    for part in parts:
        if part.startswith("measure_all"):
            has_measure = True
            continue

        paren_idx = part.find("(")
        if paren_idx == -1:
            print(f"Error: invalid expression part: {part}", file=sys.stderr)
            return 1

        gate_name = part[:paren_idx].strip()
        args_str = part[paren_idx + 1 : part.rindex(")")].strip()
        try:
            qubit_indices = [int(a.strip()) for a in args_str.split(",")]
        except ValueError:
            print(f"Error: invalid qubit indices in: {part}", file=sys.stderr)
            return 1

        for q in qubit_indices:
            if q > max_qubit:
                max_qubit = q

        parsed.append((gate_name, qubit_indices))

    num_qubits = max_qubit + 1
    qc = QuantumCircuit(num_qubits)

    for gate_name, qubit_indices in parsed:
        gate_obj = getattr(_gate_mod, gate_name, None)
        if gate_obj is None:
            print(f"Error: unknown gate '{gate_name}'", file=sys.stderr)
            return 1

        if callable(gate_obj) and not isinstance(gate_obj, _gate_mod.Gate):
            print(
                f"Error: '{gate_name}' is a parametric gate factory, "
                f"use e.g. Rx(0, 1.5708) — angle as second arg not supported in CLI expression. "
                f"Only fixed gates are supported.",
                file=sys.stderr,
            )
            return 1

        qc.apply(gate_obj, *qubit_indices)

    sv = qc.statevector()
    probs = np.abs(sv) ** 2

    print(f"Circuit: {num_qubits} qubits, {len(qc.instructions)} gates")
    print()
    print("Statevector:")
    for i, amp in enumerate(sv):
        if abs(amp) > 1e-10:
            bitstring = format(i, f"0{num_qubits}b")
            print(f"  |{bitstring}> : {amp:.6f}  (prob {probs[i]:.6f})")

    if has_measure:
        print()
        counts = qc.measure_all(shots=shots)
        print(f"Measurement results ({shots} shots):")
        for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  |{bitstring}> : {count:5d}  ({count / shots:.4f})")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _get_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "info":
        return _cmd_info()
    elif args.command == "check":
        return _cmd_check()
    elif args.command == "benchmark":
        return _cmd_benchmark(args.max_qubits)
    elif args.command == "run":
        return _cmd_run(args.file, args.shots)
    elif args.command == "circuit":
        return _cmd_circuit(args.expression, args.shots)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
