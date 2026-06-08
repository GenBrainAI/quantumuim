# Command-Line Interface

After installing quantumium, the `quantumium` command is available in your terminal.

```bash
pip install quantumium
quantumium --version
```

You can also invoke it as a Python module:

```bash
python -m quantumium --version
```

## Subcommands

### `quantumium info`

Show version, Python version, installed backends, and available features.

```bash
$ quantumium info
quantumium 0.3.0
Python     3.11.9

Backends:
  qiskit     1.2.0
  cirq       not installed
  braket     not installed

Features:
  visualization available
  interop       available
  notebooks     not available

NumPy      1.26.4
```

### `quantumium check`

Run a quick self-test: creates a Bell state, verifies the statevector
amplitudes, and runs measurement sampling.

```bash
$ quantumium check
Running self-test...
  PASS  (0.003s)
```

Exits with code 0 on success, 1 on failure.

### `quantumium benchmark`

Run a condensed benchmark suite covering circuit creation, simulation at
various qubit counts, measurement sampling, and Q+ factoring.

```bash
$ quantumium benchmark
Quantumium Benchmark Suite
============================================================

Benchmark                              Time         Detail
------------------------------------------------------------
Circuit creation (1000x 5q)          0.1234s       8100/s
Simulate 4-qubit GHZ                0.0002s    2^4 = 16 amps
Simulate 8-qubit GHZ                0.0005s   2^8 = 256 amps
Simulate 12-qubit GHZ               0.0030s  2^12 = 4096 amps
Simulate 16-qubit GHZ               0.0450s 2^16 = 65536 amps
Measure 10q x 10k shots             0.0120s
Q+ factor N=15                      0.0300s         (3, 5)
Gate apply (5000x single)           0.0200s     250000/s
```

Use `--max-qubits` to limit simulation size:

```bash
quantumium benchmark --max-qubits 12
```

### `quantumium run <file.qasm>`

Parse and simulate an OpenQASM 2.0 file. Prints the non-zero statevector
probabilities and measurement results.

```bash
$ quantumium run examples/bell.qasm
Circuit: 2 qubits, 2 gates

Statevector probabilities (non-zero):
  |00> : 0.500000
  |11> : 0.500000

Measurement results (1024 shots):
  |00> :   518  (0.5059)
  |11> :   506  (0.4941)
```

Options:

- `--shots N` — Number of measurement shots (default: 1024)

### `quantumium circuit <expression>`

Build a circuit from a semicolon-separated gate expression and show the
statevector. Supported gates: H, X, Y, Z, S, T, CNOT, CZ, SWAP, CCX.

```bash
$ quantumium circuit "H(0); CNOT(0,1)"
Circuit: 2 qubits, 2 gates

Statevector:
  |00> : 0.707107+0.000000j  (prob 0.500000)
  |11> : 0.707107+0.000000j  (prob 0.500000)
```

Add `measure_all()` to include measurement results:

```bash
$ quantumium circuit "H(0); CNOT(0,1); measure_all()" --shots 2000
Circuit: 2 qubits, 2 gates

Statevector:
  |00> : 0.707107+0.000000j  (prob 0.500000)
  |11> : 0.707107+0.000000j  (prob 0.500000)

Measurement results (2000 shots):
  |00> :  1012  (0.5060)
  |11> :   988  (0.4940)
```
