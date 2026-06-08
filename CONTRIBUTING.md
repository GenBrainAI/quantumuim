# Contributing to Quantumium

Thank you for your interest in contributing to Quantumium! This guide covers
everything you need to get started.

## Development Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/GenBrainAI/quantumuim.git
   cd quantumuim
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows
   ```

3. **Install in development mode:**

   ```bash
   pip install -e ".[dev]"
   ```

   This installs the SDK along with testing and linting tools (pytest, ruff,
   mypy, hypothesis, coverage).

4. **Install optional backend dependencies** (for interop development):

   ```bash
   pip install -e ".[qiskit,cirq]"
   ```

## Project Structure

```
quantumuim/
├── src/quantumium/         # Source code
│   ├── core/               # Qubits, gates, circuits, noise, parameters
│   ├── algorithms/         # Quantum algorithms (Grover, VQE, QAOA, etc.)
│   ├── qplus/              # Q+ algorithm (beyond-qubit-count factoring)
│   ├── applications/       # Domain applications (chemistry, optimization)
│   ├── backends/           # Hardware backend connectors (IBM, Google, Amazon)
│   ├── interop/            # Cross-SDK conversion (Qiskit, Cirq)
│   ├── circuits/           # Pre-built circuit templates
│   ├── transpiler/         # Circuit optimization, routing, QASM export
│   ├── error_correction/   # QEC codes (surface, steane, shor)
│   ├── mitigation/         # Error mitigation (ZNE, measurement calibration)
│   ├── fermion/            # Fermionic simulation (Jordan-Wigner, Bravyi-Kitaev)
│   ├── shadows/            # Classical shadow tomography
│   ├── dd/                 # Dynamical decoupling sequences
│   ├── resource_estimation/# Fault-tolerant resource estimation
│   ├── qml/                # Quantum machine learning
│   ├── crypto/             # Quantum cryptography (QKD)
│   ├── finance/            # Quantum finance (portfolio, options)
│   ├── qsp/                # Quantum signal processing / QSVT
│   ├── quantum_info/       # Quantum information theory
│   ├── quantum_walks/      # Quantum walks on graphs
│   ├── tomography/         # State and process tomography
│   ├── benchmarks/         # Benchmarking (RB, quantum volume)
│   ├── tensor_networks/    # MPS tensor network simulation
│   ├── optimal_control/    # GRAPE pulse optimization
│   ├── open_systems/       # Lindblad master equation
│   ├── stabilizer/         # Clifford tableau simulation
│   ├── adiabatic/          # Adiabatic quantum computing
│   └── utils/              # Visualization, entanglement measures
├── tests/                  # Test suite (1692 tests)
├── examples/               # Runnable example scripts
├── notebooks/              # Jupyter notebook tutorials
└── docs/                   # MkDocs documentation site
```

## Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -q

# Run tests for a specific module
python -m pytest tests/test_core.py -v

# Run with coverage
python -m pytest tests/ --cov=quantumium --cov-report=term-missing

# Run property-based tests only
python -m pytest tests/ -k hypothesis -v
```

All pull requests must pass the full test suite with zero failures.
New features should include corresponding tests, including property-based
tests with [Hypothesis](https://hypothesis.readthedocs.io/) where appropriate.

## Code Style

We use **ruff** for linting and formatting:

```bash
# Check for issues
ruff check src/ tests/

# Auto-fix
ruff check --fix src/ tests/

# Format
ruff format src/ tests/
```

We use **mypy** in strict mode for type checking:

```bash
mypy --strict src/quantumium/
```

### Style Guidelines

- **Type hints:** All public functions and methods must have complete type
  annotations (parameters and return types). The codebase passes `mypy --strict`.
- **Docstrings:** All public classes, functions, and methods should have
  docstrings describing what they do, their parameters, and return values.
- **Naming:** Follow PEP 8. Use `snake_case` for functions and variables,
  `PascalCase` for classes. Gate constants (H, X, CNOT) use uppercase.
- **Line length:** 100 characters maximum.
- **Imports:** Group as stdlib, third-party, local. Use `from __future__ import
  annotations` in all modules.

## Quality Gate

Before submitting a PR, verify all checks pass locally:

```bash
python -m pytest tests/ -q              # All tests pass (0 failures)
mypy --strict src/quantumium/           # No type errors
ruff check src/ tests/                  # No lint issues
ruff format --check src/ tests/         # Formatting correct
mkdocs build --strict                   # Documentation builds cleanly
```

CI runs these same checks on every push and PR.

## Making Changes

1. **Create a branch** from `main`:

   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes** -- keep commits focused and atomic.

3. **Add tests** for any new functionality.

4. **Run the quality gate** (see above).

5. **Verify examples run** if you changed public APIs:

   ```bash
   python examples/bell_state.py
   ```

6. **Submit a pull request** against `main` with a clear description of what
   changed and why.

## Adding a New Algorithm

1. Create `src/quantumium/algorithms/my_algorithm.py`.
2. Add exports to `src/quantumium/algorithms/__init__.py`.
3. Write tests in `tests/test_my_algorithm.py`.
4. Add a runnable example in `examples/`.
5. Update the API reference in `docs/`.

## Adding a New Backend

1. Subclass `Backend` from `quantumium.backends.base`.
2. Implement `run(circuit, shots)` to translate and execute circuits.
3. Add the provider's SDK to optional dependencies in `pyproject.toml`.
4. Write tests that skip gracefully when the provider SDK is absent
   (`pytest.importorskip`).

## Adding Cross-SDK Interop

1. Add a converter in `src/quantumium/interop/` (see `qiskit_convert.py` for
   the pattern).
2. Update `interop/__init__.py` with try/except lazy loading.
3. Add roundtrip tests that verify statevector fidelity.
4. Update `docs/interop.md` with the supported gate table.

## Reporting Issues

Open an issue on GitHub using the provided templates:

- **Bug reports** -- include reproduction steps, expected vs actual behavior,
  and your environment (Python version, OS, quantumium version).
- **Feature requests** -- describe the use case and any alternatives considered.

## Questions?

Open a GitHub Discussion or reach out to the maintainers.
