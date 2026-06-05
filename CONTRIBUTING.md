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
   mypy).

## Project Structure

```
quantumuim/
├── src/quantumium/         # Source code
│   ├── core/               # Qubits, gates, circuits, noise, parameters
│   ├── algorithms/         # Quantum algorithms (Grover, VQE, QAOA, etc.)
│   ├── applications/       # Domain applications (chemistry, optimization)
│   ├── backends/           # Hardware backend connectors
│   ├── circuits/           # Pre-built circuit templates
│   ├── transpiler/         # Circuit optimization and QASM export
│   ├── error_correction/   # QEC codes
│   ├── mitigation/         # Error mitigation techniques
│   ├── qml/                # Quantum machine learning
│   ├── crypto/             # Quantum cryptography
│   ├── finance/            # Quantum finance
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
├── tests/                  # Test suite (877+ tests)
├── examples/               # Runnable example scripts
├── kb/                     # Quantum computing knowledge base
└── docs/                   # Documentation
```

## Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -q

# Run tests for a specific module
python -m pytest tests/test_core.py -v

# Run with coverage
python -m pytest tests/ --cov=quantumium --cov-report=term-missing
```

All pull requests must pass the full test suite. New features should include
corresponding tests.

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

We use **mypy** for type checking:

```bash
mypy src/quantumium/
```

### Style Guidelines

- **Type hints:** All public functions and methods must have type annotations.
- **Docstrings:** All public classes, functions, and methods should have
  docstrings describing what they do, their parameters, and return values.
- **Naming:** Follow PEP 8. Use `snake_case` for functions and variables,
  `PascalCase` for classes. Gate constants (H, X, CNOT) use uppercase.
- **Line length:** 100 characters maximum.
- **Imports:** Group as stdlib, third-party, local. Use `from __future__ import
  annotations` in all modules.

## Making Changes

1. **Create a branch** from `main`:

   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes** — keep commits focused and atomic.

3. **Add tests** for any new functionality.

4. **Verify your changes:**

   ```bash
   python -m pytest tests/ -q      # All tests pass
   ruff check src/ tests/          # No lint issues
   mypy src/quantumium/            # No type errors
   ```

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
5. Update the algorithm table in `README.md`.

## Adding a New Backend

1. Subclass `Backend` from `quantumium.backends.base`.
2. Implement `run(circuit, shots)` to translate and execute circuits.
3. Add the provider's SDK to optional dependencies in `pyproject.toml`.
4. Write tests that work without the provider SDK installed (mock the external
   calls).

## Adding a New Module

1. Create `src/quantumium/my_module/` with `__init__.py`.
2. Write tests in `tests/test_my_module.py`.
3. Add an example in `examples/`.
4. Update the module list in `README.md`.

## Reporting Issues

Open an issue on GitHub with:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

## Questions?

Open a GitHub Discussion or reach out to the maintainers.
