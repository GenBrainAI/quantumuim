# Contributing

Thanks for your interest in contributing to Quantumium! This guide covers the development workflow.

## Setup

```bash
git clone https://github.com/GenBrainAI/quantumuim.git
cd quantumuim
pip install -e ".[dev]"
```

## Development Workflow

1. Create a feature branch from `main`
2. Write code and tests
3. Run the test suite: `python -m pytest tests/ -q`
4. Run linting: `ruff check src/`
5. Run type checking: `mypy src/quantumium/`
6. Submit a pull request

## Code Style

- **Formatter**: [Ruff](https://docs.astral.sh/ruff/) with 100-char line length
- **Type hints**: Required for all public APIs
- **Docstrings**: Google style, required for all public functions and classes
- **Tests**: Every new feature must include tests

## Project Structure

```
src/quantumium/
    core/           # Circuit, gates, noise, parameters
    algorithms/     # Quantum algorithms (Grover, VQE, etc.)
    backends/       # Hardware backends (IBM, Google, Amazon)
    applications/   # Chemistry, optimization
    qml/            # Quantum machine learning
    crypto/         # BB84, QRNG
    finance/        # Option pricing, risk analysis
    transpiler/     # Circuit optimization, OpenQASM
    ...
tests/              # Test suite (1095+ tests)
examples/           # Runnable example scripts (33)
docs/               # MkDocs documentation
kb/                 # Knowledge base articles (21)
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -q

# Run specific module tests
python -m pytest tests/test_algorithms.py -v

# Run with coverage
python -m pytest tests/ --cov=quantumium --cov-report=term-missing
```

## Adding a New Module

1. Create `src/quantumium/<module>/` with `__init__.py`
2. Add tests in `tests/test_<module>.py`
3. Add an example in `examples/<module>_demo.py`
4. Update `README.md` and `ROADMAP.md`
5. Add API docs page in `docs/api/`

## Dependencies

Quantumium's core has a single dependency: `numpy`. Keep it that way.

- No external quantum SDKs in core code
- Hardware backends are optional (`pip install quantumium[qiskit]`)
- Dev tools (pytest, ruff, mypy, mkdocs) are dev-only dependencies

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Include tests for all new functionality
- Update documentation if the public API changes
- All CI checks must pass before merge

## Reporting Issues

Use [GitHub Issues](https://github.com/GenBrainAI/quantumuim/issues) with the appropriate template:

- **Bug Report** — for unexpected behavior
- **Feature Request** — for new functionality suggestions
