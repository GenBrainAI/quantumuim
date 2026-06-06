# Hardware Validation — Practical Guide

Validating a quantum SDK against real hardware ensures that circuit translation, execution, and result parsing work correctly end-to-end. This guide covers methodology for testing Quantumium against IBM Quantum, Google Quantum AI, and Amazon Braket.

## Why Hardware Validation

1. **Translation correctness** — verify gate decomposition matches hardware native gates
2. **Result fidelity** — confirm measurement distributions match simulator predictions within noise bounds
3. **API compatibility** — ensure backend SDKs (Qiskit, Cirq, Braket) haven't introduced breaking changes
4. **Performance baseline** — establish wall-clock and queue-time benchmarks for each provider

## Validation Strategy

### Tier 1: Simulator Cross-Check

Run identical circuits on Quantumium's local simulator and each vendor's simulator. Results should be statistically identical (within sampling variance).

```
Quantumium LocalSimulator  ──┐
                              ├── Compare distributions (χ² test, p > 0.01)
Vendor Simulator (Aer, etc.) ─┘
```

**Test circuits:**
- Bell state (2 qubits) — expect 50/50 split on |00⟩ and |11⟩
- GHZ state (3-5 qubits) — expect equal superposition of |00...0⟩ and |11...1⟩
- Grover's search (3 qubits) — expect marked state with ~78% probability
- QFT + inverse QFT — expect identity (all |0⟩)

### Tier 2: Hardware Execution

Run the same circuits on real hardware. Results will differ from ideal due to noise, but should show correct structure.

**Acceptance criteria:**
- Bell state: |00⟩ + |11⟩ together > 85% (typical for current hardware)
- GHZ(3): |000⟩ + |111⟩ together > 70%
- Grover(3): marked state is the most frequent outcome
- Measured fidelity > classical random baseline

### Tier 3: Noise Model Calibration

Compare Quantumium's depolarizing noise model predictions against actual hardware noise:

1. Run randomized benchmarking on hardware to extract per-gate error rates
2. Configure Quantumium's NoiseModel with measured error rates
3. Run test circuits with noise model and compare to hardware results
4. Fidelity between noise-model prediction and hardware should exceed 0.9

## Provider-Specific Notes

### IBM Quantum (via Qiskit)

**Access:** Free tier available (ibm_brisbane, ibm_kyoto, etc.)

**Native gate set:** ECR (or CX), ID, RZ, SX, X

**Key considerations:**
- Transpilation: Quantumium circuits must decompose to native gates
- Coupling map: not all qubit pairs are connected; SWAP routing required
- Dynamic circuits: mid-circuit measurement supported on some devices
- Queue times: free tier can have 1-4 hour waits; use simulators for rapid iteration
- Qiskit Runtime: prefer `SamplerV2` over deprecated `execute()`

**Validation script pattern:**
```python
from quantumium.backends.qiskit_backend import QiskitBackend
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT

qc = QuantumCircuit(2)
qc.apply(H, 0)
qc.apply(CNOT, 0, 1)

backend = QiskitBackend(device="ibm_brisbane")
result = backend.run(qc, shots=4096)

# Verify Bell state structure
p00 = result.probabilities.get("00", 0)
p11 = result.probabilities.get("11", 0)
assert p00 + p11 > 0.85, f"Bell fidelity too low: {p00 + p11:.3f}"
```

### Google Quantum AI (via Cirq)

**Access:** Requires approved Google Cloud project

**Native gate set:** √iSWAP, Sycamore, PhasedXZ, virtual Z

**Key considerations:**
- Google uses √iSWAP as the native two-qubit gate (not CNOT)
- Device topology is a 2D grid with specific qubit connectivity
- Calibration data available via `cirq_google.get_engine_calibration()`
- XEB (cross-entropy benchmarking) is the standard fidelity metric

### Amazon Braket (IonQ, Rigetti, OQC, QuEra)

**Access:** Pay-per-shot, no free tier (but SV1/DM1/TN1 simulators available)

**IonQ (trapped ions):**
- Native gates: GPI, GPI2, MS (Molmer-Sorensen)
- All-to-all connectivity — no SWAP routing needed
- Higher fidelity but slower gate times

**Rigetti (superconducting):**
- Native gates: CZ, RX, RZ
- Parametric compilation for variational circuits
- Quilc compiler handles decomposition

**OQC (superconducting):**
- ECR native two-qubit gate
- Limited qubit count but competitive fidelity

**QuEra (neutral atoms):**
- Analog mode: Hamiltonian evolution on atom arrays
- Digital mode: emerging, limited gate set
- Best for optimization and simulation workloads

## Statistical Validation Methods

### Chi-Squared Test

Compare two measurement distributions:

```
χ² = Σ (O_i - E_i)² / E_i

where O_i = observed counts, E_i = expected counts
```

For simulator cross-check: p-value > 0.01 indicates agreement.

### Hellinger Fidelity

```
F_H = (Σ √(p_i · q_i))²

where p_i, q_i are probability distributions
```

- F_H = 1: identical distributions
- F_H > 0.9: good agreement for noisy hardware
- F_H < 0.5: likely translation or execution error

### Total Variation Distance

```
TVD = (1/2) Σ |p_i - q_i|
```

- TVD < 0.1: excellent agreement
- TVD < 0.2: acceptable for NISQ hardware

## Test Matrix

| Circuit | Qubits | Ideal Result | IBM Pass | IonQ Pass | Rigetti Pass |
|---------|--------|-------------|----------|-----------|-------------|
| Bell | 2 | 50/50 |00⟩,|11⟩ | F_H > 0.9 | F_H > 0.95 | F_H > 0.85 |
| GHZ(3) | 3 | 50/50 |000⟩,|111⟩ | F_H > 0.85 | F_H > 0.9 | F_H > 0.8 |
| Grover(3) | 3 | ~78% marked | top-1 correct | top-1 correct | top-1 correct |
| VQE(H2) | 4 | E₀ ± 0.1 Ha | within 0.2 Ha | within 0.15 Ha | within 0.2 Ha |
| RB(1Q) | 1 | exp decay | r < 0.01 | r < 0.005 | r < 0.01 |

## Integration Test Structure

```
tests/
  test_backends/
    test_local_simulator.py      # Always runs
    test_qiskit_backend.py       # Requires qiskit, uses Aer by default
    test_cirq_backend.py         # Requires cirq
    test_braket_backend.py       # Requires amazon-braket-sdk
    conftest.py                  # Fixtures: skip if SDK not installed
    
    # Hardware tests (manual, require credentials)
    test_ibm_hardware.py         # @pytest.mark.hardware
    test_ionq_hardware.py        # @pytest.mark.hardware
    test_rigetti_hardware.py     # @pytest.mark.hardware
```

**conftest.py pattern:**
```python
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "hardware: requires real quantum hardware")

@pytest.fixture
def skip_no_qiskit():
    pytest.importorskip("qiskit")

@pytest.fixture  
def skip_no_braket():
    pytest.importorskip("braket")
```

## Noise Model Calibration Workflow

1. **Extract hardware parameters:**
   - T1, T2 coherence times per qubit
   - Single-qubit gate error rate (from RB)
   - Two-qubit gate error rate (from interleaved RB or XEB)
   - Readout error rates per qubit

2. **Build Quantumium NoiseModel:**
   ```python
   from quantumium.core.noise import NoiseModel, depolarizing, amplitude_damping
   
   noise = NoiseModel()
   noise.add_single_qubit_noise(depolarizing(p=0.001))      # from RB
   noise.add_two_qubit_noise(depolarizing(p=0.01))           # from 2Q RB
   noise.add_readout_noise(p0_given1=0.02, p1_given0=0.01)   # from calibration
   ```

3. **Compare predictions vs hardware:**
   - Run 5+ circuits with noise model and on hardware
   - Compute Hellinger fidelity between noise-model and hardware distributions
   - Target: F_H > 0.85 across all test circuits

## Performance Benchmarks

Track these metrics for each backend:

| Metric | What It Measures | How to Collect |
|--------|-----------------|----------------|
| Translation time | Circuit → vendor format | `time.time()` around conversion |
| Queue time | Submit → start execution | Backend metadata timestamps |
| Execution time | Actual quantum computation | Backend metadata |
| Total wall-clock | End-to-end from `run()` call | `time.time()` around full call |
| Shots/second | Throughput | shots / execution_time |

## Quantumium Hardware Validation API

```python
from quantumium.validation import (
    HardwareValidator,
    ValidationSuite,
    cross_check_simulators,
    calibrate_noise_model,
)

# Run full validation suite
validator = HardwareValidator(backend="ibm_brisbane")
report = validator.run_suite(
    circuits=["bell", "ghz3", "grover3", "qft4"],
    shots=4096,
    compare_to="local",
)
print(report.summary())
# Backend: ibm_brisbane
# Circuits tested: 4
# Average Hellinger fidelity: 0.91
# All pass: True

# Calibrate noise model from hardware data
noise_model = calibrate_noise_model(
    backend="ibm_brisbane",
    rb_depths=[1, 5, 10, 20, 50],
    shots=1024,
)
```

## Further Reading

- IBM Quantum documentation: hardware specs, error rates, native gates
- Google Quantum AI: Cirq device specifications, XEB methodology
- Amazon Braket: device properties API, pricing model
- Quantumium backends source: `src/quantumium/backends/`
