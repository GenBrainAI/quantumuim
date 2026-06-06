# Dynamical Decoupling Guide

## Overview

Dynamical decoupling (DD) is a noise suppression technique that applies sequences of fast control pulses to refocus unwanted qubit-environment interactions. Unlike error correction (which detects and corrects errors after they occur), DD proactively prevents decoherence by averaging out noise during idle periods.

DD is critical for NISQ hardware where qubits sit idle between gates — these idle windows are where most T2 decoherence occurs.

## Physical Principle

A qubit coupled to an environment evolves under H = H_system + H_env + H_interaction. The interaction term causes decoherence. By applying π-pulses (X or Y gates) at carefully timed intervals, the interaction Hamiltonian is periodically inverted, causing its time-average to approach zero.

Key insight: if the noise is slow compared to the pulse spacing (the "bang-bang" limit), the qubit effectively decouples from the environment.

## DD Sequences

### Free Induction Decay (FID)

No pulses — baseline for comparison. Qubit decoheres at rate 1/T2*.

### Spin Echo (Hahn Echo)

Single π-pulse at the midpoint of an idle period:

```
|idle| — X — |idle|
  τ/2         τ/2
```

Refocuses static (DC) noise. Extends coherence from T2* to T2.

### CPMG (Carr-Purcell-Meiboom-Gill)

N equally spaced π_Y pulses:

```
|idle| — Y — |idle| — Y — |idle| — ... — Y — |idle|
  τ/2N        τ/N          τ/N              τ/2N
```

Better than Hahn echo for N > 1 — suppresses noise at higher frequencies. The filter function peaks at f = N/(2τ), making CPMG a narrow-band noise filter.

### XY-4

Alternating X and Y pulses to handle both σ_x and σ_y noise:

```
|idle| — X — |idle| — Y — |idle| — X — |idle| — Y — |idle|
  τ/8         τ/4          τ/4          τ/4          τ/8
```

Symmetric, self-correcting for pulse errors. Suppresses both dephasing and amplitude noise.

### XY-8

Two repetitions of XY-4 for stronger suppression:

```
X — Y — X — Y — Y — X — Y — X
```

The second half is the time-reverse of the first, providing additional robustness against systematic pulse errors.

### Uhrig Dynamical Decoupling (UDD)

Non-uniform pulse spacing optimized for specific noise spectra:

```
t_j = τ * sin²(πj / (2N+2))    for j = 1, ..., N
```

Optimal for noise with a hard high-frequency cutoff (Ohmic bath). Suppresses decoherence to O(τ^(N+1)) for N pulses.

### Concatenated DD (CDD)

Recursive nesting of DD sequences for universal decoupling:

```
CDD_1 = X — idle — X — idle
CDD_2 = CDD_1_Z — Z — CDD_1_Z — Z    (replace idle with Z-CDD_1)
```

Achieves arbitrary-order decoupling but requires exponentially many pulses.

## Implementation Architecture

### DDSequence

Core data structure representing a pulse sequence:

```python
class DDSequence:
    name: str                    # "xy4", "cpmg", "udd", etc.
    pulses: list[str]           # gate names: ["X", "Y", "X", "Y"]
    spacings: list[float]       # normalized time fractions between pulses
    
    @staticmethod
    def xy4() -> DDSequence
    
    @staticmethod
    def xy8() -> DDSequence
    
    @staticmethod
    def cpmg(n_pulses: int) -> DDSequence
    
    @staticmethod
    def udd(n_pulses: int) -> DDSequence
    
    @staticmethod
    def spin_echo() -> DDSequence
```

### DDScheduler

Inserts DD sequences into circuit idle windows:

```python
class DDScheduler:
    def __init__(self, sequence: DDSequence, min_idle_duration: int = 2):
        ...
    
    def schedule(self, circuit: QuantumCircuit, 
                 durations: dict[str, int] = None) -> QuantumCircuit:
        """Insert DD pulses into idle windows.
        
        Args:
            circuit: input circuit
            durations: gate durations in dt units (default: 1 for all)
        
        Returns:
            New circuit with DD pulses inserted
        """
```

Algorithm:
1. Compute scheduling: when each gate starts/finishes per qubit
2. Identify idle windows (gaps between gates on each qubit)
3. For each idle window >= min_idle_duration: insert DD sequence
4. Adjust timing to fill the window evenly

### NoiseFilter

Analytical noise filter function for DD sequences:

```python
def filter_function(sequence: DDSequence, frequencies: ndarray, 
                    total_time: float) -> ndarray:
    """Compute the noise filter function |F(ω)|² for a DD sequence.
    
    The filter function determines which noise frequencies are suppressed.
    Lower values = better suppression at that frequency.
    """
```

For CPMG with N pulses: peaks at f = (2k-1)N/(2τ) for k = 1, 2, ...

### Simulation Integration

```python
def dd_fidelity(circuit: QuantumCircuit, sequence: DDSequence,
                noise_spectrum: callable, total_time: float) -> float:
    """Estimate fidelity after DD sequence under given noise spectrum.
    
    Uses the overlap integral:
    χ(τ) = exp(-1/π ∫ S(ω)|F(ω)|²/ω² dω)
    """
```

## Gate Duration Model

For realistic DD scheduling, we need gate durations:

```python
DEFAULT_DURATIONS = {
    "I": 1,     # identity (idle)
    "X": 1,     # single-qubit gates
    "Y": 1,
    "Z": 0,     # virtual Z (zero duration)
    "H": 1,
    "S": 1,
    "T": 1,
    "CNOT": 2,  # two-qubit gates take longer
    "CZ": 2,
    "SWAP": 6,  # 3 CNOTs
}
```

## Testing Strategy

1. **Sequence construction**: verify pulse counts, spacing normalization (sum = 1.0)
2. **UDD timing**: verify sin² formula matches known values
3. **Filter function**: verify CPMG peaks at expected frequencies
4. **Schedule correctness**: DD pulses inserted only in idle windows, don't overlap gates
5. **Identity preservation**: DD sequence on |0⟩ with no noise returns |0⟩ (X² = I, Y² = I)
6. **Noise suppression**: with simulated dephasing noise, DD fidelity > FID fidelity
7. **Gate count**: verify correct number of pulses inserted per idle window
8. **Edge cases**: no idle windows, single-qubit circuit, circuit with measurements

## Noise Suppression Hierarchy

Under typical 1/f noise (common in superconducting qubits):

FID < Spin Echo < CPMG-2 < XY-4 < CPMG-4 < XY-8

For Ohmic noise with hard cutoff:

FID < Spin Echo < CPMG < UDD (optimal)

## References

- Viola, Knill, Lloyd (1999): dynamical decoupling framework
- Carr, Purcell (1954): CPMG pulse sequence
- Uhrig (2007): optimal pulse timing for specific noise spectra
- Khodjasteh, Viola (2005): concatenated dynamical decoupling
- Suter, Alvarez (2016): review of DD in quantum information
