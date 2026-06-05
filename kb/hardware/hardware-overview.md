# Quantum Hardware Technologies

## Qubit Technologies Comparison

| Technology | Coherence | Gate Fidelity | Connectivity | Scale | Speed | Operating Temp |
|-----------|-----------|---------------|-------------|-------|-------|----------------|
| Superconducting | ~100 μs | ~99.5% (2Q) | Nearest-neighbor | 100+ qubits | ~100 ns gates | ~15 mK |
| Trapped Ion | ~1-10 s | ~99.9% (2Q) | All-to-all | ~30 qubits | ~1-100 μs gates | Room temp (ions) |
| Photonic | N/A (flying) | ~99% | Programmable | ~200 modes | Speed of light | Room temp |
| Neutral Atom | ~1 s | ~99.5% (2Q) | Reconfigurable | 100+ qubits | ~1 μs gates | ~μK |
| Topological | Theoretical | Theoretical | Theoretical | 0 (R&D) | Theoretical | ~15 mK |

## 1. Superconducting Qubits

**Used by:** IBM, Google, Rigetti, Amazon (OQC)

Josephson junctions creating anharmonic oscillators. Qubits are microwave-frequency LC circuits with non-linear inductance.

### Pros
- Fast gate operations (~10-100 ns)
- Mature fabrication (semiconductor industry)
- Scaling to 1000+ qubits underway

### Cons
- Short coherence times
- Nearest-neighbor connectivity requires SWAP routing
- Dilution refrigerator required (~15 millikelvin)
- Frequency crowding at scale

## 2. Trapped Ions

**Used by:** IonQ, Quantinuum (Honeywell), AQT

Individual ions suspended in electromagnetic traps, manipulated with lasers.

### Pros
- Highest gate fidelities achieved
- Long coherence times (seconds to minutes)
- All-to-all connectivity (via shuttling or phonon modes)
- Identical qubits (atomic physics)

### Cons
- Slower gate operations
- Scaling challenges (ion chain length)
- Complex laser control systems

## 3. Photonic

**Used by:** Xanadu (PsiQuantum, ORCA)

Photons encoded in optical modes. Computation via beam splitters, phase shifters, squeezed light, and photon detection.

### Pros
- Room temperature operation
- Natural for networking/communication
- Gaussian boson sampling advantage demonstrated

### Cons
- Photon loss is primary error source
- Probabilistic gates (requires multiplexing)
- Difficult to create deterministic photon-photon interactions

## 4. Neutral Atoms

**Used by:** QuEra, Pasqal, Atom Computing

Individual atoms trapped in optical tweezer arrays, interacting via Rydberg excitation.

### Pros
- Reconfigurable connectivity (move atoms)
- Large qubit counts achievable
- Long coherence times
- Native multi-qubit gates via Rydberg blockade

### Cons
- Relatively new technology
- Gate fidelities still improving
- Atom loss during computation

## 5. Topological (Majorana)

**Used by:** Microsoft (Station Q)

Qubits encoded in non-abelian anyons (Majorana zero modes). Information stored non-locally, inherently protected from local noise.

### Status
- Still in R&D — Microsoft announced first topological qubit in 2025
- If realized: dramatically lower error rates, built-in error protection
- Key advantage: topological protection = fewer physical qubits per logical qubit

## Error Correction Landscape

Current NISQ devices have error rates of ~10⁻³ per gate. Fault-tolerant quantum computing requires ~10⁻⁶ or better.

**Surface Code:** Leading error correction scheme
- Threshold: ~1% physical error rate
- Overhead: ~1000 physical qubits per logical qubit (current estimates)
- IBM, Google, and others actively pursuing

**Timeline to Fault Tolerance:** Most estimates: 2028-2035 for practical fault-tolerant systems
