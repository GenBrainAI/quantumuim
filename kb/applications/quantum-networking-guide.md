# Quantum Networking — Practical Guide

Quantum networking extends quantum information processing beyond single devices, enabling distributed quantum computing, secure communication, and entanglement distribution across physical distances.

## Why Quantum Networking Matters

1. **Quantum key distribution (QKD)** — information-theoretically secure communication
2. **Distributed quantum computing** — connect small quantum processors into larger virtual machines
3. **Quantum sensing networks** — synchronized quantum sensors for precision measurement
4. **Blind quantum computing** — delegate computation to a remote server without revealing the algorithm

## Core Concepts

### Entanglement Distribution

The fundamental primitive: create and share entangled Bell pairs between distant nodes.

```
Node A                     Node B
  |                           |
  |--- EPR source ---|        |
  |   (generates     |        |
  |    Bell pairs)    |------->|
  |                           |
  |  |Φ+⟩ = (|00⟩+|11⟩)/√2  |
```

Challenges:
- **Photon loss** — optical fiber attenuation ~0.2 dB/km at 1550nm; after ~100 km, probability of successful transmission drops below useful thresholds
- **No cloning** — cannot amplify quantum signals like classical repeaters
- **Decoherence** — entangled states degrade during storage and transmission

### Quantum Repeaters

Extend entanglement range by breaking the channel into shorter segments and using entanglement swapping.

#### Entanglement Swapping

```
Node A ---[Bell pair 1]--- Node R ---[Bell pair 2]--- Node B
       (A entangled with R)      (R entangled with B)

Node R performs Bell measurement on its two qubits.
Result: A becomes entangled with B (without ever interacting directly).
```

Protocol:
1. Create Bell pairs: A-R₁ and R₁-B (or R₁-R₂, R₂-B for longer chains)
2. Repeater R₁ performs Bell state measurement on its two qubits
3. Communicate classical measurement result to B
4. B applies correction based on result
5. A and B now share a Bell pair

#### Repeater Chain

For distance L with n repeater nodes:
- Each segment has length L/(n+1)
- Success probability per segment: p_seg ≈ exp(-L/(n+1)/L_att)
- Total time scales polynomially with n (vs exponentially without repeaters)

### Entanglement Purification

Real entangled pairs have imperfect fidelity. Purification distills high-fidelity pairs from multiple low-fidelity ones.

#### BBPSSW Protocol (Bennett et al., 1996)

Starting with two imperfect Bell pairs with fidelity F:
1. Both parties apply CNOT with one pair as control, other as target
2. Both measure the target pair
3. If measurements agree, keep the control pair (now with higher fidelity)
4. If measurements disagree, discard both

Fidelity improvement per round:
```
F' = F² + (1-F)²/9
     ─────────────────────────
     F² + 2F(1-F)/3 + 5(1-F)²/9
```

For F > 0.5, purification improves fidelity. Multiple rounds approach F → 1.

### Quantum Teleportation Over Noisy Channels

Standard teleportation requires a pre-shared Bell pair:
1. Sender performs Bell measurement on (qubit-to-send, half-of-Bell-pair)
2. Sender communicates 2 classical bits to receiver
3. Receiver applies correction (I, X, Z, or XZ) based on classical bits
4. Receiver now has the original qubit state

Over noisy channels:
- The shared Bell pair has fidelity F < 1
- Teleportation fidelity = (2F + 1)/3 for depolarizing noise
- Must exceed classical limit of 2/3 → requires F > 1/2

## Network Architectures

### Star Topology

Central node connected to all others. Simple but single point of failure.
```
    B
    |
A---Hub---C
    |
    D
```

### Chain (Linear) Topology

Repeater chain for point-to-point links. Minimal infrastructure.
```
A---R1---R2---R3---B
```

### Mesh Topology

Multiple paths between nodes. Fault-tolerant, complex routing.
```
A---R1---B
|   |    |
R2--R3--R4
|   |    |
C---R5---D
```

### Quantum Internet Protocol Stack

| Layer | Classical Analog | Quantum Function |
|-------|-----------------|------------------|
| Application | HTTP/SMTP | QKD, distributed computing, sensing |
| Transport | TCP | End-to-end entanglement management |
| Network | IP | Entanglement routing, path selection |
| Link | Ethernet | Point-to-point entanglement generation |
| Physical | Fiber/radio | Photon transmission, quantum memory |

## Key Metrics

### Entanglement Generation Rate

Bell pairs per second between two nodes:
- Depends on: channel loss, detector efficiency, memory coherence time
- Current state of the art: ~10-100 Hz over short distances (lab)
- Target for useful QKD: ~1 kHz

### Fidelity

How close the distributed state is to a perfect Bell pair:
- F = ⟨Φ+|ρ|Φ+⟩
- Threshold for QKD: F > 1/2
- Threshold for fault-tolerant distributed computing: F > ~0.99

### Secret Key Rate (for QKD)

Secure bits per channel use:
- BB84: r = 1 - h(e_bit) - h(e_phase), where h is binary entropy
- E91: based on CHSH inequality violation
- Device-independent QKD: security without trusting devices

## Simulation Considerations

### What to Model

1. **Channel loss**: exponential decay with distance, parameterized by attenuation length
2. **Depolarizing noise**: on quantum memories and during transmission
3. **Measurement errors**: imperfect Bell state analyzers
4. **Classical communication delay**: speed of light latency for measurement results
5. **Memory decoherence**: T1/T2 decay in quantum memories while waiting

### Simplifications for SDK

- Discrete time steps (not continuous)
- Perfect classical communication (no bit errors)
- Depolarizing noise model (captures essential physics without full master equation)
- Fixed topology (no dynamic routing)

## Quantum Networking in Quantumium

```python
from quantumium.networking import (
    QuantumNetwork,
    QuantumRepeater,
    entanglement_swap,
    purify_bbpssw,
    teleport,
)

# Build a 3-node repeater chain
network = QuantumNetwork()
network.add_node("Alice")
network.add_node("Repeater", is_repeater=True)
network.add_node("Bob")
network.add_link("Alice", "Repeater", distance_km=50, loss_db_per_km=0.2)
network.add_link("Repeater", "Bob", distance_km=50, loss_db_per_km=0.2)

# Generate entanglement
pair_ar = network.generate_entanglement("Alice", "Repeater")
pair_rb = network.generate_entanglement("Repeater", "Bob")

# Swap at repeater
pair_ab = entanglement_swap(pair_ar, pair_rb)
print(f"Alice-Bob fidelity: {pair_ab.fidelity:.4f}")

# Purify
purified = purify_bbpssw(pair_ab, pair_ab_copy)
print(f"Purified fidelity: {purified.fidelity:.4f}")

# Teleport a qubit
state = [0.6, 0.8]  # |ψ⟩ = 0.6|0⟩ + 0.8|1⟩
received = teleport(state, pair_ab)
```

## Further Reading

- Kimble, "The quantum internet" (Nature, 2008)
- Wehner, Elkouss, Hanson, "Quantum internet: A vision for the road ahead" (Science, 2018)
- Briegel et al., "Quantum Repeaters: The Role of Imperfect Local Operations" (PRL, 1998)
- Bennett et al., "Purification of Noisy Entanglement and Faithful Teleportation" (PRL, 1996)
