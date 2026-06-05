# Quantum Cryptography & Post-Quantum Security

## 1. Overview

Quantum computing impacts cryptography in two fundamental ways:
1. **Breaking classical crypto:** Shor's algorithm factors RSA and computes discrete logs in polynomial time
2. **Enabling new crypto:** Quantum key distribution (QKD) provides information-theoretic security

This creates both urgency (migrate to post-quantum algorithms) and opportunity (quantum-native security protocols).

## 2. Quantum Threats to Classical Cryptography

### Shor's Algorithm
- **Factoring:** Given N, finds factors in O((log N)³) quantum time
- **Discrete log:** Solves DLP in O((log p)³) for group order p
- **Impact:** Breaks RSA, DSA, ECDSA, ECDH, Diffie-Hellman — the foundation of internet security

### Grover's Algorithm
- **Symmetric key search:** Finds key in O(√2ⁿ) = O(2^{n/2}) operations
- **Impact on AES:** AES-128 reduced to 64-bit security, AES-256 reduced to 128-bit
- **Mitigation:** Double key lengths — AES-256 remains secure against quantum attacks

### Timeline
- **Cryptographically relevant quantum computer (CRQC):** estimated 2030-2040
- Requires ~4000 logical qubits (millions of physical qubits with error correction)
- **Harvest-now-decrypt-later:** adversaries can store encrypted traffic today and decrypt when CRQCs exist — migration is urgent

## 3. Quantum Key Distribution (QKD)

### BB84 Protocol (Bennett-Brassard, 1984)
The first and most widely implemented QKD protocol.

**Steps:**
1. **Alice:** Randomly choose bits and bases (rectilinear + or diagonal ×)
   - Encode: |0⟩, |1⟩ (+ basis) or |+⟩, |−⟩ (× basis)
2. **Bob:** Randomly measure in + or × basis
3. **Sift:** Compare bases over classical channel, keep matching bases (~50% of bits)
4. **Eavesdropper detection:** Sample subset, check error rate
   - No eavesdropper: 0% QBER (quantum bit error rate)
   - Intercept-resend attack: 25% QBER (detectable)
   - Any attack: bounded by information-disturbance trade-off
5. **Privacy amplification:** Hash remaining key to remove any partial information Eve may have

**Security proof:** Based on no-cloning theorem and uncertainty principle. Any measurement by Eve necessarily disturbs the quantum states, introducing detectable errors.

### E91 Protocol (Ekert, 1991)
Uses entanglement instead of single photons:
- Distribute Bell pairs |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
- Measure in randomly chosen bases
- Bell inequality violation confirms entanglement (no eavesdropper)
- CHSH inequality: S ≤ 2 classically, S = 2√2 quantum

### B92 Protocol
Simplified: only two non-orthogonal states needed (|0⟩ and |+⟩).

### Practical QKD
| System | Distance | Key Rate | Technology |
|--------|----------|----------|------------|
| Fiber QKD | ~100 km | ~Mbps | Single photon detectors |
| Free-space | ~1200 km | ~kbps | Satellite (Micius) |
| Twin-field | ~600 km | ~kbps | Interference-based |
| QKD networks | City-scale | Varies | Trusted relay nodes |

**Limitations:**
- Point-to-point only (no quantum internet yet for arbitrary routing)
- Requires dedicated quantum channel (fiber or free-space)
- Authentication of classical channel still requires pre-shared secret or post-quantum crypto
- Key rates decrease exponentially with distance (without quantum repeaters)

## 4. Quantum Random Number Generation (QRNG)

True randomness from quantum mechanics — not pseudorandom.

### Method
- Apply Hadamard to |0⟩: H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
- Measure: outcome is fundamentally random (Born rule)
- **Device-independent QRNG:** Bell test certifies randomness without trusting the device

### Applications
- Cryptographic key generation
- Monte Carlo simulations
- Gambling/lottery certification
- One-time pad generation (combined with QKD)

## 5. Post-Quantum Cryptography (PQC)

Classical algorithms believed resistant to quantum attacks. NIST standardized four in 2024.

### Lattice-Based
- **ML-KEM (Kyber):** Key encapsulation — NIST standard (FIPS 203)
- **ML-DSA (Dilithium):** Digital signatures — NIST standard (FIPS 204)
- **Security basis:** Learning With Errors (LWE) and Module-LWE problems
- Best known quantum algorithms offer only polynomial speedup over classical for lattice problems

### Hash-Based Signatures
- **SLH-DSA (SPHINCS+):** Stateless hash-based signatures — NIST standard (FIPS 205)
- Security relies only on hash function properties (collision resistance, preimage resistance)
- **Conservative choice:** minimal security assumptions, well-understood

### Code-Based
- **McEliece:** Key encapsulation based on error-correcting codes (1978, still unbroken)
- Large key sizes (~1 MB) limit practicality for some applications
- Under NIST evaluation for future standardization

### Isogeny-Based
- **SIKE/SIDH:** Was a NIST finalist, broken in 2022 by classical attack
- Demonstrates that "post-quantum" doesn't mean permanently secure — ongoing research needed

## 6. Hybrid Approaches

During the transition period, combine classical and post-quantum:
- **Hybrid key exchange:** X25519 + ML-KEM (both must be broken to compromise session)
- **Dual signatures:** Ed25519 + ML-DSA
- Chrome, Signal, iMessage already deploying hybrid approaches

## 7. Quantum Internet & Quantum Networks

### Quantum Repeaters
- Extend QKD range beyond fiber loss limits
- Entanglement swapping: connect short entangled links into long ones
- Quantum error correction or entanglement purification for fidelity
- **Status:** lab demonstrations, no production deployment yet

### Quantum Memory
- Store quantum states for synchronization in repeater networks
- Technologies: trapped ions (~minutes), NV centers (~seconds), atomic ensembles (~ms)
- **Requirement:** coherence time >> communication time between nodes

## 8. Relation to Quantumium SDK

Our SDK implements:
- **BB84:** Full protocol simulation with eavesdropper detection and QBER analysis
- **QRNG:** Hadamard + measurement based true random number generation
- **Shor's algorithm:** Factoring demonstration (educational, not cryptographically relevant at current qubit counts)
- See `quantumium.crypto` package
