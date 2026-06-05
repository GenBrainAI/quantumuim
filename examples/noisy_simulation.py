"""Noisy circuit simulation demo.

Shows how noise degrades quantum circuits using density matrix simulation.
Compares ideal vs noisy Bell state preparation and measures fidelity.
"""

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT, X
from quantumium.core.noise import NoiseModel, depolarizing, amplitude_damping, bit_flip


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute fidelity between two density matrices."""
    sqrt_rho = np.linalg.matrix_power(
        rho.astype(complex), 1
    )
    eigvals = np.linalg.eigvalsh(rho)
    sqrt_rho = sum(
        np.sqrt(max(v, 0)) * np.outer(vec, vec.conj())
        for v, vec in zip(eigvals, np.linalg.eigh(rho)[1].T)
    )
    product = sqrt_rho @ sigma @ sqrt_rho
    return float(np.real(np.trace(np.linalg.matrix_power(
        product.astype(complex), 1
    )))) ** 0.5 if np.trace(product).real > 0 else 0.0


def simple_fidelity(rho: np.ndarray, rho_ideal: np.ndarray) -> float:
    """State fidelity: Tr(rho_ideal @ rho) when rho_ideal is pure."""
    return float(np.real(np.trace(rho_ideal @ rho)))


def main():
    print("=" * 60)
    print("  Noisy Quantum Circuit Simulation")
    print("=" * 60)

    # --- 1. Bell state: ideal vs noisy ---
    print("\n1. Bell State Preparation")
    print("-" * 40)

    qc = QuantumCircuit(2)
    qc.apply(H, 0)
    qc.apply(CNOT, 0, 1)

    rho_ideal = qc.density_matrix()
    print(f"   Ideal purity:  {np.real(np.trace(rho_ideal @ rho_ideal)):.6f}")

    for p in [0.01, 0.05, 0.1, 0.2]:
        noise = NoiseModel(
            default_single_qubit_noise=depolarizing(p),
            default_two_qubit_noise=depolarizing(p * 2),
        )
        rho_noisy = qc.density_matrix(noise_model=noise)
        purity = float(np.real(np.trace(rho_noisy @ rho_noisy)))
        fid = simple_fidelity(rho_noisy, rho_ideal)
        print(f"   p={p:.2f}  purity={purity:.4f}  fidelity={fid:.4f}")

    # --- 2. Amplitude damping on excited state ---
    print("\n2. Amplitude Damping (T1 Decay)")
    print("-" * 40)

    for gamma in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
        noise = NoiseModel(default_single_qubit_noise=amplitude_damping(gamma))
        qc_x = QuantumCircuit(1)
        qc_x.apply(X, 0)
        rho = qc_x.density_matrix(noise_model=noise)
        p0 = rho[0, 0].real
        p1 = rho[1, 1].real
        print(f"   gamma={gamma:.1f}  P(|0⟩)={p0:.4f}  P(|1⟩)={p1:.4f}")

    # --- 3. Noisy measurement sampling ---
    print("\n3. Noisy Measurement (Bell state, 10000 shots)")
    print("-" * 40)

    qc_bell = QuantumCircuit(2)
    qc_bell.apply(H, 0)
    qc_bell.apply(CNOT, 0, 1)

    ideal_counts = qc_bell.measure_all(shots=10000)
    print(f"   Ideal:  {dict(sorted(ideal_counts.items()))}")

    noise = NoiseModel(
        default_single_qubit_noise=depolarizing(0.05),
        default_two_qubit_noise=depolarizing(0.10),
    )
    noisy_counts = qc_bell.noisy_measure(noise, shots=10000)
    print(f"   Noisy:  {dict(sorted(noisy_counts.items()))}")

    correlated = sum(noisy_counts.get(k, 0) for k in ["00", "11"])
    total = sum(noisy_counts.values())
    print(f"   Correlated (00+11): {correlated/total*100:.1f}%  (ideal: ~100%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
