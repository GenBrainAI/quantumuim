"""Lindblad master equation solver.

Integrates the Lindblad equation:
    dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k† L_k, ρ})

Uses fourth-order Runge-Kutta for time integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .dynamics import EvolutionResult, expect


@dataclass
class LindbladSystem:
    """Specification of an open quantum system.

    Attributes:
        hamiltonian: System Hamiltonian H (d × d), or callable H(t).
        collapse_ops: List of (rate, operator) pairs for Lindblad terms.
        dim: Hilbert space dimension.
    """

    hamiltonian: NDArray[np.complexfloating] | Callable[[float], NDArray[np.complexfloating]]
    collapse_ops: list[tuple[float, NDArray[np.complexfloating]]] = field(default_factory=list)

    @property
    def dim(self) -> int:
        """Return the Hilbert space dimension of the system.

        Returns:
            int: Dimension d of the system Hamiltonian (d x d).
        """
        H = self._get_hamiltonian(0.0)
        return H.shape[0]

    def _get_hamiltonian(self, t: float) -> NDArray[np.complexfloating]:
        if callable(self.hamiltonian):
            return self.hamiltonian(t)
        return self.hamiltonian


def lindblad_rhs(
    rho: NDArray[np.complexfloating],
    system: LindbladSystem,
    t: float = 0.0,
) -> NDArray[np.complexfloating]:
    """Compute dρ/dt from the Lindblad master equation.

    Args:
        rho: Current density matrix.
        system: Lindblad system specification.
        t: Current time (for time-dependent Hamiltonians).

    Returns:
        Time derivative dρ/dt.
    """
    H = system._get_hamiltonian(t)

    # Coherent part: -i[H, ρ]
    drho = -1j * (H @ rho - rho @ H)

    # Dissipative part: Σ γ_k (L ρ L† - ½{L†L, ρ})
    for gamma, L in system.collapse_ops:
        if gamma == 0:
            continue
        Ldag = L.conj().T
        LdagL = Ldag @ L
        drho += gamma * (L @ rho @ Ldag - 0.5 * (LdagL @ rho + rho @ LdagL))

    return drho


def lindblad_evolve(
    rho0: NDArray[np.complexfloating],
    system: LindbladSystem,
    dt: float,
    num_steps: int,
) -> NDArray[np.complexfloating]:
    """Evolve a density matrix for a fixed number of steps.

    Uses 4th-order Runge-Kutta integration.

    Args:
        rho0: Initial density matrix.
        system: Lindblad system specification.
        dt: Time step.
        num_steps: Number of integration steps.

    Returns:
        Final density matrix after evolution.
    """
    rho = rho0.astype(complex).copy()
    t = 0.0

    for _ in range(num_steps):
        k1 = lindblad_rhs(rho, system, t)
        k2 = lindblad_rhs(rho + 0.5 * dt * k1, system, t + 0.5 * dt)
        k3 = lindblad_rhs(rho + 0.5 * dt * k2, system, t + 0.5 * dt)
        k4 = lindblad_rhs(rho + dt * k3, system, t + dt)
        rho = rho + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt

    return rho


def mesolve(
    system: LindbladSystem,
    rho0: NDArray[np.complexfloating],
    tlist: NDArray[np.floating],
    observables: dict[str, NDArray[np.complexfloating]] | None = None,
) -> EvolutionResult:
    """Solve the Lindblad master equation over a time grid.

    Uses adaptive-step 4th-order Runge-Kutta between output times.

    Args:
        system: Lindblad system specification.
        rho0: Initial density matrix.
        tlist: Array of output time points (must be sorted).
        observables: Optional dict of {label: operator} for tracking
            expectation values.

    Returns:
        EvolutionResult with states and expectation values.
    """
    tlist = np.asarray(tlist, dtype=float)
    rho = rho0.astype(complex).copy()

    states = [rho.copy()]
    expect_values = {
        label: [float(np.real(expect(rho, op)))] for label, op in (observables or {}).items()
    }

    for i in range(1, len(tlist)):
        dt_interval = tlist[i] - tlist[i - 1]
        num_substeps = max(1, int(np.ceil(dt_interval / 0.01)))
        dt = dt_interval / num_substeps
        t = tlist[i - 1]

        for _ in range(num_substeps):
            k1 = lindblad_rhs(rho, system, t)
            k2 = lindblad_rhs(rho + 0.5 * dt * k1, system, t + 0.5 * dt)
            k3 = lindblad_rhs(rho + 0.5 * dt * k2, system, t + 0.5 * dt)
            k4 = lindblad_rhs(rho + dt * k3, system, t + dt)
            rho = rho + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt

        states.append(rho.copy())
        for label, op in (observables or {}).items():
            expect_values[label].append(float(np.real(expect(rho, op))))

    result_expect = {label: np.array(vals) for label, vals in expect_values.items()}
    return EvolutionResult(times=tlist, states=states, expect_values=result_expect)


def steady_state(
    system: LindbladSystem,
    method: str = "eigen",
) -> NDArray[np.complexfloating]:
    """Find the steady state ρ_ss satisfying dρ/dt = 0.

    Args:
        system: Lindblad system (must have time-independent Hamiltonian).
        method: "eigen" (default) solves via the Liouvillian superoperator,
                "evolve" integrates for a long time.

    Returns:
        Steady-state density matrix.
    """
    if callable(system.hamiltonian):
        raise ValueError("steady_state requires a time-independent Hamiltonian")

    d = system.dim

    if method == "eigen":
        return _steady_state_eigen(system, d)
    elif method == "evolve":
        return _steady_state_evolve(system, d)
    else:
        raise ValueError(f"Unknown method: {method}")


def _steady_state_eigen(system: LindbladSystem, d: int) -> NDArray[np.complexfloating]:
    """Find steady state via Liouvillian eigendecomposition."""
    L = _liouvillian(system, d)
    eigenvalues, eigenvectors = np.linalg.eig(L)

    # Steady state is the eigenvector with eigenvalue closest to 0
    idx = np.argmin(np.abs(eigenvalues))
    rho_vec = eigenvectors[:, idx]

    rho = rho_vec.reshape((d, d))
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho)
    return rho


def _steady_state_evolve(system: LindbladSystem, d: int) -> NDArray[np.complexfloating]:
    """Find steady state by long-time evolution."""
    rho = np.eye(d, dtype=complex) / d
    rho = lindblad_evolve(rho, system, dt=0.01, num_steps=10000)
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho)
    return rho


def _liouvillian(system: LindbladSystem, d: int) -> NDArray[np.complexfloating]:
    """Build the Liouvillian superoperator in column-stacking convention.

    L vec(ρ) = vec(dρ/dt)

    where vec(ρ) stacks columns of ρ into a d²-dimensional vector.
    """
    H = system._get_hamiltonian(0.0)
    I = np.eye(d, dtype=complex)

    # -i[H, ρ] = -i(H⊗I - I⊗H^T) vec(ρ)
    L_sup = -1j * (np.kron(H, I) - np.kron(I, H.T))

    for gamma, Lop in system.collapse_ops:
        if gamma == 0:
            continue
        Ldag = Lop.conj().T
        LdagL = Ldag @ Lop

        # γ(L⊗L* - ½(L†L⊗I + I⊗(L†L)^T))
        L_sup += gamma * (
            np.kron(Lop, Lop.conj()) - 0.5 * np.kron(LdagL, I) - 0.5 * np.kron(I, LdagL.T)
        )

    return L_sup
