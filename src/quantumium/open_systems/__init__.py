"""Open quantum systems — Lindblad master equation dynamics.

Simulates the evolution of quantum systems coupled to an environment,
modeling decoherence, dissipation, and thermalization through the
Lindblad master equation formalism.

The Lindblad equation:
    dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k† L_k, ρ})

References:
    Lindblad, "On the generators of quantum dynamical semigroups"
    (Commun. Math. Phys. 48, 119-130, 1976)
    Breuer & Petruccione, "The Theory of Open Quantum Systems"
    (Oxford University Press, 2002)
"""

from .lindblad import (
    LindbladSystem,
    lindblad_rhs,
    lindblad_evolve,
    mesolve,
    steady_state,
)
from .operators import (
    decay_operator,
    dephasing_operator,
    thermal_operator,
    sigma_minus,
    sigma_plus,
    number_operator,
    annihilation_operator,
    creation_operator,
)
from .dynamics import (
    EvolutionResult,
    expect,
    purity,
    von_neumann_entropy,
    coherence,
)

__all__ = [
    "LindbladSystem",
    "lindblad_rhs",
    "lindblad_evolve",
    "mesolve",
    "steady_state",
    "decay_operator",
    "dephasing_operator",
    "thermal_operator",
    "sigma_minus",
    "sigma_plus",
    "number_operator",
    "annihilation_operator",
    "creation_operator",
    "EvolutionResult",
    "expect",
    "purity",
    "von_neumann_entropy",
    "coherence",
]
