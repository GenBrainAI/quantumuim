"""Symbolic parameters for parameterized quantum circuits.

Enables building circuit templates with symbolic rotation angles that
can be efficiently bound to concrete values for VQE/QAOA workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .gate import Gate, Rx, Ry, Rz
from .circuit import QuantumCircuit


@dataclass(frozen=True)
class Parameter:
    """A symbolic parameter for rotation gates."""

    name: str

    def __repr__(self) -> str:
        return f"Parameter({self.name!r})"


class ParameterVector:
    """A named vector of parameters for convenience."""

    def __init__(self, name: str, length: int) -> None:
        self.name = name
        self.params = [Parameter(f"{name}[{i}]") for i in range(length)]

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, idx: int) -> Parameter:
        return self.params[idx]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.params)


GATE_FACTORIES: dict[str, Callable[[float], Gate]] = {
    "Rx": Rx,
    "Ry": Ry,
    "Rz": Rz,
}


@dataclass
class ParameterizedInstruction:
    """A circuit instruction that may contain a symbolic parameter."""

    gate_type: str
    parameter: Parameter | None
    fixed_gate: Gate | None
    target_qubits: tuple[int, ...]


class ParameterizedCircuit:
    """A quantum circuit template with symbolic parameters.

    Build the template with apply() for fixed gates and apply_parameterized()
    for rotation gates with symbolic parameters. Call bind() with a parameter
    dict to produce a concrete QuantumCircuit.
    """

    def __init__(self, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError("Circuit must have at least 1 qubit")
        self.num_qubits = num_qubits
        self._instructions: list[ParameterizedInstruction] = []
        self._parameters: set[Parameter] = set()

    @property
    def parameters(self) -> set[Parameter]:
        return set(self._parameters)

    @property
    def num_parameters(self) -> int:
        return len(self._parameters)

    def apply(self, gate: Gate, *qubits: int) -> ParameterizedCircuit:
        """Add a fixed (non-parameterized) gate."""
        self._instructions.append(
            ParameterizedInstruction(
                gate_type="fixed",
                parameter=None,
                fixed_gate=gate,
                target_qubits=tuple(qubits),
            )
        )
        return self

    def rx(self, param: Parameter, qubit: int) -> ParameterizedCircuit:
        """Add a parameterized Rx rotation."""
        self._parameters.add(param)
        self._instructions.append(
            ParameterizedInstruction(
                gate_type="Rx",
                parameter=param,
                fixed_gate=None,
                target_qubits=(qubit,),
            )
        )
        return self

    def ry(self, param: Parameter, qubit: int) -> ParameterizedCircuit:
        """Add a parameterized Ry rotation."""
        self._parameters.add(param)
        self._instructions.append(
            ParameterizedInstruction(
                gate_type="Ry",
                parameter=param,
                fixed_gate=None,
                target_qubits=(qubit,),
            )
        )
        return self

    def rz(self, param: Parameter, qubit: int) -> ParameterizedCircuit:
        """Add a parameterized Rz rotation."""
        self._parameters.add(param)
        self._instructions.append(
            ParameterizedInstruction(
                gate_type="Rz",
                parameter=param,
                fixed_gate=None,
                target_qubits=(qubit,),
            )
        )
        return self

    def bind(self, values: dict[Parameter, float]) -> QuantumCircuit:
        """Bind parameter values and return a concrete QuantumCircuit."""
        missing = self._parameters - set(values.keys())
        if missing:
            names = ", ".join(p.name for p in missing)
            raise ValueError(f"Missing parameter values: {names}")

        qc = QuantumCircuit(self.num_qubits)
        for inst in self._instructions:
            if inst.gate_type == "fixed":
                qc.apply(inst.fixed_gate, *inst.target_qubits)  # type: ignore[arg-type]
            else:
                theta = values[inst.parameter]  # type: ignore[index]
                gate = GATE_FACTORIES[inst.gate_type](theta)
                qc.apply(gate, *inst.target_qubits)
        return qc

    def bind_array(
        self,
        values: NDArray[np.floating] | list[float],
        parameter_order: list[Parameter] | None = None,
    ) -> QuantumCircuit:
        """Bind parameters from a flat array (convenient for optimizers).

        Args:
            values: Array of parameter values.
            parameter_order: Order of parameters matching the array.
                If None, uses sorted order by parameter name.
        """
        if parameter_order is None:
            parameter_order = sorted(self._parameters, key=lambda p: p.name)

        if len(values) != len(parameter_order):
            raise ValueError(f"Expected {len(parameter_order)} values, got {len(values)}")

        param_dict = {p: float(v) for p, v in zip(parameter_order, values)}
        return self.bind(param_dict)

    def __repr__(self) -> str:
        return (
            f"ParameterizedCircuit(qubits={self.num_qubits}, "
            f"gates={len(self._instructions)}, "
            f"parameters={self.num_parameters})"
        )


def hardware_efficient_template(
    num_qubits: int,
    depth: int = 1,
) -> tuple[ParameterizedCircuit, list[Parameter]]:
    """Create a hardware-efficient ansatz template.

    Same structure as VQE's ansatz but with symbolic parameters.
    Returns the circuit and the parameter list in order.
    """
    from .gate import CNOT

    params = []
    pqc = ParameterizedCircuit(num_qubits)

    for layer in range(depth):
        for q in range(num_qubits):
            p_ry = Parameter(f"ry_{layer}_{q}")
            p_rz = Parameter(f"rz_{layer}_{q}")
            params.extend([p_ry, p_rz])
            pqc.ry(p_ry, q)
            pqc.rz(p_rz, q)
        for q in range(num_qubits - 1):
            pqc.apply(CNOT, q, q + 1)

    return pqc, params
