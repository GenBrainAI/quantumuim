"""Quantum circuit construction and execution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .gate import Gate
from .noise import NoiseChannel, NoiseModel


@dataclass
class Instruction:
    gate: Gate
    target_qubits: tuple[int, ...]


class QuantumCircuit:
    """A quantum circuit that applies gates to qubits."""

    def __init__(self, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError("Circuit must have at least 1 qubit")
        self.num_qubits = num_qubits
        self._instructions: list[Instruction] = []

    def apply(self, gate: Gate, *qubits: int) -> "QuantumCircuit":
        """Apply a gate to the specified qubit(s). Returns self for chaining."""
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                raise IndexError(f"Qubit index {q} out of range for {self.num_qubits}-qubit circuit")
        if gate.num_qubits != len(qubits):
            raise ValueError(
                f"Gate {gate.name} requires {gate.num_qubits} qubits, got {len(qubits)}"
            )
        if len(set(qubits)) != len(qubits):
            raise ValueError("Duplicate qubit indices")
        self._instructions.append(Instruction(gate, tuple(qubits)))
        return self

    @property
    def instructions(self) -> list[Instruction]:
        return list(self._instructions)

    @property
    def depth(self) -> int:
        if not self._instructions:
            return 0
        layers: list[set[int]] = []
        for inst in self._instructions:
            placed = False
            for layer in reversed(layers):
                if any(q in layer for q in inst.target_qubits):
                    break
                placed = True
            if not placed:
                layers.append(set(inst.target_qubits))
            else:
                idx = layers.index(layer) + 1 if not placed else len(layers)
                if idx >= len(layers):
                    layers.append(set(inst.target_qubits))
                else:
                    layers[idx].update(inst.target_qubits)
        return len(layers)

    def statevector(self) -> NDArray[np.complexfloating]:
        """Simulate the circuit and return the final state vector."""
        n = self.num_qubits
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0

        for inst in self._instructions:
            full_matrix = self._expand_gate(inst.gate.matrix, inst.target_qubits, n)
            state = full_matrix @ state
        return state

    def _expand_gate(
        self,
        gate_matrix: NDArray[np.complexfloating],
        target_qubits: tuple[int, ...],
        num_qubits: int,
    ) -> NDArray[np.complexfloating]:
        dim = 2**num_qubits
        result = np.zeros((dim, dim), dtype=complex)

        gate_size = len(target_qubits)
        for i in range(dim):
            for j in range(dim):
                other_bits_i = [((i >> q) & 1) for q in range(num_qubits) if q not in target_qubits]
                other_bits_j = [((j >> q) & 1) for q in range(num_qubits) if q not in target_qubits]

                if other_bits_i != other_bits_j:
                    continue

                gate_i = 0
                gate_j = 0
                for idx, q in enumerate(target_qubits):
                    gate_i |= ((i >> q) & 1) << (gate_size - 1 - idx)
                    gate_j |= ((j >> q) & 1) << (gate_size - 1 - idx)

                result[i, j] = gate_matrix[gate_i, gate_j]

        return result

    def density_matrix(
        self,
        noise_model: NoiseModel | None = None,
    ) -> NDArray[np.complexfloating]:
        """Simulate the circuit as a density matrix, optionally with noise.

        Without a noise model this is equivalent to |ψ⟩⟨ψ| from statevector().
        With a noise model, Kraus operators are applied after each gate.
        """
        n = self.num_qubits
        dim = 2**n
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0

        for inst in self._instructions:
            U = self._expand_gate(inst.gate.matrix, inst.target_qubits, n)
            rho = U @ rho @ U.conj().T

            if noise_model is not None:
                channel = noise_model.get_noise(inst.gate.name, inst.gate.num_qubits)
                if channel is not None:
                    for q in inst.target_qubits:
                        rho = self._apply_single_qubit_channel(rho, channel, q, n)

        return rho

    @staticmethod
    def _apply_single_qubit_channel(
        rho: NDArray[np.complexfloating],
        channel: NoiseChannel,
        qubit: int,
        num_qubits: int,
    ) -> NDArray[np.complexfloating]:
        """Apply a single-qubit noise channel to one qubit of a density matrix."""
        dim = 2**num_qubits
        result = np.zeros((dim, dim), dtype=complex)
        for kraus in channel.kraus_operators:
            K = np.eye(1, dtype=complex)
            for q in range(num_qubits):
                K = np.kron(K, kraus if q == qubit else np.eye(2, dtype=complex))
            result += K @ rho @ K.conj().T
        return result

    def noisy_measure(
        self,
        noise_model: NoiseModel,
        shots: int = 1024,
    ) -> dict[str, int]:
        """Run the circuit with noise and sample measurement outcomes."""
        rho = self.density_matrix(noise_model)
        probabilities = np.real(np.diag(rho))
        probabilities = np.clip(probabilities, 0, None)
        probabilities /= probabilities.sum()
        indices = np.arange(len(probabilities))
        samples = np.random.choice(indices, size=shots, p=probabilities)
        counts: dict[str, int] = {}
        for s in samples:
            bitstring = format(s, f"0{self.num_qubits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return dict(sorted(counts.items()))

    def measure_all(self, shots: int = 1024) -> dict[str, int]:
        """Measure all qubits in the computational basis and return bitstring counts."""
        state = self.statevector()
        probabilities = np.abs(state) ** 2
        indices = np.arange(len(state))
        samples = np.random.choice(indices, size=shots, p=probabilities)
        counts: dict[str, int] = {}
        for s in samples:
            bitstring = format(s, f"0{self.num_qubits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return dict(sorted(counts.items()))

    def __repr__(self) -> str:
        return f"QuantumCircuit(qubits={self.num_qubits}, gates={len(self._instructions)})"
