"""Transpiler optimization passes for quantum circuits.

Each pass takes a list of Instructions and returns an optimized list.
Passes can be composed via the transpile() pipeline.
"""

from __future__ import annotations

import numpy as np

from ..core.circuit import QuantumCircuit, Instruction
from ..core.gate import Gate, CNOT, SWAP


def _gates_equal(a: Gate, b: Gate) -> bool:
    """Check whether two gates have numerically identical matrices.

    Args:
        a (Gate): First gate.
        b (Gate): Second gate.

    Returns:
        bool: ``True`` if the unitary matrices are element-wise close
            within an absolute tolerance of 1e-10.
    """
    return np.allclose(a.matrix, b.matrix, atol=1e-10)


def _is_identity(gate: Gate) -> bool:
    """Check whether a gate is the identity matrix.

    Args:
        gate (Gate): The gate to test.

    Returns:
        bool: ``True`` if the gate's matrix is close to the identity matrix
            of the same dimension.
    """
    return np.allclose(gate.matrix, np.eye(gate.matrix.shape[0], dtype=complex), atol=1e-10)


def _product_gate(a: Gate, b: Gate) -> Gate:
    """Compute the product gate b @ a (b applied after a).

    Args:
        a (Gate): The gate applied first.
        b (Gate): The gate applied second.

    Returns:
        Gate: A new gate whose matrix is ``b.matrix @ a.matrix`` and whose
            name is the concatenation of both gate names.
    """
    matrix = b.matrix @ a.matrix
    return Gate(matrix, f"{b.name}{a.name}")


# --- Individual passes ---


def cancel_inverses(instructions: list[Instruction]) -> list[Instruction]:
    """Remove adjacent gate pairs that multiply to identity.

    Scans for consecutive instructions on the same qubits whose combined
    matrix product equals the identity. This catches self-inverse gates
    (X-X, Y-Y, Z-Z, H-H) as well as any gate followed by its inverse.

    Args:
        instructions (list[Instruction]): The instruction list to optimize.

    Returns:
        list[Instruction]: A new list with inverse pairs removed.
    """
    result: list[Instruction] = []
    i = 0
    while i < len(instructions):
        if i + 1 < len(instructions):
            curr = instructions[i]
            nxt = instructions[i + 1]
            if curr.target_qubits == nxt.target_qubits:
                product = nxt.gate.matrix @ curr.gate.matrix
                if np.allclose(product, np.eye(product.shape[0], dtype=complex), atol=1e-10):
                    i += 2
                    continue
        result.append(instructions[i])
        i += 1
    return result


def remove_identity_gates(instructions: list[Instruction]) -> list[Instruction]:
    """Remove gates that are the identity matrix.

    Any instruction whose gate matrix is numerically equal to the identity
    is dropped, since it has no effect on the quantum state.

    Args:
        instructions (list[Instruction]): The instruction list to filter.

    Returns:
        list[Instruction]: A new list with identity-gate instructions removed.
    """
    return [inst for inst in instructions if not _is_identity(inst.gate)]


def merge_single_qubit_gates(instructions: list[Instruction]) -> list[Instruction]:
    """Merge consecutive single-qubit gates on the same qubit into one unitary.

    Scans for runs of single-qubit instructions targeting the same qubit
    and collapses them into a single gate whose matrix is the product of
    all constituent matrices. If the merged result is the identity, the
    entire run is dropped.

    Args:
        instructions (list[Instruction]): The instruction list to optimize.

    Returns:
        list[Instruction]: A new list with consecutive single-qubit gates
            merged where possible.
    """
    if not instructions:
        return []

    result: list[Instruction] = []
    i = 0
    while i < len(instructions):
        inst = instructions[i]
        if inst.gate.num_qubits != 1:
            result.append(inst)
            i += 1
            continue

        merged_matrix = inst.gate.matrix.copy()
        qubit = inst.target_qubits
        names = [inst.gate.name]
        j = i + 1
        while j < len(instructions):
            nxt = instructions[j]
            if nxt.gate.num_qubits == 1 and nxt.target_qubits == qubit:
                merged_matrix = nxt.gate.matrix @ merged_matrix
                names.append(nxt.gate.name)
                j += 1
            else:
                break

        if j - i == 1:
            result.append(inst)
        elif _is_identity(Gate(merged_matrix)):
            pass
        else:
            merged_name = "+".join(names)
            result.append(Instruction(Gate(merged_matrix, merged_name), qubit))
        i = j
    return result


def cancel_cnots(instructions: list[Instruction]) -> list[Instruction]:
    """Remove adjacent CNOT pairs on the same qubits (CNOT^2 = I).

    Two consecutive CNOT gates with identical control and target qubits
    cancel each other because CNOT is self-inverse.

    Args:
        instructions (list[Instruction]): The instruction list to optimize.

    Returns:
        list[Instruction]: A new list with adjacent CNOT pairs removed.
    """
    result: list[Instruction] = []
    i = 0
    while i < len(instructions):
        if i + 1 < len(instructions):
            curr = instructions[i]
            nxt = instructions[i + 1]
            if (
                curr.gate.num_qubits == 2
                and nxt.gate.num_qubits == 2
                and curr.target_qubits == nxt.target_qubits
                and _gates_equal(curr.gate, CNOT)
                and _gates_equal(nxt.gate, CNOT)
            ):
                i += 2
                continue
        result.append(instructions[i])
        i += 1
    return result


def commute_and_cancel(instructions: list[Instruction]) -> list[Instruction]:
    """Move commuting gates closer together to enable cancellation.

    Single-qubit gates on disjoint qubits commute freely. This pass
    iteratively swaps neighbouring instructions that act on non-overlapping
    qubit sets whenever doing so brings an inverse pair adjacent. After
    reaching a fixed point, ``cancel_inverses`` is applied to eliminate the
    newly adjacent inverse pairs.

    Args:
        instructions (list[Instruction]): The instruction list to optimize.

    Returns:
        list[Instruction]: A new list with commuted and cancelled gates.
    """
    working = list(instructions)
    changed = True
    while changed:
        changed = False
        for i in range(len(working) - 1):
            a = working[i]
            b = working[i + 1]
            qubits_a = set(a.target_qubits)
            qubits_b = set(b.target_qubits)
            if not qubits_a.isdisjoint(qubits_b):
                continue
            should_swap = False
            if i > 0:
                prev = working[i - 1]
                if prev.target_qubits == b.target_qubits:
                    product = b.gate.matrix @ prev.gate.matrix
                    if np.allclose(product, np.eye(product.shape[0], dtype=complex), atol=1e-10):
                        should_swap = True
            if not should_swap and i + 2 < len(working):
                nxt = working[i + 2]
                if a.target_qubits == nxt.target_qubits:
                    product = nxt.gate.matrix @ a.gate.matrix
                    if np.allclose(product, np.eye(product.shape[0], dtype=complex), atol=1e-10):
                        should_swap = True
            if should_swap:
                working[i], working[i + 1] = working[i + 1], working[i]
                changed = True

    return cancel_inverses(working)


def decompose_swap(instructions: list[Instruction]) -> list[Instruction]:
    """Decompose SWAP gates into three CNOT gates.

    Applies the identity SWAP(a,b) = CNOT(a,b) . CNOT(b,a) . CNOT(a,b).
    Non-SWAP instructions are passed through unchanged.

    Args:
        instructions (list[Instruction]): The instruction list to process.

    Returns:
        list[Instruction]: A new list with every SWAP replaced by three
            CNOT instructions.
    """
    result: list[Instruction] = []
    for inst in instructions:
        if inst.gate.num_qubits == 2 and _gates_equal(inst.gate, SWAP):
            a, b = inst.target_qubits
            result.append(Instruction(CNOT, (a, b)))
            result.append(Instruction(CNOT, (b, a)))
            result.append(Instruction(CNOT, (a, b)))
        else:
            result.append(inst)
    return result


def count_gates(instructions: list[Instruction]) -> dict[str, int]:
    """Count the number of gates of each type in the instruction list.

    Args:
        instructions (list[Instruction]): The instructions to tally.

    Returns:
        dict[str, int]: Mapping of gate name to occurrence count.
    """
    counts: dict[str, int] = {}
    for inst in instructions:
        name = inst.gate.name
        counts[name] = counts.get(name, 0) + 1
    return counts


def circuit_depth(instructions: list[Instruction], num_qubits: int) -> int:
    """Compute the circuit depth (longest path through the circuit).

    The depth equals the maximum number of sequential gate layers across
    all qubits, determined by tracking the latest time-step on each qubit.

    Args:
        instructions (list[Instruction]): The instructions comprising the
            circuit.
        num_qubits (int): Total number of qubits in the circuit.

    Returns:
        int: The circuit depth. Returns 0 for an empty circuit.
    """
    qubit_depth = [0] * num_qubits
    for inst in instructions:
        max_d = max(qubit_depth[q] for q in inst.target_qubits)
        for q in inst.target_qubits:
            qubit_depth[q] = max_d + 1
    return max(qubit_depth) if qubit_depth else 0


# --- Transpile pipeline ---

DEFAULT_PASSES = [
    remove_identity_gates,
    cancel_inverses,
    cancel_cnots,
    merge_single_qubit_gates,
    remove_identity_gates,
]


def transpile(
    circuit: QuantumCircuit,
    passes: list | None = None,  # type: ignore[type-arg]
    repeat: int = 2,
) -> QuantumCircuit:
    """Run optimization passes on a circuit.

    Args:
        circuit: Input circuit.
        passes: List of pass functions. Uses DEFAULT_PASSES if None.
        repeat: Number of times to repeat the pass pipeline (fixed-point iteration).

    Returns:
        New optimized QuantumCircuit.
    """
    if passes is None:
        passes = DEFAULT_PASSES

    instructions = circuit.instructions

    for _ in range(repeat):
        prev_count = len(instructions)
        for pass_fn in passes:
            instructions = pass_fn(instructions)
        if len(instructions) == prev_count:
            break

    result = QuantumCircuit(circuit.num_qubits)
    for inst in instructions:
        result.apply(inst.gate, *inst.target_qubits)
    return result
