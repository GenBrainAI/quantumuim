"""Grover's algorithm — search for a marked item in an unsorted database."""

from quantumium.algorithms.grover import grover_search

num_qubits = 4
target = 11

print(f"Searching for state |{target}> in {2**num_qubits} items ({num_qubits} qubits)")
qc = grover_search(num_qubits, [target])
counts = qc.measure_all(shots=1000)

print("\nTop results:")
for bitstring, count in sorted(counts.items(), key=lambda x: -x[1])[:5]:
    print(f"  |{bitstring}> = {int(bitstring, 2):>2d}  ({count/10:.1f}%)")
