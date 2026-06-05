"""Create and measure a Bell state (|00⟩ + |11⟩) / √2."""

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT

qc = QuantumCircuit(2)
qc.apply(H, 0)
qc.apply(CNOT, 0, 1)

print("Statevector:", qc.statevector())
print("Measurement counts:", qc.measure_all(shots=1000))
