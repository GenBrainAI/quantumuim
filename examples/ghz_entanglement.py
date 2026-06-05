"""GHZ state — maximally entangled multi-qubit state."""

from quantumium.circuits.templates import ghz_state

for n in [3, 4, 5]:
    qc = ghz_state(n)
    counts = qc.measure_all(shots=2000)

    print(f"\n{n}-qubit GHZ state:")
    for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count > 10:
            print(f"  |{bitstring}>  {count/20:.1f}%")
