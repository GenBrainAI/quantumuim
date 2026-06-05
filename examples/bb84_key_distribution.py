"""Example: BB84 Quantum Key Distribution.

Demonstrates secure key exchange between Alice and Bob,
and eavesdropper detection when Eve intercepts the channel.
"""

from quantumium.crypto.bb84 import bb84_protocol
from quantumium.crypto.qrng import quantum_random_bits

print("=" * 55)
print("BB84 Quantum Key Distribution")
print("=" * 55)

# --- Secure channel (no eavesdropper) ---
print("\n--- Scenario 1: Secure Channel ---")
result = bb84_protocol(num_qubits=500, eavesdropper=False, seed=42)
print(f"Qubits sent:        {result.num_qubits_sent}")
print(f"Sifted key length:  {result.sifted_length}")
print(f"Final key length:   {result.final_key_length}")
print(f"QBER:               {result.qber:.2%}")
print(f"Keys match:         {result.alice_key == result.bob_key}")
print(f"Eve detected:       {result.eavesdropper_detected}")
print(f"First 20 key bits:  {''.join(map(str, result.alice_key[:20]))}")

# --- Eavesdropped channel ---
print("\n--- Scenario 2: Eve Intercepts ---")
result_eve = bb84_protocol(num_qubits=500, eavesdropper=True, seed=42)
print(f"Qubits sent:        {result_eve.num_qubits_sent}")
print(f"Sifted key length:  {result_eve.sifted_length}")
print(f"Final key length:   {result_eve.final_key_length}")
print(f"QBER:               {result_eve.qber:.2%}")
print(f"Keys match:         {result_eve.alice_key == result_eve.bob_key}")
print(f"Eve detected:       {result_eve.eavesdropper_detected}")

key_errors = sum(
    a != b for a, b in zip(result_eve.alice_key, result_eve.bob_key)
)
print(f"Key bit errors:     {key_errors}/{result_eve.final_key_length} "
      f"({key_errors/max(1,result_eve.final_key_length):.1%})")

# --- Quantum random number generation ---
print("\n--- Quantum Random Number Generator ---")
bits = quantum_random_bits(64, seed=None)
hex_str = hex(sum(b << i for i, b in enumerate(bits)))
print(f"64 quantum random bits: {hex_str}")
ones = sum(bits)
print(f"Distribution: {ones} ones, {64-ones} zeros ({ones/64:.0%} ones)")

print("\n" + "=" * 55)
