"""Deutsch-Jozsa algorithm — determine if a function is constant or balanced."""

from quantumium.algorithms.deutsch_jozsa import classify_function

print("Deutsch-Jozsa Algorithm Demo")
print("=" * 40)

print("\nf(x) = 0 (constant):")
result = classify_function(4, lambda x: 0)
print(f"  Classification: {result}")

print("\nf(x) = 1 (constant):")
result = classify_function(4, lambda x: 1)
print(f"  Classification: {result}")

print("\nf(x) = x mod 2 (balanced):")
result = classify_function(4, lambda x: x % 2)
print(f"  Classification: {result}")

print("\nf(x) = parity of x (balanced):")
result = classify_function(4, lambda x: bin(x).count("1") % 2)
print(f"  Classification: {result}")
