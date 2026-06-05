"""Example: Factor 15 using Shor's algorithm.

Demonstrates the educational implementation of Shor's factoring
algorithm using quantum order-finding via QPE.
"""

from quantumium.algorithms.shor import shor_factor

print("=" * 50)
print("Shor's Algorithm — Factor 15")
print("=" * 50)

# Factor 15 with base a=7
# Order of 7 mod 15 is 4, so 7^2 = 49 ≡ 4 mod 15
# gcd(4+1, 15) = 5, gcd(4-1, 15) = 3
result = shor_factor(N=15, a=7, num_counting_qubits=5)

print(f"\nN = {result.N}")
print(f"Base a = {result.a}")
print(f"Phase estimate = {result.phase_estimate:.4f}")
print(f"Order found = {result.order}")
print(f"Success = {result.success}")

if result.success and result.factors:
    f1, f2 = result.factors
    print(f"\nFactors: {f1} × {f2} = {f1 * f2}")
else:
    print("\nFactoring failed on this attempt (probabilistic algorithm)")

print("\n" + "=" * 50)
print("Trivial cases")
print("=" * 50)

# Even number — trivial
r2 = shor_factor(N=6)
print(f"\nN=6: factors = {r2.factors}")

# GCD shortcut
r3 = shor_factor(N=15, a=3)
print(f"N=15, a=3: gcd(3,15)=3, factors = {r3.factors}")
