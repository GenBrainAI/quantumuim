# Security Policy

## Supported Versions

Security fixes are applied to the latest minor release line.

| Version | Supported |
| ------- | --------- |
| latest release | ✅ |
| older releases | ❌ |

## Reporting a Vulnerability

Please **do not open a public GitHub issue** for security vulnerabilities.

Instead, report privately via one of:

1. **GitHub private vulnerability reporting** — use the *Security* tab → *Report a vulnerability* on this repository (preferred).
2. **Email** — security@genbrain.ai with subject line `[SECURITY] quantumium: <short summary>`.

Include where possible:

- A description of the vulnerability and its impact
- Steps to reproduce (proof-of-concept code or circuit input)
- Affected version(s) / commit hash
- Any suggested mitigation

## What to Expect

- **Acknowledgement** within 3 business days.
- **Initial assessment** (severity, affected versions) within 7 business days.
- **Fix or mitigation plan** communicated within 30 days for confirmed issues; critical issues are prioritized for an out-of-band patch release.
- You will be credited in the release notes unless you ask otherwise.

## Scope Notes

quantumium is a numerical/quantum-simulation library. Reports of particular interest:

- Arbitrary code execution via crafted circuit files, QASM input, or serialized objects
- Path traversal or unsafe file handling in I/O utilities
- Dependency-chain vulnerabilities that are exploitable through our public API

Out of scope: results-accuracy bugs without a security impact (please file a regular issue), and vulnerabilities only reachable with already-privileged local access.
