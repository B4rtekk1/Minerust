# Security Policy

## Supported Versions

Only the latest release is actively maintained and receives security updates.

| Version | Supported |
| ------- | --------- |
| Latest (main) | ✅ |
| Older releases | ❌ |

---

## Reporting a Vulnerability

If you discover a security vulnerability in Render3D, **please do not open a public GitHub issue**. Public disclosure before a fix is available can put users at risk.

Instead, report it privately via email:

📧 **bartoszkasyna@gmail.com**

Please include in your report:
- A clear description of the vulnerability
- Steps to reproduce
- Potential impact (what could an attacker do?)
- Your suggested fix, if any

---

## What to Expect

- You will receive an acknowledgement within **72 hours**.
- We will investigate and aim to release a fix within **14 days** for critical issues.
- You will be credited in the release notes if you wish (just let us know).

---

## Scope

Areas most relevant to security in this project:

- **Multiplayer networking** — QUIC/UDP server/client communication, packet validation, state synchronization
- **World file parsing** — loading `.r3d` world files or `settings.bin` from untrusted sources
- **Shader execution** — WGSL shaders compiled and run on the GPU

Out of scope:
- Issues in third-party dependencies (please report those upstream to the relevant crate maintainers)
- Performance issues that are not exploitable as denial-of-service

---

## Disclosure Policy

Once a fix is released, we will publish a summary of the vulnerability and the fix in the GitHub releases/changelog. We follow a **coordinated disclosure** approach — we ask that you give us reasonable time to patch before any public disclosure.

---

Thank you for helping keep Render3D and its users safe.
