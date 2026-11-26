# Contributing to Minerust

Thank you for your interest in contributing! This document will help you get started.

> For a broader overview of the codebase, see [DEVELOPMENT.md](DEVELOPMENT.md) and [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md).

---

## Table of Contents

- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Pull Requests](#pull-requests)
- [Code Style](#code-style)
- [Building & Testing](#building--testing)
- [Code of Conduct](#code-of-conduct)

---

## Reporting Bugs

- Check if the issue has already been reported in [GitHub Issues](https://github.com/B4rtekk1/Minerust/issues).
- If not, open a new issue with:
  - A clear title and description
  - Steps to reproduce
  - Your OS, GPU, and driver version
  - Whether you're running a debug or release build
  - Relevant application logs from the platform-specific application data directory

---

## Suggesting Enhancements

- Open an issue to discuss your idea before starting work — this avoids wasted effort if the direction doesn't fit the project.
- Describe the problem you're solving, not just the solution.

---

## Pull Requests

1. **Fork** the repository and clone your fork locally.
2. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** — keep PRs focused on a single feature or fix.
4. **Format and lint** before committing (see [Code Style](#code-style)).
5. **Test** your changes locally in both debug and release mode.
6. **Push** to your fork and open a Pull Request with a clear description of what you changed and why.

> ⚠️ Note: first-time compilation can take **up to 5 minutes** due to shader compilation and Rust's compile times. This is expected.

---

## Code Style

Before submitting, always run:

```bash
# Auto-format code
cargo fmt

# Check for common issues
cargo clippy --release
```

PRs that fail `clippy` or are not formatted with `cargo fmt` will be asked to fix those before merging.

Additional guidelines:
- Write clear comments, especially in rendering and GPU-related code.
- Keep shader code (`.wgsl`) readable — add comments for non-obvious passes.
- Match naming conventions and module structure already present in the codebase.

---

## Building & Testing

```bash
# Debug build (fast to compile, slow to run)
cargo build

# Release build (recommended for testing performance)
cargo build --release

# Run
cargo run --release

# Run tests
cargo test --lib
```

> Vulkan is the recommended backend for best performance. Make sure your GPU supports Vulkan 1.2+, DirectX 12, or Metal.

---

## Code of Conduct

Please be respectful and constructive in all interactions. This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

---

Thanks for contributing! 🚀
