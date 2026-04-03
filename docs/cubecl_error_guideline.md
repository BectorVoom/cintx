Below is an English document draft you can use for Codex.

---

# Manual for Documenting Resolved Build Errors Related to the Rust `cubecl` Crate

## 1. Purpose

This document defines how to create a clear and reusable resolution manual when a build error occurs in a Rust project that uses the `cubecl` crate and the issue has been resolved.

The goal is to ensure that:

* the root cause is recorded accurately,
* the resolution can be reproduced,
* similar incidents can be resolved faster in the future, and
* Codex can reference a consistent troubleshooting format.

## 2. Scope

This manual applies to:

* Rust projects that depend on the `cubecl` crate,
* build failures during local development, CI, or release pipelines,
* errors caused by dependency mismatches, feature flags, toolchain issues, platform-specific behavior, or configuration problems,
* cases where the issue has already been identified and fixed.

This manual does not replace a full postmortem for major production incidents. It is intended for build-error knowledge capture and operational reuse.

## 3. When to Create This Document

Create this document when all of the following conditions are met:

1. A build error occurred in a Rust project using the `cubecl` crate.
2. The cause of the error was identified.
3. A working fix or workaround was confirmed.
4. The result can be reproduced or validated by another engineer or automation workflow.

## 4. Intended Audience

This document is written for:

* Codex operators,
* Rust developers,
* CI/CD maintainers,
* build and release engineers,
* team members who may encounter the same issue later.

## 5. Documentation Principles

When writing the resolution record, follow these principles:

### 5.1 Be specific

Do not write vague statements such as “dependency issue” or “version conflict fixed.”
State exactly what failed, where it failed, and what changed.

### 5.2 Be reproducible

Include the exact commands, environment, versions, and conditions required to reproduce the error and verify the fix.

### 5.3 Separate symptoms from root cause

The visible error message is not always the real cause. Record both clearly.

### 5.4 Record the final verified state

Document the final working configuration, not only the intermediate debugging steps.

### 5.5 Prefer durable knowledge

Focus on information that helps future troubleshooting:

* dependency versions,
* Cargo configuration,
* feature flags,
* Rust toolchain version,
* target platform,
* environment variables,
* CI configuration changes.

## 6. Required Sections

Every resolved build-error document should contain the following sections.

### 6.1 Title

Use a title with the following format:

**Resolved Build Error: `[short error summary]` in project using `cubecl`**

Example:

**Resolved Build Error: feature mismatch during Cargo build in project using `cubecl`**

### 6.2 Summary

Provide a short overview of:

* what failed,
* what caused it,
* what fixed it.

Example:

> The build failed because the `cubecl` dependency was compiled with incompatible feature settings across workspace members. The issue was resolved by aligning feature flags in `Cargo.toml` and updating the lockfile.

### 6.3 Impact

Describe the operational impact:

* local build blocked,
* CI pipeline failed,
* release delayed,
* specific target platform affected.

### 6.4 Environment

Record the environment in detail:

* Project name
* Repository or workspace name
* OS
* Architecture
* Rust version
* Cargo version
* Toolchain channel
* Target triple
* `cubecl` version
* Related dependency versions
* CI runner information, if applicable

### 6.5 Error Details

Include:

* the exact command executed,
* the exact error message,
* relevant log excerpts,
* where the error occurred.

Do not include unnecessary full logs if a shorter excerpt is sufficient.

### 6.6 Root Cause

Explain the actual reason for failure.
This section should answer:

* Why did the build fail?
* Why did the error appear at that time?
* Why was the configuration invalid or inconsistent?

### 6.7 Resolution

Describe the fix in a step-by-step form.

Include:

* files changed,
* versions updated,
* flags added or removed,
* commands executed,
* configuration values corrected.

### 6.8 Verification

List how the fix was confirmed.

Examples:

* `cargo build` succeeded,
* `cargo test` passed,
* CI pipeline completed successfully,
* affected target built without error.

### 6.9 Prevention

Describe what should be done to avoid recurrence.

Examples:

* pin dependency versions,
* standardize workspace features,
* add CI validation,
* document required toolchain version,
* add a pre-build check.

### 6.10 References

Include related internal references such as:

* issue tracker ID,
* pull request ID,
* commit hash,
* CI job URL,
* internal troubleshooting page.

## 7. Standard Writing Template

Use the following template for each resolved issue.

---

# Resolved Build Error: [Short Error Summary] in Project Using `cubecl`

## Summary

[Provide a concise description of the issue, root cause, and final fix.]

## Impact

[Describe who or what was affected.]

## Environment

* Project:
* Repository/Workspace:
* OS:
* Architecture:
* Rust Version:
* Cargo Version:
* Toolchain:
* Target Triple:
* `cubecl` Version:
* Related Dependencies:
* CI Environment:

## Trigger Condition

[Describe when the issue occurs and under what conditions.]

## Command That Failed

```bash
[Insert command]
```

## Observed Error

```text
[Insert relevant error output]
```

## Root Cause

[Explain the actual cause of the failure.]

## Resolution

1. [Step 1]
2. [Step 2]
3. [Step 3]

### Files Changed

* [file path]
* [file path]

### Configuration or Version Changes

```toml
[Insert relevant Cargo.toml or configuration diff if needed]
```

## Verification

```bash
[Insert verification commands]
```

[Describe the successful result.]

## Prevention

* [Preventive action 1]
* [Preventive action 2]
* [Preventive action 3]

## References

* Issue:
* Pull Request:
* Commit:
* CI Job:
* Additional Notes:

---

## 8. Example Entry

Below is an example of a properly written resolution document.

---

# Resolved Build Error: Workspace Feature Mismatch in Project Using `cubecl`

## Summary

A Cargo build failed in a Rust workspace using `cubecl` because different workspace members enabled inconsistent features for the same dependency chain. This caused compilation to fail during dependency resolution and code generation. The issue was resolved by standardizing dependency declarations and feature flags across all workspace crates.

## Impact

* Local development builds were blocked.
* CI failed on all pull requests affecting the workspace.
* Release validation could not complete.

## Environment

* Project: compute-backend
* Repository/Workspace: gpu-services-monorepo
* OS: Ubuntu 26.04
* Architecture: x86_64
* edition = "2024"
* Toolchain: stable
* Target Triple: x86_64-unknown-linux-gnu
* `cubecl` Version: 0.9.0
* gpu device :wgpu
* Related Dependencies: workspace internal crates, GPU backend dependencies
* CI Environment: GitHub Actions, Ubuntu runner

## Trigger Condition

The issue occurred when running a full workspace build after updating one crate to use a different dependency configuration from the rest of the workspace.

## Command That Failed

```bash
cargo build --workspace
```

## Observed Error

```text
error: failed to compile due to incompatible feature configuration across workspace dependencies
```

## Root Cause

One workspace member declared dependency settings that were not aligned with the rest of the workspace. As a result, Cargo resolved a dependency graph that introduced incompatible feature combinations affecting the build path used by `cubecl`.

## Resolution

1. Reviewed all workspace `Cargo.toml` files that referenced the affected dependency chain.
2. Standardized feature flags across crates.
3. Removed duplicate or conflicting dependency declarations.
4. Regenerated the lockfile.
5. Re-ran the full build and test workflow.

### Files Changed

* `Cargo.toml`
* `crates/backend/Cargo.toml`
* `crates/runtime/Cargo.toml`
* `Cargo.lock`

### Configuration or Version Changes

```toml
[workspace.dependencies]
cubecl = "0.9.0"
```

## Verification

```bash
cargo build --workspace
cargo test --workspace
```

The build and tests completed successfully in both local and CI environments.

## Prevention

* Define shared dependencies in the workspace root whenever possible.
* Review feature flags before merging dependency changes.
* Add CI checks for dependency consistency.
* Document the supported Rust toolchain version.

## References

* Issue: BUILD-142
* Pull Request: #381
* Commit: abcdef123456
* CI Job: build-and-test / run 2481
* Additional Notes: This issue was only visible in full workspace builds.

---

## 9. Review Checklist

Before publishing the document, confirm the following:

* The error message is included.
* The root cause is explicitly stated.
* The exact fix is recorded.
* The working environment is documented.
* Verification steps are included.
* Preventive actions are listed.
* The content is specific enough for another engineer to repeat the fix.

## 10. Recommended File Naming

Use a consistent naming convention such as:

`resolved-build-error-cubecl-[short-topic].md`

Examples:

* `resolved-build-error-cubecl-feature-mismatch.md`
* `resolved-build-error-cubecl-toolchain-version.md`
* `resolved-build-error-cubecl-ci-config.md`

## 11. Manual Directory
* `/home/chemtech/workspace/cintx/docs/manual/Cubecl`


## 12. Final Note

A good resolution document should allow another engineer or Codex workflow to answer these questions without additional investigation:

* What failed?
* Why did it fail?
* What changed to fix it?
* How was the fix verified?
* How can the same issue be prevented next time?

---
