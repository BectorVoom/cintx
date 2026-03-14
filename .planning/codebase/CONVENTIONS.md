# Coding Conventions

**Analysis Date:** 2026-03-14

## Naming Patterns

**Files:**
- Core library C files use lowercase names with underscores and numeric integral family markers (for example `cint1e.c`, `cint2e.c`, `rys_roots.c`, `g2e.c`).
- Public C API declarations are centralized in `include/cint_funcs.h`.
- Auto-generated integral implementations live under `src/autocode/*.c`.
- Python test files use `test_*.py` names in a flat `testsuite` directory.
- The Rust crate currently has only `src/main.rs` as a minimal binary entrypoint.

**Functions:**
- Public/internal C functions commonly use `CINT`-prefixed names with mixed case and numeric family identifiers (for example `CINT1e_loop`, `CINTall_2e_optimizer`).
- Compatibility wrappers use lowercase `c...` names with `_cart`, `_sph`, `_spinor` suffixes.
- Python helpers and test routines use snake_case (for example `make_cintopt`, `test_int1e_sph`, `test_rys_roots_weights`).

**Variables:**
- C local variables are snake_case (`i_prim`, `expcutoff`, `common_factor`).
- C preprocessor constants/macros are upper snake case (`MIN`, `MAX`, `SQUARE`, `MALLOC_INSTACK`).
- Python constants are upper snake case (`PTR_EXPCUTOFF`, `ANG_OF`), locals are snake_case (`natm`, `nbas`, `off`).

**Types:**
- C typedef-based API names use a `CINT` prefix (`CINTOptimizerFunction`, `CINTIntegralFunction`).
- C integer alias `FINT` is used pervasively in API signatures and locals.
- Python ctypes struct names use CamelCase (`class CINTEnvVars`).

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/CMakeLists.txt:66`
- `/home/chemtech/workspace/cintx/libcint-master/include/cint_funcs.h:12`
- `/home/chemtech/workspace/cintx/libcint-master/src/cint1e.c:39`
- `/home/chemtech/workspace/cintx/libcint-master/src/misc.h:11`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_cint.py:19`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_rys_roots.py:87`
- `/home/chemtech/workspace/cintx/src/main.rs:1`

## Code Style

**Formatting:**
- C functions place the opening brace on the next line and use explicit braces for control blocks.
- Multi-line argument lists align continuation lines for readability in numerical kernels.
- C preprocessor macro bodies use backslash-continued lines.
- CMake files use two-space indentation in control blocks.
- Rust formatting follows standard rustfmt-style defaults (4-space indentation, braces on same line for `fn main`).

**Linting:**
- No repository-level lint/format configuration was found for C, Python, or Rust (`.clang-format`, `.clang-tidy`, `.rustfmt.toml`, `pytest.ini`, `pyproject.toml` absent in this tree).
- Style is inferred from existing source rather than a checked-in formatter/linter policy.

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/src/cint1e.c:39`
- `/home/chemtech/workspace/cintx/libcint-master/src/cint1e.c:188`
- `/home/chemtech/workspace/cintx/libcint-master/src/misc.h:26`
- `/home/chemtech/workspace/cintx/libcint-master/CMakeLists.txt:10`
- `/home/chemtech/workspace/cintx/src/main.rs:1`

## Import Organization

**Order:**
1. C files include system headers first, then project headers.
2. Python test files typically import stdlib modules first, then third-party modules (`numpy`, `pyscf`), then local script modules.
3. Build dependencies in CMake are declared via `include(...)` and `find_package(...)` before target/link setup.

**Grouping:**
- C files use contiguous include blocks with a visual split between standard and local headers.
- Python files frequently keep a small grouped import block, then setup statements (`sys.path.insert(...)`) when needed.
- There is no explicit alphabetical import sorting rule enforced in code.

**Path Aliases:**
- No alias system is used in C or Rust imports.
- Python tests rely on relative paths (`os.path.join(__file__, '../../build/...')`) and explicit `sys.path.insert(...)` for local helper modules.

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/src/cint1e.c:7`
- `/home/chemtech/workspace/cintx/libcint-master/src/cint1e.c:9`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_int2e.py:3`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_rys_roots.py:2`
- `/home/chemtech/workspace/cintx/libcint-master/CMakeLists.txt:39`

## Error Handling

**Patterns:**
- Performance-critical C kernels often return status flags (`0`/non-zero) rather than raising rich error objects.
- Internal invariants are protected with `assert(...)` in lower-level numeric routines.
- For unsupported paths or serious numeric failures, code emits an error message to `stderr` and may `exit(...)`.

**Error Types:**
- Return-based handling: many integrator branches return `0` to indicate "integral not available / screened / failed to converge".
- Fail-fast handling: unimplemented drivers and hard failures call `exit(1)` or `exit(err)` after logging.
- Python tests treat failures via `assert` or explicit `"* FAIL"` print markers.

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/src/cint1e.c:67`
- `/home/chemtech/workspace/cintx/libcint-master/src/g2e.c:4447`
- `/home/chemtech/workspace/cintx/libcint-master/src/rys_roots.c:115`
- `/home/chemtech/workspace/cintx/libcint-master/src/cint3c1e.c:453`
- `/home/chemtech/workspace/cintx/libcint-master/src/g1e.c:78`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_rys_roots.py:119`

## Logging

**Framework:**
- C code uses direct `fprintf(stderr, ...)` output for diagnostics.
- Python tests use `print(...)` for pass/fail and numerical diffs.
- The Rust entrypoint currently uses `println!`.

**Patterns:**
- Logging is event-driven (numerical failure, unsupported path), not structured.
- There is no centralized logger abstraction or levels API in this snapshot.
- CI relies on script stdout/stderr output from test scripts.

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/src/g2e.c:4448`
- `/home/chemtech/workspace/cintx/libcint-master/src/rys_roots.c:116`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_cint.py:167`
- `/home/chemtech/workspace/cintx/src/main.rs:2`

## Comments

**When to Comment:**
- C files use block headers describing module purpose and domain context.
- Inline comments explain numerical or memory-layout assumptions near critical loops/macros.
- Tests include inline notes for precision caveats and skipped stress cases.

**JSDoc/TSDoc:**
- Not applicable in this codebase (no JS/TS sources found).
- C headers rely on short C comments adjacent to declarations instead of doc generators.

**TODO Comments:**
- TODO/FIXME comments exist without owner tags and without issue IDs.
- TODOs are used for unimplemented transformations or known convergence/test gaps.

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/src/cint1e.c:36`
- `/home/chemtech/workspace/cintx/libcint-master/src/cint1e.c:91`
- `/home/chemtech/workspace/cintx/libcint-master/src/cint3c1e.c:449`
- `/home/chemtech/workspace/cintx/libcint-master/src/eigh.c:621`
- `/home/chemtech/workspace/cintx/libcint-master/src/rys_roots.h:42`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_rys_wheeler.py:103`

## Function Design

**Size:**
- Core numerical drivers are often large and loop-heavy (particularly integral kernels).
- Helper/static functions are used for decomposition in support utilities, but hot-path functions remain substantial.

**Parameters:**
- Public C APIs pass raw pointer buffers plus explicit dimension/metadata arguments (`out`, `dims`, `shls`, `atm`, `bas`, `env`, `opt`, `cache`).
- Python test helpers similarly pass explicit arrays/pointers and thresholds.

**Return Values:**
- Many C functions return status integers/flags (`FINT`/`int`) or cache sizes (`CACHE_SIZE_T`), with `out == NULL` used as a query path.
- Python helpers return booleans/numeric comparisons or print failures while continuing batch checks.

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/include/cint_funcs.h:14`
- `/home/chemtech/workspace/cintx/libcint-master/src/cint1e.c:188`
- `/home/chemtech/workspace/cintx/libcint-master/src/cint1e.c:191`
- `/home/chemtech/workspace/cintx/libcint-master/src/g2e.c:4425`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_int2e.py:53`

## Module Design

**Exports:**
- `include/cint_funcs.h` acts as the primary exported integral API surface.
- C sources are grouped by integral domain (`cint1e`, `cint2e`, `cint3c*`, `rys_*`, transform files).
- Compatibility wrappers for old interface naming are generated through macros in shared headers.

**Barrel Files:**
- No TypeScript-style barrel pattern exists.
- CMake acts as the central module manifest via `cintSrc` and feature-gated source additions.

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/include/cint_funcs.h:19`
- `/home/chemtech/workspace/cintx/libcint-master/src/misc.h:35`
- `/home/chemtech/workspace/cintx/libcint-master/CMakeLists.txt:66`
- `/home/chemtech/workspace/cintx/libcint-master/CMakeLists.txt:88`

*Convention analysis: 2026-03-14*
*Update when patterns change*
