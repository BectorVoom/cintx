# Testing Patterns

**Analysis Date:** 2026-03-14

## Test Framework

**Runner:**
- Primary test execution is Python script based (`python test_*.py`) in `libcint-master/testsuite`.
- CTest integration exists when configured with `-DENABLE_TEST=1`; CMake registers test script invocations.
- CI (GitHub Actions) compiles the C library and runs an explicit list of Python test scripts on Ubuntu.

**Assertion Library:**
- Native Python `assert` is used for threshold checks.
- Numeric comparisons use `numpy` (`numpy.allclose`, `numpy.linalg.norm`, absolute max checks).
- Some suites also emit manual `"pass"` / `"* FAIL"` text.

**Run Commands:**
```bash
# Build + run ctest path from project docs
cd libcint-master
mkdir build && cd build
cmake -DENABLE_EXAMPLE=1 -DENABLE_TEST=1 -DQUICK_TEST=1 ..
make
make test ARGS=-V

# CI-style compile + scripted test run
cd libcint-master
cmake -DWITH_CINT2_INTERFACE=1 -DWITH_RANGE_COULOMB=1 -DWITH_COULOMB_ERF=1 -DWITH_F12=1 -DWITH_4C1E=1 -Bbuild -DKEEP_GOING=1 .
cmake --build build
pip install numpy mpmath pyscf
cd testsuite
python test_rys_roots.py
python test_cint.py --quick
python test_3c2e.py --quick
python test_int2e.py --quick

# Direct single-file run (common local pattern)
cd libcint-master/testsuite
python test_int2e.py --quick
```

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/README.rst:174`
- `/home/chemtech/workspace/cintx/libcint-master/CMakeLists.txt:207`
- `/home/chemtech/workspace/cintx/libcint-master/.github/workflows/ci.yml:21`
- `/home/chemtech/workspace/cintx/libcint-master/.github/workflows/ci.yml:33`

## Test File Organization

**Location:**
- Test files are centralized in `libcint-master/testsuite/` (single directory, no nested test package tree).
- The top-level Rust crate (`src/main.rs`) currently has no unit/integration test files.

**Naming:**
- Python test files follow `test_*.py` naming (for example `test_cint.py`, `test_int2e.py`, `test_rys_roots.py`).
- Individual test functions also use `test_*` naming, which remains compatible with pytest discovery even when run as scripts.

**Structure:**
```text
libcint-master/
  testsuite/
    test_cint.py
    test_3c2e.py
    test_int1e.py
    test_int2e.py
    test_rys_roots.py
    test_cart2sph.py
src/
  main.rs  # no Rust tests yet
```

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_cint.py:1`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_int2e.py:1`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_rys_roots.py:1`
- `/home/chemtech/workspace/cintx/src/main.rs:1`

## Test Structure

**Suite Organization:**
```python
# Typical pattern in testsuite scripts
def run(intor, comp=1, suffix='_sph', thr=1e-7):
    failed = False
    ...
    if numpy.linalg.norm(ref-buf) > thr:
        failed = True
    ...
    print('pass' if not failed else 'failed')

if __name__ == "__main__":
    run('int2e', thr=1e-12)
    if '--quick' in sys.argv:
        exit()
    run("int2e_spsp1", suffix='_spinor')
```

**Patterns:**
- Heavy numeric loops over shell indices compare computed values to references.
- Tolerance-driven validation (`thr`, decimal place checks, max-error checks).
- `--quick` gates are used to skip expensive test subsets.
- Some scripts use deterministic seeds (`np.random.seed(...)`) for reproducibility.

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_int2e.py:53`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_int2e.py:99`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_cint.py:534`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_3c2e.py:308`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_cart2sph.py:21`

## Mocking

**Framework:**
- No mocking framework is used (`unittest.mock`, `pytest` fixtures/mocks, monkeypatch patterns not present).

**Patterns:**
- Tests use real compiled library calls through `ctypes` / `numpy.ctypeslib.load_library`.
- External scientific references (for example PySCF) are used as numerical baselines rather than mocks.

**What to Mock:**
- Current codebase does not define a mocking convention for this layer.

**What NOT to Mock:**
- Existing tests intentionally exercise real numerical kernels and compare end-to-end integral outputs.

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_cint.py:16`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_int2e.py:8`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_int2e.py:11`

## Fixtures and Factories

**Test Data:**
- Most fixtures are constructed inline inside each test script (arrays like `atm`, `bas`, `env` and basis metadata constants).
- Reusable helper functions act as lightweight factories in-file (`make_cintopt`, `cint_call`, numeric helper closures).
- No shared `tests/fixtures` or `tests/factories` directory exists.

**Location:**
- Fixtures/helpers live inside each `libcint-master/testsuite/test_*.py`.
- Molecular reference data is often embedded directly as multi-line basis strings in the script.

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_cint.py:46`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_int2e.py:13`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_int2e.py:39`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_rys_roots.py:15`

## Coverage

**Requirements:**
- No explicit line/branch coverage target is defined in repository docs, CMake config, or CI workflow.
- Current strategy emphasizes numeric correctness checks and regression scripts rather than coverage thresholds.

**Configuration:**
- No coverage tool configuration (`gcov`, `lcov`, `pytest-cov`, `codecov`) is present in checked-in build/test configs.

**View Coverage:**
```bash
# No built-in coverage command documented in current repository state.
# Add toolchain support explicitly before relying on coverage metrics.
```

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/README.rst:174`
- `/home/chemtech/workspace/cintx/libcint-master/CMakeLists.txt:207`
- `/home/chemtech/workspace/cintx/libcint-master/.github/workflows/ci.yml:24`

## Test Types

**Unit Tests:**
- Algorithm-focused tests validate specific numerical kernels and root solvers with strict tolerances (`test_rys_roots.py`, `test_c2s.py`, `test_cart2sph.py`).

**Integration Tests:**
- Integral API tests compare libcint outputs against reference calculations (often via PySCF) across many shells/components (`test_cint.py`, `test_int1e.py`, `test_int2e.py`, `test_3c2e.py`).

**E2E Tests:**
- No UI/service-level E2E framework is present.
- The highest-level flow here is compile library then run Python integration scripts in CI.

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_rys_roots.py:87`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_c2s.py:163`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_cint.py:152`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_int2e.py:53`
- `/home/chemtech/workspace/cintx/libcint-master/.github/workflows/ci.yml:29`

## Common Patterns

**Async Testing:**
- Not used; tests are synchronous numerical scripts.

**Error Testing:**
```python
# Numeric threshold assert
assert max_r_error < 1e-3
assert max_w_error < 1e-7

# Decomposition identity check
if abs(dat_sr + dat_lr - dat).max() > 1e-8:
    print('FAIL', i, j, k, l, abs(dat_sr + dat_lr - dat).max())
```

**Snapshot Testing:**
- Not used in this codebase.

**Evidence:**
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_rys_roots.py:119`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_rys_roots.py:151`
- `/home/chemtech/workspace/cintx/libcint-master/testsuite/test_cint.py:450`

*Testing analysis: 2026-03-14*
*Update when test patterns change*
