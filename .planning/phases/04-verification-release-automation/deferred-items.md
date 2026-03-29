# Deferred Items

- Task 3 oracle profile gate surfaced pre-existing parity drift in optional profiles when running `oracle-compare` with `base,with-f12,with-4c1e,with-f12+with-4c1e`: `with-f12` (10 mismatches), `with-4c1e` (2 mismatches), `with-f12+with-4c1e` (12 mismatches). Gate remains fail-closed by design; underlying parity deltas were not introduced by this plan's xtask command-surface work.
