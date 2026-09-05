# Vendored GTH-MOLOPT basis data — provenance and terms

`BASIS_MOLOPT` in this directory is a **verbatim, unmodified** copy of the
GTH-MOLOPT basis-set library distributed with
[CP2K](https://github.com/cp2k/cp2k). It is a data input to `cintx-basis`,
not source code, and is redistributed here so that `cintx`'s GTH fixtures are
reproducible without a network fetch.

## License scope — this directory only

The `cintx` repository as a whole is MIT-licensed (see `LICENSE` at the
repository root). **`BASIS_MOLOPT` and `COPYING` in this directory are the one
exception**: CP2K is distributed under the GNU General Public License,
version 2 (GPLv2), and its data files carry the same license by default —
`BASIS_MOLOPT` has no separate, more permissive license notice of its own.
`COPYING` in this directory is a verbatim copy of the GPLv2 text, included per
the license's own redistribution requirement.

This scoping mirrors `../README.md`, which documents the CC-BY-4.0 terms that
apply only to the def2 files in the parent directory. Nothing outside
`crates/cintx-basis/data/gth/` is affected by either license.

`BASIS_MOLOPT` is only compiled into `cintx-basis` when the crate's `gth`
Cargo feature is enabled (off by default) — `cargo build -p cintx-basis`
without `--features gth` never embeds this GPLv2 data, so a consumer who
doesn't opt in stays on an MIT-only artifact. Enable it with:

```toml
cintx-basis = { path = "...", features = ["gth"] }
```

## File

| File | Upstream source | Role | Elements |
|---|---|---|---|
| `BASIS_MOLOPT` | `cp2k/cp2k`, `data/BASIS_MOLOPT` | orbital (multiple named basis families) | varies by named basis, see below |

The file is a multi-basis library: it carries every GTH-MOLOPT family CP2K
publishes (`SZV-MOLOPT-GTH`, `DZVP-MOLOPT-GTH`, `TZVP-MOLOPT-GTH`,
`TZV2P-MOLOPT-GTH`, `TZV2PX-MOLOPT-GTH`, the `-SR-GTH` short-range variants,
and further per-element entries added upstream over time), each keyed by name
on its element header line. `cintx-basis::catalog::GthBasis` currently parses
out and exposes exactly two of these by name:

- `DZVP-MOLOPT-SR-GTH` — short-range double-zeta-valence-plus-polarization,
  71 elements (H through Zr, and most of the remaining periodic table up to
  Rn; see the `catalog_parses_every_embedded_gth_table` test for the exact
  list).
- `TZVP-MOLOPT-GTH` — full-range triple-zeta-valence-plus-polarization, the 9
  elements from the original VandeVondele & Hutter paper (H, C, N, O, F, Si,
  P, S, Cl).

There is **no published `TZVP-MOLOPT-SR-GTH`** (short-range triple-zeta) —
CP2K's own MOLOPT library only ships short-range variants at SZV and DZVP
quality. `GthBasis` does not expose a name that does not exist upstream.

## Provenance

- **Source**: https://github.com/cp2k/cp2k/blob/master/data/BASIS_MOLOPT
- **Retrieved**: 2026-09-05, via
  `https://raw.githubusercontent.com/cp2k/cp2k/master/data/BASIS_MOLOPT`
  (`master` branch at time of retrieval).
- **Modifications**: none. The file is byte-for-byte as fetched, including
  its own header comment block, which documents the basis-set format and
  cites the source paper.

## Citation

Work that uses these basis sets should cite:

- **GTH-MOLOPT basis sets**: J. VandeVondele and J. Hutter,
  *Gaussian basis sets for accurate calculations on molecular systems in gas
  and condensed phases*, J. Chem. Phys. **127**, 114105 (2007).
  DOI: `10.1063/1.2770708`
- **CP2K**: T. D. Kühne et al., *CP2K: An electronic structure and molecular
  dynamics software package*, J. Chem. Phys. **152**, 194103 (2020).
  DOI: `10.1063/5.0007045`

## Scope: basis data only, no pseudopotential integrals

`GTH-MOLOPT` basis sets are designed to pair with GTH-type pseudopotentials
(a separable local+nonlocal form, unrelated to the semi-local ECP formalism
`cintx-core::ecp` implements for def2-ECP). **cintx does not implement GTH
pseudopotential integrals.** `GthBasis` exposes only the parsed orbital-basis
primitive/contraction data — the same `BasisTable` shape `StandardBasis`
returns — and is deliberately **not** wired into `Molecule`/`to_basis_set`'s
automatic core-electron or ECP-shell logic (which remains def2-only). A
caller using these shells for anything beyond overlap/kinetic-type integrals
on light elements is responsible for supplying and applying the matching GTH
pseudopotential themselves; cintx will not silently substitute def2's ECP or
a bare point-charge nucleus in its place.

## Adding another named basis from this file

The `BASIS_MOLOPT` file already contains other named families
(`SZV-MOLOPT-GTH`, `TZV2P-MOLOPT-GTH`, ...); no re-fetch is needed to expose
one:

1. Confirm the exact name string on the element header line (case as
   upstream writes it).
2. Add a `GthBasis` variant and register it — the `-q`N suffix is a per-element
   pseudopotential-valence tag, not part of the canonical name.
3. Add a coverage/composition test mirroring the ones already in
   `catalog.rs`.
