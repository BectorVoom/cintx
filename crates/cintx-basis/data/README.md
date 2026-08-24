# Vendored basis-set data — provenance and terms

The three `.nwchem` files in this directory are **verbatim, unmodified**
exports from the [Basis Set Exchange](https://www.basissetexchange.org)
(BSE). They are data inputs to `cintx-basis`, not source code, and are
redistributed here so that `cintx`'s def2 fixtures and oracle parity gates are
reproducible without a network fetch.

## Files

| File | BSE basis set | Role | Elements |
|---|---|---|---|
| `def2-svp.nwchem` | `def2-SVP` | orbital | H–Rn |
| `def2-tzvp.nwchem` | `def2-TZVP` | orbital | H–Rn |
| `def2-ecp.nwchem` | `def2-ECP` | ECP | Rb–Rn (Z ≥ 37) |

## Provenance

- **Source**: https://www.basissetexchange.org
- **BSE software version**: `0.12` (recorded in each file's header block)
- **Basis-set data version**: `1` — "Data from Turbomole 7.3" (recorded in
  each file's header block)
- **Export format**: NWChem, spherical (`BASIS "ao basis" SPHERICAL PRINT`)
- **Retrieved**: 2026-08-23, during the Phase 32 `cintx-basis` work
- **Modifications**: none. The files are byte-for-byte as exported; the header
  comment block that BSE emits is retained deliberately so the provenance
  travels with the data.

Direct download URLs of the form
`https://www.basissetexchange.org/api/basis/def2-svp/format/nwchem/` reproduce
these files for the same BSE version.

## Terms

Basis Set Exchange data is published under the
**Creative Commons Attribution 4.0 International licence (CC-BY-4.0)**, which
permits redistribution — including inside a public library — provided
attribution is given. This file is that attribution; the BSE header block
inside each data file carries it as well, which is why the headers are not
stripped.

The CC-BY-4.0 terms apply to the contents of *this directory only*. The rest
of `cintx` remains under the repository's MIT licence (see `LICENSE` at the
repository root); the data files are not derived from, and do not affect, that
licence.

## Citation

Work that uses these basis sets should cite both the data and the exchange:

- **def2 basis sets**: F. Weigend and R. Ahlrichs,
  *Balanced basis sets of split valence, triple zeta valence and quadruple
  zeta valence quality for H to Rn: Design and assessment of accuracy*,
  Phys. Chem. Chem. Phys. **7**, 3297 (2005). DOI: `10.1039/B508541A`
- **def2 ECPs**: D. Andrae, U. Häußermann, M. Dolg, H. Stoll, H. Preuß,
  Theor. Chim. Acta **77**, 123 (1990), and the subsequent Stuttgart/Cologne
  ECP series referenced by the def2 papers.
- **Basis Set Exchange**: B. P. Pritchard, D. Altarawy, B. Didier,
  T. D. Gibson, T. L. Windus, *A New Basis Set Exchange: An Open, Up-to-date
  Resource for the Molecular Sciences Community*, J. Chem. Inf. Model. **59**,
  4814 (2019). DOI: `10.1021/acs.jcim.9b00725`

## Adding a basis set

Keep the rules that make this directory auditable:

1. Export from BSE in the **NWChem** format, spherical, unmodified.
2. Do **not** strip the BSE header block.
3. Add a row to the table above and record the retrieval date.
4. Register the file in `crates/cintx-basis/src/catalog.rs`; the parser in
   `format.rs` reads this exact dialect.
