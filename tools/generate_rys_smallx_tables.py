#!/usr/bin/env python3
"""Extract libcint's nroots=1..12 small-x Rys tables into Rust source."""

import re
from pathlib import Path


SOURCE = Path("libcint-master/src/roots_for_x0.dat")
OUTPUT = Path("crates/cintx-cubecl/src/math/rys_smallx_data.rs")
NROOTS = 12
NVALUES = NROOTS * (NROOTS + 1) // 2


def extract(name: str, source: str) -> list[str]:
    match = re.search(rf"static double {name}\[\] = \{{(.*?)\}};", source, re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    values = re.findall(r"[-+]?\d+\.\d+e[-+]?\d+", match.group(1), re.I)
    if len(values) < NVALUES:
        raise SystemExit(f"{name}: expected at least {NVALUES}, found {len(values)}")
    return values[:NVALUES]


def render(name: str, values: list[str]) -> str:
    lines = [f"pub const {name}: [f64; {NVALUES}] = ["]
    lines.extend(f"    {value}_f64," for value in values)
    lines.append("];\n")
    return "\n".join(lines)


def main():
    source = SOURCE.read_text(encoding="utf-8")
    content = (
        "//! Generated from libcint `roots_for_x0.dat`; do not edit.\n"
        "//! Covers the exact global `x <= 3e-7` branch for nroots 1..=12.\n\n"
    )
    for name in ["POLY_SMALLX_R0", "POLY_SMALLX_R1", "POLY_SMALLX_W0", "POLY_SMALLX_W1"]:
        content += render(name, extract(name, source))
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
