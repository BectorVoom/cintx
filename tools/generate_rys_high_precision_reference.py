#!/usr/bin/env python3
"""Generate the Phase-4, host-only Rys reference with mpmath.

This never participates in a shipped compute path.  It constructs the Gaussian
quadrature rule from the defining Boys moments at 100 decimal digits, using a
Stieltjes recurrence and a symmetric eigensolve.  The emitted JSON stores
decimal strings so the reference itself does not lose precision on serialization.
"""

import argparse
import json
from pathlib import Path

import mpmath as mp


mp.mp.dps = 100
X_VALUES = ["0", "0.0000003", "1", "10", "11", "15", "18", "22", "33", "40", "95"]


def boys(moment: int, x: mp.mpf) -> mp.mpf:
    if x == 0:
        return mp.mpf(1) / (2 * moment + 1)
    return mp.gammainc(moment + mp.mpf("0.5"), 0, x) / (2 * x ** (moment + mp.mpf("0.5")))


def inner(left, right, moments):
    return sum(left[i] * right[j] * moments[i + j] for i in range(len(left)) for j in range(len(right)))


def multiply_y(poly):
    return [mp.mpf(0)] + poly


def high_precision_rule(nroots: int, x: mp.mpf):
    # Measure: 1/2 * y^-1/2 exp(-x y) dy on y in [0, 1].  Its moments are F_m(x).
    moments = [boys(i, x) for i in range(2 * nroots + 1)]
    orthogonal = []
    for degree in range(nroots):
        poly = [mp.mpf(0)] * degree + [mp.mpf(1)]
        for basis in orthogonal:
            projection = inner(poly, basis, moments)
            for i, value in enumerate(basis):
                poly[i] -= projection * value
        norm = mp.sqrt(inner(poly, poly, moments))
        orthogonal.append([value / norm for value in poly])

    jacobi = mp.matrix(nroots)
    for row in range(nroots):
        for col in range(nroots):
            jacobi[row, col] = inner(multiply_y(orthogonal[row]), orthogonal[col], moments)
    nodes, vectors = mp.eigsy(jacobi)
    roots = [nodes[i] / (1 - nodes[i]) for i in range(nroots)]
    weights = [moments[0] * vectors[0, i] ** 2 for i in range(nroots)]
    return roots, weights, moments[: 2 * nroots]


def decimal(value: mp.mpf) -> str:
    return mp.nstr(value, 90, strip_zeros=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/rys_high_precision_reference.json")
    args = parser.parse_args()

    cases = []
    for nroots in range(1, 13):
        for x_text in X_VALUES:
            x = mp.mpf(x_text)
            roots, weights, boys_values = high_precision_rule(nroots, x)
            cases.append({
                "nroots": nroots,
                "x": decimal(x),
                "roots": [decimal(value) for value in roots],
                "weights": [decimal(value) for value in weights],
                "boys": [decimal(value) for value in boys_values],
            })

    payload = {
        "generator": "tools/generate_rys_high_precision_reference.py",
        "precision_decimal_digits": 100,
        "definition": "Gaussian rule for Boys moments F_m(x) over y=t^2, with u=y/(1-y)",
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} reference cases to {output}")


if __name__ == "__main__":
    main()
