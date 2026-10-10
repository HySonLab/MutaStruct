from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_TEMPLATE_PATH = Path(__file__).with_name("ideal_residue_templates.json")


@lru_cache(maxsize=1)
def load_residue_templates() -> dict[str, Any]:
    with _TEMPLATE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atom_order(amino_acid: str, include_oxt: bool = False) -> list[str]:
    template = load_residue_templates()[amino_acid]
    names = list(template["atom_names"])
    if not include_oxt:
        names = [name for name in names if name != "OXT"]
    return names


def residue_bonds(amino_acid: str, include_oxt: bool = False) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for atom_a, atom_b, _order in load_residue_templates()[amino_acid]["bonds"]:
        if not include_oxt and (atom_a == "OXT" or atom_b == "OXT"):
            continue
        result.append((atom_a, atom_b))
    return result
