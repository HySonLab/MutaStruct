from __future__ import annotations

from pathlib import Path

# Defaults are relative to this checkout, with optional environment overrides.
# No unpublished data are included at these paths.
import os

MODEL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODEL_ROOT.parent
DATASET_ROOT = Path(os.environ.get("MUTASTRUCT_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
DEFAULT_MANIFEST = Path(os.environ.get(
    "MUTASTRUCT_MANIFEST", DATASET_ROOT / "tier_a_pairs_with_splits.parquet"
)).expanduser()
DEFAULT_STRUCTURES_DIR = Path(os.environ.get(
    "MUTASTRUCT_STRUCTURES", DATASET_ROOT / "structures"
)).expanduser()
DEFAULT_CACHE_DIR = Path(os.environ.get(
    "MUTASTRUCT_CACHE", PROJECT_ROOT / "cache" / "graph_v1_r6_n32"
)).expanduser()

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i for i, aa in enumerate(AA_ORDER)}
UNK_AA_ID = len(AA_ORDER)

ELEMENTS = ["C", "N", "O", "S", "P", "SE", "OTHER"]
ELEMENT_TO_ID = {element: i for i, element in enumerate(ELEMENTS)}

# All heavy-atom names that occur in standard amino acids. An unknown bucket is
# added at runtime for modified/rare coordinate records outside the mutation site.
ATOM_NAMES = [
    "N", "CA", "C", "O", "OXT", "CB", "CG", "CG1", "CG2", "CD", "CD1",
    "CD2", "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2", "ND1",
    "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "OD1", "OD2", "OE1",
    "OE2", "OG", "OG1", "OH", "SD", "SG",
]
ATOM_TO_ID = {name: i for i, name in enumerate(ATOM_NAMES)}
UNK_ATOM_ID = len(ATOM_NAMES)

BACKBONE_ATOMS = {"N", "CA", "C", "O"}

# Canonical modified-residue mappings used only to interpret deposited mmCIF
# chemistry. The benchmark sequence remains the source of truth for residue IDs.
AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "SEP": "S", "TPO": "T", "PTR": "Y",
    "CSO": "C", "CSD": "C", "NLE": "L", "HYP": "P", "FME": "M",
}

VDW_RADII = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "SE": 1.90,
    "OTHER": 1.70,
}
