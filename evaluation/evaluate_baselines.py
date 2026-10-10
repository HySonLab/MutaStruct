#!/usr/bin/env python3
"""Evaluate ESMFold, PreMut, FoldX, and Rosetta predictions on MutaStruct.

The script uses exactly the same five primary metrics as the benchmark evaluator:

1. Global C-alpha RMSD (lower is better)
2. All-atom lDDT (higher is better)
3. Local C-alpha RMSD within the WT-defined 8 A mutation neighborhood
4. Mutation-site side-chain heavy-atom RMSD
5. Mutation-response RMSD within the WT-defined 8 A neighborhood

Fair-comparison policy
----------------------
The benchmark cached graph for each Tier-A test pair is used as the common evaluation
universe. Therefore, every method is scored on the same model-represented and
experimentally observed residues/atoms. Prediction PDB residues are mapped back to
target sequence positions before scoring.

Default prediction layouts
--------------------------
ESMFold:
  .../ESMFold/results/facebook_esmfold_v1/predictions/**/SEQ_<sha256>.pdb

PreMut:
  .../PreMut/mutastruct_test_predictions/pair_to_prediction.csv
  or pair_predictions/<pair_id>.pdb

FoldX:
  <project-root>/FoldX/results/predictions/
  <REF>_E<ENTITY>_<CHAIN>_<MUTATION>__<jobhash>/predicted_mutant_A.pdb

Rosetta:
  <project-root>/baseline_comparison/Rosetta/results/predictions/
  <REF>_E<ENTITY>_<CHAIN>_<MUTATION>__<jobhash>/predicted_mutant_A.pdb

FoldX and Rosetta are indexed and scored independently even though their job-folder
naming convention is the same.

Example
-------
Smoke test:
  python evaluate_baselines_main5.py --max-cases 32

Full test set:
  python evaluate_baselines_main5.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "predictions"
GT_ROOT = PROJECT_ROOT / "geometric_transformer"

if str(GT_ROOT) not in sys.path:
    sys.path.insert(0, str(GT_ROOT))

from common.constants import (  # noqa: E402
    AA3_TO_AA1,
    AA_ORDER,
    ATOM_NAMES,
    DEFAULT_CACHE_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_STRUCTURES_DIR,
)
from common.data import PairGraphDataset  # noqa: E402
from common.geometry import kabsch_mobile_to_reference  # noqa: E402
from common.graph import GraphBuildConfig  # noqa: E402


DEFAULT_ESMFOLD_ROOT = (
    BASELINE_ROOT
    / "ESMFold"
    / "results"
    / "facebook_esmfold_v1"
    / "predictions"
)
DEFAULT_PREMUT_ROOT = BASELINE_ROOT / "PreMut" / "mutastruct_test_predictions"
DEFAULT_FOLDX_ROOT = BASELINE_ROOT / "FoldX" / "results" / "predictions"
DEFAULT_ROSETTA_ROOT = (
    BASELINE_ROOT / "Rosetta" / "results" / "predictions"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "external_baselines"

BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT"}

SYMMETRY_SWAPS: dict[str, dict[str, str]] = {
    "D": {"OD1": "OD2", "OD2": "OD1"},
    "E": {"OE1": "OE2", "OE2": "OE1"},
    "R": {"NH1": "NH2", "NH2": "NH1"},
    "L": {"CD1": "CD2", "CD2": "CD1"},
    "V": {"CG1": "CG2", "CG2": "CG1"},
    "F": {
        "CD1": "CD2",
        "CD2": "CD1",
        "CE1": "CE2",
        "CE2": "CE1",
    },
    "Y": {
        "CD1": "CD2",
        "CD2": "CD1",
        "CE1": "CE2",
        "CE2": "CE1",
    },
}

PRIMARY_METRICS = [
    "ca_rmsd",
    "lddt",
    "local_ca_rmsd_8a",
    "mutation_sidechain_rmsd",
    "mutation_response_rmsd_8a",
]

FOLDX_JOB_RE = re.compile(
    r"^(?P<pdb>[A-Za-z0-9]{4})_E(?P<entity>[^_]+)_"
    r"(?P<chain>[^_]+)_(?P<mutation>[A-Za-z]\d+[A-Za-z])__(?P<suffix>.+)$"
)
ESMFOLD_FILE_RE = re.compile(r"^SEQ_([0-9a-fA-F]{64})\.pdb$")


@dataclass
class AtomCandidate:
    occupancy: float
    alt_priority: int
    coord: np.ndarray


@dataclass
class ResidueRecord:
    order_index: int
    chain_id: str
    resseq: int
    insertion_code: str
    residue_name: str
    aa: str
    atoms: dict[str, np.ndarray]


@dataclass
class ParsedPrediction:
    path: Path
    chain_id: str
    residue_to_position: dict[int, int]
    atoms_by_position: dict[tuple[int, str], np.ndarray]
    mapping_method: str
    mapping_identity: float
    mapped_residue_count: int
    prediction_residue_count: int


@dataclass
class MetricContext:
    target: np.ndarray
    initial: np.ndarray
    target_mask: np.ndarray
    is_ca: np.ndarray
    distance_to_mutation: np.ndarray
    node_residue: np.ndarray
    seq_ids: np.ndarray
    atom_names: np.ndarray
    mutation_mask: np.ndarray
    mt_aa_by_node: np.ndarray
    full_lddt_i: np.ndarray
    full_lddt_j: np.ndarray


@dataclass
class FoldXIndex:
    pair_map: dict[str, Path]
    exact: dict[tuple[str, str, str, str], list[Path]]
    without_chain: dict[tuple[str, str, str], list[Path]]
    without_entity: dict[tuple[str, str, str], list[Path]]
    by_pdb_mutation: dict[tuple[str, str], list[Path]]
    indexed_files: int
    mapping_csv: str


# ---------------------------------------------------------------------------
# CLI and generic helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--structures-dir", type=Path, default=DEFAULT_STRUCTURES_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)

    parser.add_argument("--esmfold-root", type=Path, default=DEFAULT_ESMFOLD_ROOT)
    parser.add_argument("--premut-root", type=Path, default=DEFAULT_PREMUT_ROOT)

    parser.add_argument("--foldx-root", type=Path, default=DEFAULT_FOLDX_ROOT)
    parser.add_argument("--foldx-map-csv", type=Path, default=None)
    parser.add_argument("--foldx-filename", default="predicted_mutant_A.pdb")

    parser.add_argument("--rosetta-root", type=Path, default=DEFAULT_ROSETTA_ROOT)
    parser.add_argument("--rosetta-map-csv", type=Path, default=None)
    parser.add_argument("--rosetta-filename", default="predicted_mutant_A.pdb")

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--pair-class", default="all")
    parser.add_argument("--local-radius", type=float, default=8.0)
    parser.add_argument("--lddt-radius", type=float, default=15.0)

    parser.add_argument("--graph-radius", type=float, default=6.0)
    parser.add_argument("--graph-max-neighbors", type=int, default=32)
    parser.add_argument("--cache-schema-version", type=int, default=1)

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["esmfold", "premut", "foldx", "rosetta"],
        default=["esmfold", "premut", "foldx", "rosetta"],
    )
    parser.add_argument(
        "--strict-foldx-mapping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail instead of choosing among non-identical ambiguous FoldX/Rosetta "
            "files. Applied independently to both mutation-model roots."
        ),
    )

    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>", ".", "?"} else text


def normalize_entity(value: Any) -> str:
    text = clean(value)
    if not text:
        return ""
    try:
        x = float(text)
        if x.is_integer():
            return str(int(x))
    except ValueError:
        pass
    return text


def target_sequence_from_row(row: pd.Series) -> str:
    sequence = "".join(clean(row.get("target_sequence")).split()).upper()
    if not sequence:
        raise ValueError("target_sequence is missing")
    return sequence


def mutation_from_row(row: pd.Series) -> str:
    mutation = clean(row.get("mutation")).upper()
    if mutation:
        return mutation

    ref = clean(row.get("reference_aa")).upper()
    target = clean(row.get("target_aa")).upper()
    index = clean(row.get("mutation_index_0"))
    if not (ref and target and index):
        raise ValueError("Cannot construct mutation label")
    return f"{ref}{int(float(index)) + 1}{target}"


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def choose_equivalent_or_raise(
    candidates: Iterable[Path],
    strict: bool,
    description: str,
) -> tuple[Path, int]:
    unique = sorted({Path(path).resolve() for path in candidates})
    if not unique:
        raise FileNotFoundError(description)
    if len(unique) == 1:
        return unique[0], 1

    hashes = {file_sha256(path) for path in unique}
    if len(hashes) == 1:
        return unique[0], len(unique)

    if strict:
        preview = "\n  ".join(str(path) for path in unique[:20])
        raise ValueError(
            f"Ambiguous non-identical predictions for {description}:\n  {preview}"
        )
    return unique[0], len(unique)


# ---------------------------------------------------------------------------
# Prediction discovery
# ---------------------------------------------------------------------------


def index_esmfold_predictions(root: Path) -> dict[str, list[Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"ESMFold prediction root not found: {root}")

    index: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("SEQ_*.pdb"):
        match = ESMFOLD_FILE_RE.match(path.name)
        if match:
            index[match.group(1).lower()].append(path.resolve())
    return dict(index)


def resolve_esmfold_prediction(
    row: pd.Series,
    index: dict[str, list[Path]],
) -> tuple[Path, str, int]:
    target_sequence = target_sequence_from_row(row)
    sequence_hash = hashlib.sha256(target_sequence.encode("utf-8")).hexdigest()
    path, duplicate_count = choose_equivalent_or_raise(
        index.get(sequence_hash, []),
        strict=True,
        description=f"ESMFold target sequence SHA256={sequence_hash}",
    )
    return path, "target_sequence_sha256", duplicate_count


def resolve_csv_path(raw: Any, csv_path: Path, root: Path) -> Path | None:
    text = clean(raw)
    if not text:
        return None
    path = Path(text)
    candidates = [path, csv_path.parent / path, root / path, root / path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def load_pair_mapping_csv(
    csv_path: Path,
    root: Path,
) -> dict[str, Path]:
    frame = pd.read_csv(csv_path, dtype=str)
    pair_columns = ["pair_id", "pair", "sample_id"]
    path_columns = [
        "prediction_path",
        "predicted_path",
        "output_path",
        "pdb_path",
        "path",
    ]

    pair_col = next((x for x in pair_columns if x in frame.columns), None)
    path_col = next((x for x in path_columns if x in frame.columns), None)
    if pair_col is None or path_col is None:
        return {}

    output: dict[str, Path] = {}
    for row in frame.itertuples(index=False):
        values = row._asdict()
        pair_id = clean(values.get(pair_col))
        path = resolve_csv_path(values.get(path_col), csv_path, root)
        if pair_id and path is not None:
            output[pair_id] = path
    return output


def discover_mapping_csv(root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit

    names = [
        "pair_to_prediction.csv",
        "prediction_manifest.csv",
        "jobs_manifest.csv",
        "manifest.csv",
    ]
    search_dirs = [root, root.parent, root.parent.parent]
    for directory in search_dirs:
        for name in names:
            path = directory / name
            if path.is_file():
                return path
    return None


def build_foldx_index(
    root: Path,
    filename: str,
    mapping_csv: Path | None,
) -> FoldXIndex:
    if not root.is_dir():
        raise FileNotFoundError(f"FoldX/Rosetta prediction root not found: {root}")

    pair_map: dict[str, Path] = {}
    csv_used = ""
    found_csv = discover_mapping_csv(root, mapping_csv)
    if found_csv is not None and found_csv.exists():
        pair_map = load_pair_mapping_csv(found_csv, root)
        csv_used = str(found_csv.resolve())

    exact: dict[tuple[str, str, str, str], list[Path]] = defaultdict(list)
    without_chain: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    without_entity: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    by_pdb_mutation: dict[tuple[str, str], list[Path]] = defaultdict(list)

    paths = sorted(root.rglob(filename))
    for path in paths:
        match = FOLDX_JOB_RE.match(path.parent.name)
        if not match:
            continue
        pdb = match.group("pdb").upper()
        entity = normalize_entity(match.group("entity"))
        chain = match.group("chain")
        mutation = match.group("mutation").upper()
        resolved = path.resolve()

        exact[(pdb, entity, chain, mutation)].append(resolved)
        without_chain[(pdb, entity, mutation)].append(resolved)
        without_entity[(pdb, chain, mutation)].append(resolved)
        by_pdb_mutation[(pdb, mutation)].append(resolved)

    return FoldXIndex(
        pair_map=pair_map,
        exact=dict(exact),
        without_chain=dict(without_chain),
        without_entity=dict(without_entity),
        by_pdb_mutation=dict(by_pdb_mutation),
        indexed_files=len(paths),
        mapping_csv=csv_used,
    )


def reference_chain_candidates(row: pd.Series) -> list[str]:
    candidates = [
        clean(row.get("reference_chain_label_asym_id")),
        clean(row.get("reference_chain_auth_asym_id")),
        clean(row.get("reference_selected_label_asym_id")),
        clean(row.get("reference_selected_auth_asym_id")),
    ]
    wild = clean(row.get("wild"))
    if "_" in wild:
        candidates.append(wild.rsplit("_", 1)[-1])

    output: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in output:
            output.append(candidate)
    return output


def resolve_foldx_prediction(
    row: pd.Series,
    index: FoldXIndex,
    strict: bool,
) -> tuple[Path, str, int]:
    pair_id = clean(row.get("pair_id"))
    if pair_id in index.pair_map:
        path = index.pair_map[pair_id]
        if not path.exists():
            raise FileNotFoundError(
                f"FoldX mapping CSV path does not exist for pair {pair_id}: {path}"
            )
        return path.resolve(), "pair_mapping_csv", 1

    pdb = clean(row.get("reference_pdb_id")).upper()
    entity = normalize_entity(row.get("reference_entity_id"))
    mutation = mutation_from_row(row)
    chains = reference_chain_candidates(row)

    for chain in chains:
        candidates = index.exact.get((pdb, entity, chain, mutation), [])
        if candidates:
            path, n = choose_equivalent_or_raise(
                candidates,
                strict=strict,
                description=f"FoldX exact key {(pdb, entity, chain, mutation)}",
            )
            return path, "folder_exact", n

    candidates = index.without_chain.get((pdb, entity, mutation), [])
    if candidates:
        path, n = choose_equivalent_or_raise(
            candidates,
            strict=strict,
            description=f"FoldX key without chain {(pdb, entity, mutation)}",
        )
        return path, "folder_without_chain", n

    for chain in chains:
        candidates = index.without_entity.get((pdb, chain, mutation), [])
        if candidates:
            path, n = choose_equivalent_or_raise(
                candidates,
                strict=strict,
                description=f"FoldX key without entity {(pdb, chain, mutation)}",
            )
            return path, "folder_without_entity", n

    candidates = index.by_pdb_mutation.get((pdb, mutation), [])
    if candidates:
        path, n = choose_equivalent_or_raise(
            candidates,
            strict=strict,
            description=f"FoldX key by PDB+mutation {(pdb, mutation)}",
        )
        return path, "folder_pdb_mutation", n

    raise FileNotFoundError(
        f"No FoldX/Rosetta-style prediction found for pair={pair_id}, "
        f"reference={pdb}, entity={entity}, chains={chains}, mutation={mutation}"
    )


def load_premut_map(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"PreMut output root not found: {root}")

    mapping_csv = root / "pair_to_prediction.csv"
    mapping: dict[str, Path] = {}
    if mapping_csv.exists():
        mapping = load_pair_mapping_csv(mapping_csv, root)

    pair_dir = root / "pair_predictions"
    if pair_dir.is_dir():
        for path in pair_dir.glob("*.pdb"):
            mapping.setdefault(path.stem, path.resolve())
    return mapping


def resolve_premut_prediction(
    row: pd.Series,
    mapping: dict[str, Path],
) -> tuple[Path, str, int]:
    pair_id = clean(row.get("pair_id"))
    path = mapping.get(pair_id)
    if path is None:
        raise FileNotFoundError(f"No PreMut prediction mapped for pair {pair_id}")
    if not path.exists():
        raise FileNotFoundError(f"PreMut prediction path does not exist: {path}")
    return path.resolve(), "pair_id", 1


# ---------------------------------------------------------------------------
# PDB parsing and residue-to-sequence mapping
# ---------------------------------------------------------------------------


def parse_int_field(text: str) -> int | None:
    try:
        return int(text.strip())
    except ValueError:
        return None


def parse_float_field(text: str, default: float) -> float:
    try:
        value = float(text.strip())
        return value if np.isfinite(value) else default
    except ValueError:
        return default


def parse_pdb_chains(path: Path) -> dict[str, list[ResidueRecord]]:
    residue_candidates: dict[
        tuple[str, int, str, str],
        dict[str, list[AtomCandidate]],
    ] = defaultdict(lambda: defaultdict(list))
    residue_order: dict[tuple[str, int, str, str], int] = {}
    order_counter = 0

    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[0:6].strip().upper()
            if record not in {"ATOM", "HETATM"}:
                continue
            if len(line) < 54:
                continue

            atom_name = line[12:16].strip().upper()
            altloc = line[16:17].strip()
            residue_name = line[17:20].strip().upper()
            chain_id = line[21:22].strip() or "_"
            resseq = parse_int_field(line[22:26])
            insertion_code = line[26:27].strip()
            if not atom_name or resseq is None:
                continue
            if atom_name.startswith("H"):
                continue

            x = parse_float_field(line[30:38], np.nan)
            y = parse_float_field(line[38:46], np.nan)
            z = parse_float_field(line[46:54], np.nan)
            coord = np.asarray([x, y, z], dtype=np.float64)
            if not np.isfinite(coord).all():
                continue

            occupancy = parse_float_field(line[54:60], 1.0)
            alt_priority = 2 if altloc == "" else 1 if altloc == "A" else 0
            if altloc not in {"", "A"}:
                continue

            key = (chain_id, resseq, insertion_code, residue_name)
            if key not in residue_order:
                residue_order[key] = order_counter
                order_counter += 1

            residue_candidates[key][atom_name].append(
                AtomCandidate(
                    occupancy=occupancy,
                    alt_priority=alt_priority,
                    coord=coord,
                )
            )

    chains: dict[str, list[ResidueRecord]] = defaultdict(list)
    for key, atoms in residue_candidates.items():
        chain_id, resseq, insertion_code, residue_name = key
        aa = AA3_TO_AA1.get(residue_name, "X")
        selected_atoms: dict[str, np.ndarray] = {}
        for atom_name, candidates in atoms.items():
            candidates.sort(
                key=lambda item: (item.occupancy, item.alt_priority),
                reverse=True,
            )
            selected_atoms[atom_name] = candidates[0].coord

        chains[chain_id].append(
            ResidueRecord(
                order_index=residue_order[key],
                chain_id=chain_id,
                resseq=resseq,
                insertion_code=insertion_code,
                residue_name=residue_name,
                aa=aa,
                atoms=selected_atoms,
            )
        )

    for chain_id in chains:
        chains[chain_id].sort(key=lambda residue: residue.order_index)

    if not chains:
        raise ValueError(f"No protein atoms parsed from prediction PDB: {path}")
    return dict(chains)


def choose_prediction_chain(
    chains: dict[str, list[ResidueRecord]],
    preferred_chain: str | None,
) -> tuple[str, list[ResidueRecord]]:
    preferred = clean(preferred_chain)
    if preferred and preferred in chains:
        return preferred, chains[preferred]
    if "A" in chains:
        return "A", chains["A"]

    ranked = sorted(
        chains.items(),
        key=lambda item: (
            -sum("CA" in residue.atoms for residue in item[1]),
            -len(item[1]),
            item[0],
        ),
    )
    return ranked[0]


def identity_for_mapping(
    residues: list[ResidueRecord],
    mapping: dict[int, int],
    target_sequence: str,
) -> tuple[float, int]:
    matches = 0
    comparable = 0
    for pred_index, target_position in mapping.items():
        aa = residues[pred_index].aa
        if aa == "X" or not (1 <= target_position <= len(target_sequence)):
            continue
        comparable += 1
        matches += aa == target_sequence[target_position - 1]
    identity = matches / comparable if comparable else 0.0
    return float(identity), comparable


def direct_numbering_mapping(
    residues: list[ResidueRecord],
    target_sequence: str,
) -> tuple[dict[int, int], float]:
    mapping: dict[int, int] = {}
    used: set[int] = set()
    for index, residue in enumerate(residues):
        position = residue.resseq
        if 1 <= position <= len(target_sequence) and position not in used:
            mapping[index] = position
            used.add(position)
    identity, _ = identity_for_mapping(residues, mapping, target_sequence)
    return mapping, identity


def sequential_mapping(
    residues: list[ResidueRecord],
    target_sequence: str,
) -> tuple[dict[int, int], float]:
    n = min(len(residues), len(target_sequence))
    mapping = {index: index + 1 for index in range(n)}
    identity, _ = identity_for_mapping(residues, mapping, target_sequence)
    return mapping, identity


def offset_mapping(
    residues: list[ResidueRecord],
    target_sequence: str,
) -> tuple[dict[int, int], float, int]:
    if not residues:
        return {}, 0.0, 0

    min_resseq = min(residue.resseq for residue in residues)
    max_resseq = max(residue.resseq for residue in residues)
    min_offset = 1 - max_resseq
    max_offset = len(target_sequence) - min_resseq

    best_mapping: dict[int, int] = {}
    best_identity = -1.0
    best_comparable = -1
    best_offset = 0

    for offset in range(min_offset, max_offset + 1):
        mapping: dict[int, int] = {}
        used: set[int] = set()
        for index, residue in enumerate(residues):
            position = residue.resseq + offset
            if 1 <= position <= len(target_sequence) and position not in used:
                mapping[index] = position
                used.add(position)
        identity, comparable = identity_for_mapping(residues, mapping, target_sequence)
        rank = (identity, comparable, -abs(offset))
        best_rank = (best_identity, best_comparable, -abs(best_offset))
        if rank > best_rank:
            best_mapping = mapping
            best_identity = identity
            best_comparable = comparable
            best_offset = offset

    return best_mapping, float(max(best_identity, 0.0)), best_offset


def needleman_wunsch_mapping(
    residues: list[ResidueRecord],
    target_sequence: str,
) -> tuple[dict[int, int], float]:
    """Global sequence alignment fallback; called only when numbering is unreliable."""
    pred_sequence = "".join(residue.aa for residue in residues)
    n = len(pred_sequence)
    m = len(target_sequence)
    if n == 0 or m == 0:
        return {}, 0.0

    match_score = 2
    mismatch_score = -1
    gap_score = -2

    score = np.zeros((n + 1, m + 1), dtype=np.int32)
    trace = np.zeros((n + 1, m + 1), dtype=np.uint8)
    score[:, 0] = np.arange(n + 1, dtype=np.int32) * gap_score
    score[0, :] = np.arange(m + 1, dtype=np.int32) * gap_score
    trace[1:, 0] = 1  # up
    trace[0, 1:] = 2  # left

    for i in range(1, n + 1):
        pred_aa = pred_sequence[i - 1]
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + (
                match_score if pred_aa == target_sequence[j - 1] else mismatch_score
            )
            up = score[i - 1, j] + gap_score
            left = score[i, j - 1] + gap_score
            if diag >= up and diag >= left:
                score[i, j] = diag
                trace[i, j] = 0
            elif up >= left:
                score[i, j] = up
                trace[i, j] = 1
            else:
                score[i, j] = left
                trace[i, j] = 2

    mapping: dict[int, int] = {}
    i, j = n, m
    while i > 0 or j > 0:
        direction = trace[i, j]
        if i > 0 and j > 0 and direction == 0:
            mapping[i - 1] = j
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or direction == 1):
            i -= 1
        else:
            j -= 1

    identity, _ = identity_for_mapping(residues, mapping, target_sequence)
    return mapping, identity


def map_prediction_residues(
    residues: list[ResidueRecord],
    target_sequence: str,
) -> tuple[dict[int, int], str, float]:
    direct, direct_identity = direct_numbering_mapping(residues, target_sequence)
    if len(direct) >= 3 and direct_identity >= 0.80:
        return direct, "pdb_resseq", direct_identity

    if len(residues) == len(target_sequence):
        sequential, sequential_identity = sequential_mapping(residues, target_sequence)
        if sequential_identity >= 0.80:
            return sequential, "sequential_full_length", sequential_identity

    pred_sequence = "".join(residue.aa for residue in residues)
    if pred_sequence and "X" not in pred_sequence:
        start = target_sequence.find(pred_sequence)
        if start >= 0 and target_sequence.find(pred_sequence, start + 1) < 0:
            mapping = {index: start + index + 1 for index in range(len(residues))}
            return mapping, "unique_sequence_substring", 1.0

    offset, offset_identity, offset_value = offset_mapping(residues, target_sequence)
    if len(offset) >= 3 and offset_identity >= 0.80:
        return offset, f"pdb_resseq_offset_{offset_value:+d}", offset_identity

    aligned, aligned_identity = needleman_wunsch_mapping(residues, target_sequence)
    if len(aligned) < 3:
        raise ValueError("Prediction sequence could not be mapped to target sequence")
    return aligned, "needleman_wunsch", aligned_identity


def parse_and_map_prediction(
    path: Path,
    target_sequence: str,
    preferred_chain: str | None,
) -> ParsedPrediction:
    chains = parse_pdb_chains(path)
    chain_id, residues = choose_prediction_chain(chains, preferred_chain)
    residue_mapping, method, identity = map_prediction_residues(
        residues,
        target_sequence,
    )

    atom_map: dict[tuple[int, str], np.ndarray] = {}
    for pred_index, target_position in residue_mapping.items():
        residue = residues[pred_index]
        for atom_name, coord in residue.atoms.items():
            atom_map[(target_position, atom_name)] = coord

    return ParsedPrediction(
        path=path,
        chain_id=chain_id,
        residue_to_position=residue_mapping,
        atoms_by_position=atom_map,
        mapping_method=method,
        mapping_identity=identity,
        mapped_residue_count=len(residue_mapping),
        prediction_residue_count=len(residues),
    )


# ---------------------------------------------------------------------------
# Mapping parsed prediction coordinates onto the benchmark evaluation graph
# ---------------------------------------------------------------------------


def sample_node_seq_ids(sample: dict[str, Any]) -> np.ndarray:
    node_residue = sample["node_residue"].cpu().numpy().astype(np.int64)
    residue_seq_id = sample["residue_seq_id"].cpu().numpy().astype(np.int64)
    return residue_seq_id[node_residue]


def sample_atom_names(sample: dict[str, Any]) -> np.ndarray:
    ids = sample["atom_type"].cpu().numpy().astype(np.int64)
    names = []
    for atom_id in ids:
        if 0 <= atom_id < len(ATOM_NAMES):
            names.append(ATOM_NAMES[atom_id])
        else:
            names.append("UNK")
    return np.asarray(names, dtype=object)


def prediction_to_sample_arrays(
    prediction: ParsedPrediction,
    sample: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    seq_ids = sample_node_seq_ids(sample)
    atom_names = sample_atom_names(sample)
    n = len(seq_ids)

    coords = np.zeros((n, 3), dtype=np.float64)
    mask = np.zeros(n, dtype=bool)

    for index, (seq_id, atom_name) in enumerate(zip(seq_ids, atom_names)):
        coord = prediction.atoms_by_position.get((int(seq_id), str(atom_name)))
        if coord is not None:
            coords[index] = coord
            mask[index] = True
    return coords, mask


# ---------------------------------------------------------------------------
# Five metrics (same definitions as benchmark evaluator)
# ---------------------------------------------------------------------------


def fit_transform(
    mobile: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(mobile) < 3:
        raise ValueError("At least three C-alpha atoms are required for alignment")
    return kabsch_mobile_to_reference(
        np.asarray(mobile, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    )


def apply_transform(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ rotation + translation


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    difference = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=-1))))


def prepare_metric_context(
    sample: dict[str, Any],
    lddt_radius: float,
) -> MetricContext:
    target = sample["pos_target"].cpu().numpy().astype(np.float64)
    initial = sample["pos_init"].cpu().numpy().astype(np.float64)
    target_mask = sample["target_mask"].cpu().numpy().astype(bool)
    is_ca = sample["is_ca"].cpu().numpy().astype(bool)
    distance = sample["distance_to_mutation"].cpu().numpy().astype(np.float64)
    node_residue = sample["node_residue"].cpu().numpy().astype(np.int64)
    seq_ids = sample_node_seq_ids(sample)
    atom_names = sample_atom_names(sample)
    mutation_mask = sample["is_mutation_site"].cpu().numpy().astype(bool)
    mt_ids = sample["res_type_mt"].cpu().numpy().astype(np.int64)
    mt_aa_by_node = np.asarray(
        [AA_ORDER[x] if 0 <= x < len(AA_ORDER) else "X" for x in mt_ids],
        dtype=object,
    )

    target_indices = np.where(target_mask)[0]
    if len(target_indices) >= 2:
        tree = cKDTree(target[target_indices])
        pairs = tree.query_pairs(r=float(lddt_radius), output_type="ndarray")
        if pairs.size:
            global_i = target_indices[pairs[:, 0]]
            global_j = target_indices[pairs[:, 1]]
            different_residue = node_residue[global_i] != node_residue[global_j]
            global_i = global_i[different_residue]
            global_j = global_j[different_residue]
        else:
            global_i = np.asarray([], dtype=np.int64)
            global_j = np.asarray([], dtype=np.int64)
    else:
        global_i = np.asarray([], dtype=np.int64)
        global_j = np.asarray([], dtype=np.int64)

    return MetricContext(
        target=target,
        initial=initial,
        target_mask=target_mask,
        is_ca=is_ca,
        distance_to_mutation=distance,
        node_residue=node_residue,
        seq_ids=seq_ids,
        atom_names=atom_names,
        mutation_mask=mutation_mask,
        mt_aa_by_node=mt_aa_by_node,
        full_lddt_i=global_i,
        full_lddt_j=global_j,
    )


def symmetry_correct_prediction(
    predicted_aligned: np.ndarray,
    prediction_mask: np.ndarray,
    context: MetricContext,
) -> np.ndarray:
    result = np.asarray(predicted_aligned, dtype=np.float64).copy()
    observed = context.target_mask & prediction_mask

    for residue_id in np.unique(context.node_residue):
        indices = np.where(context.node_residue == residue_id)[0]
        if len(indices) == 0:
            continue

        aa = str(context.mt_aa_by_node[indices[0]])
        swap = SYMMETRY_SWAPS.get(aa)
        if not swap:
            continue

        name_to_index = {
            str(context.atom_names[index]): int(index)
            for index in indices
            if observed[index]
        }
        names = sorted(swap)
        if not all(name in name_to_index for name in names):
            continue

        current = [name_to_index[name] for name in names]
        source = [name_to_index[swap[name]] for name in names]
        identity_error = float(
            np.sum((result[current] - context.target[current]) ** 2)
        )
        swapped_error = float(
            np.sum((result[source] - context.target[current]) ** 2)
        )
        if swapped_error < identity_error:
            old = result.copy()
            for target_name in names:
                result[name_to_index[target_name]] = old[
                    name_to_index[swap[target_name]]
                ]

    return result


def all_atom_lddt(
    predicted: np.ndarray,
    prediction_mask: np.ndarray,
    context: MetricContext,
) -> tuple[float, int]:
    keep = prediction_mask[context.full_lddt_i] & prediction_mask[context.full_lddt_j]
    global_i = context.full_lddt_i[keep]
    global_j = context.full_lddt_j[keep]
    if len(global_i) == 0:
        return float("nan"), 0

    target_distance = np.linalg.norm(
        context.target[global_i] - context.target[global_j],
        axis=1,
    )
    predicted_distance = np.linalg.norm(
        predicted[global_i] - predicted[global_j],
        axis=1,
    )
    delta = np.abs(predicted_distance - target_distance)
    thresholds = (0.5, 1.0, 2.0, 4.0)
    score = float(
        np.mean([np.mean(delta < threshold) for threshold in thresholds])
    )
    return score, int(len(delta))


def compute_five_metrics(
    predicted: np.ndarray,
    prediction_mask: np.ndarray,
    context: MetricContext,
    local_radius: float,
) -> dict[str, Any]:
    common_ca = context.target_mask & prediction_mask & context.is_ca
    if int(common_ca.sum()) < 3:
        raise ValueError(
            f"Only {int(common_ca.sum())} common predicted/target C-alpha atoms"
        )

    rotation, translation, global_ca_rmsd = fit_transform(
        predicted[common_ca],
        context.target[common_ca],
    )
    predicted_aligned = apply_transform(predicted, rotation, translation)
    predicted_symmetry = symmetry_correct_prediction(
        predicted_aligned,
        prediction_mask,
        context,
    )

    local_ca = common_ca & (context.distance_to_mutation <= float(local_radius))
    local_count = int(local_ca.sum())
    local_rmsd = (
        rmsd(predicted_aligned[local_ca], context.target[local_ca])
        if local_count
        else float("nan")
    )

    lddt, lddt_pairs = all_atom_lddt(
        predicted_symmetry,
        prediction_mask,
        context,
    )

    sidechain = (
        context.target_mask
        & prediction_mask
        & context.mutation_mask
        & ~np.isin(context.atom_names, list(BACKBONE_NAMES))
    )
    sidechain_count = int(sidechain.sum())
    sidechain_rmsd = (
        rmsd(
            predicted_symmetry[sidechain],
            context.target[sidechain],
        )
        if sidechain_count
        else float("nan")
    )

    response_ca = common_ca & (context.distance_to_mutation <= float(local_radius))
    response_count = int(response_ca.sum())
    if response_count:
        pred_rotation, pred_translation, _ = fit_transform(
            predicted[common_ca],
            context.initial[common_ca],
        )
        pred_wt_frame = apply_transform(
            predicted,
            pred_rotation,
            pred_translation,
        )

        target_rotation, target_translation, _ = fit_transform(
            context.target[common_ca],
            context.initial[common_ca],
        )
        target_wt_frame = apply_transform(
            context.target,
            target_rotation,
            target_translation,
        )

        predicted_delta = (
            pred_wt_frame[response_ca] - context.initial[response_ca]
        )
        native_delta = (
            target_wt_frame[response_ca] - context.initial[response_ca]
        )
        response_rmsd = rmsd(predicted_delta, native_delta)
    else:
        response_rmsd = float("nan")

    return {
        "ca_rmsd": float(global_ca_rmsd),
        "lddt": lddt,
        "local_ca_rmsd_8a": local_rmsd,
        "mutation_sidechain_rmsd": sidechain_rmsd,
        "mutation_response_rmsd_8a": response_rmsd,
        "common_ca_count": int(common_ca.sum()),
        "local_ca_count_8a": local_count,
        "sidechain_atom_count": sidechain_count,
        "lddt_distance_count": lddt_pairs,
        "response_ca_count_8a": response_count,
        "predicted_node_count": int(prediction_mask.sum()),
        "target_observed_node_count": int(context.target_mask.sum()),
        "common_predicted_target_node_count": int(
            (prediction_mask & context.target_mask).sum()
        ),
    }


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summarize(frame: pd.DataFrame, unit_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in frame.groupby("model", sort=False):
        row: dict[str, Any] = {
            "model": model,
            f"n_{unit_name}": int(len(group)),
        }
        for metric in PRIMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            row[f"{metric}_n"] = int(len(finite))
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            row[f"{metric}_median"] = (
                float(finite.median()) if len(finite) else np.nan
            )
            row[f"{metric}_sd"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def manuscript_table(event_summary: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "model": "Model",
        "ca_rmsd_mean": "Cα RMSD ↓",
        "lddt_mean": "lDDT ↑",
        "local_ca_rmsd_8a_mean": "Local Cα RMSD 8Å ↓",
        "mutation_sidechain_rmsd_mean": "Mutation side-chain RMSD ↓",
        "mutation_response_rmsd_8a_mean": "MR-RMSD 8Å ↓",
    }
    return event_summary[list(columns)].rename(columns=columns)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    started = time.time()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    graph_config = GraphBuildConfig(
        radius=float(args.graph_radius),
        max_neighbors=int(args.graph_max_neighbors),
        cache_schema_version=int(args.cache_schema_version),
    )
    pair_class = None if args.pair_class == "all" else args.pair_class
    dataset = PairGraphDataset(
        manifest_path=args.manifest,
        structures_dir=args.structures_dir,
        cache_dir=args.cache_dir,
        split="test",
        graph_config=graph_config,
        local_radius=None,
        pair_class=pair_class,
        max_samples=args.max_cases,
    )

    # The dataset frame gives access to sequence/entity/chain metadata used to map
    # external predictions. It is sorted identically to __getitem__.
    manifest_frame = dataset.frame.reset_index(drop=True)

    esmfold_index: dict[str, list[Path]] = {}
    premut_map: dict[str, Path] = {}
    foldx_index: FoldXIndex | None = None
    rosetta_index: FoldXIndex | None = None

    if "esmfold" in args.models:
        esmfold_index = index_esmfold_predictions(args.esmfold_root)
    if "premut" in args.models:
        premut_map = load_premut_map(args.premut_root)
    if "foldx" in args.models:
        foldx_index = build_foldx_index(
            root=args.foldx_root,
            filename=args.foldx_filename,
            mapping_csv=args.foldx_map_csv,
        )
    if "rosetta" in args.models:
        rosetta_index = build_foldx_index(
            root=args.rosetta_root,
            filename=args.rosetta_filename,
            mapping_csv=args.rosetta_map_csv,
        )

    model_specs: list[tuple[str, str]] = []
    if "esmfold" in args.models:
        model_specs.append(("ESMFold", "esmfold"))
    if "premut" in args.models:
        model_specs.append(("PreMut", "premut"))
    if "foldx" in args.models:
        model_specs.append(("FoldX", "foldx"))
    if "rosetta" in args.models:
        model_specs.append(("Rosetta", "rosetta"))

    print("=" * 96)
    print("MutaStruct baseline evaluation — five primary metrics")
    print("=" * 96)
    print(f"Test pairs:          {len(dataset):,}")
    print(f"Models:              {[name for name, _ in model_specs]}")
    print(f"Manifest:            {args.manifest}")
    print(f"GT graph cache:      {args.cache_dir}")
    if "esmfold" in args.models:
        print(f"ESMFold files indexed:{sum(len(x) for x in esmfold_index.values()):,}")
    if "premut" in args.models:
        print(f"PreMut pair mappings:{len(premut_map):,}")
    if foldx_index is not None:
        print(f"FoldX files indexed: {foldx_index.indexed_files:,}")
        print(f"FoldX root:          {args.foldx_root}")
        if foldx_index.mapping_csv:
            print(f"FoldX mapping CSV:   {foldx_index.mapping_csv}")
    if rosetta_index is not None:
        print(f"Rosetta files indexed:{rosetta_index.indexed_files:,}")
        print(f"Rosetta root:         {args.rosetta_root}")
        if rosetta_index.mapping_csv:
            print(f"Rosetta mapping CSV:  {rosetta_index.mapping_csv}")
    print(f"Output:              {args.output_dir}")
    print("=" * 96)

    parsed_cache: dict[tuple[str, str, str], ParsedPrediction] = {}
    result_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []

    for index in tqdm(range(len(dataset)), desc="test pairs"):
        row = manifest_frame.iloc[index]
        pair_id = clean(row.get("pair_id"))
        event_id = clean(row.get("event_id"))

        try:
            sample = dataset[index]
            context = prepare_metric_context(sample, args.lddt_radius)
            target_sequence = target_sequence_from_row(row)
        except Exception as exc:
            for model_name, _ in model_specs:
                error_rows.append(
                    {
                        "model": model_name,
                        "pair_id": pair_id,
                        "event_id": event_id,
                        "stage": "load_reference_sample",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            continue

        common_metadata = {
            "pair_id": pair_id,
            "event_id": event_id,
            "pair_class": clean(row.get("pair_class")),
            "reference_pdb_id": clean(row.get("reference_pdb_id")),
            "target_pdb_id": clean(row.get("target_pdb_id")),
            "mutation": mutation_from_row(row),
            "mutation_index_0": int(float(row.get("mutation_index_0"))),
            "native_wt_mt_ca_rmsd": float(
                sample.get("target_alignment_ca_rmsd", np.nan)
            ),
        }

        for model_name, model_key in model_specs:
            try:
                if model_key == "esmfold":
                    prediction_path, discovery_method, duplicate_count = (
                        resolve_esmfold_prediction(row, esmfold_index)
                    )
                    preferred_chain = "A"
                elif model_key == "premut":
                    prediction_path, discovery_method, duplicate_count = (
                        resolve_premut_prediction(row, premut_map)
                    )
                    preferred_chain = "A"
                elif model_key == "foldx":
                    assert foldx_index is not None
                    prediction_path, discovery_method, duplicate_count = (
                        resolve_foldx_prediction(
                            row,
                            foldx_index,
                            strict=args.strict_foldx_mapping,
                        )
                    )
                    preferred_chain = "A"
                elif model_key == "rosetta":
                    assert rosetta_index is not None
                    prediction_path, discovery_method, duplicate_count = (
                        resolve_foldx_prediction(
                            row,
                            rosetta_index,
                            strict=args.strict_foldx_mapping,
                        )
                    )
                    preferred_chain = "A"
                else:
                    raise RuntimeError(model_key)

                cache_key = (
                    str(prediction_path.resolve()),
                    hashlib.sha256(target_sequence.encode("utf-8")).hexdigest(),
                    preferred_chain,
                )
                parsed = parsed_cache.get(cache_key)
                if parsed is None:
                    parsed = parse_and_map_prediction(
                        path=prediction_path,
                        target_sequence=target_sequence,
                        preferred_chain=preferred_chain,
                    )
                    parsed_cache[cache_key] = parsed

                predicted, prediction_mask = prediction_to_sample_arrays(
                    parsed,
                    sample,
                )
                metrics = compute_five_metrics(
                    predicted=predicted,
                    prediction_mask=prediction_mask,
                    context=context,
                    local_radius=args.local_radius,
                )

                result_rows.append(
                    {
                        "model": model_name,
                        **common_metadata,
                        **metrics,
                        "prediction_path": str(prediction_path),
                        "prediction_chain": parsed.chain_id,
                        "prediction_discovery_method": discovery_method,
                        "prediction_duplicate_count": duplicate_count,
                        "residue_mapping_method": parsed.mapping_method,
                        "residue_mapping_identity": parsed.mapping_identity,
                        "prediction_residue_count": parsed.prediction_residue_count,
                        "mapped_residue_count": parsed.mapped_residue_count,
                    }
                )
                mapping_rows.append(
                    {
                        "model": model_name,
                        "pair_id": pair_id,
                        "prediction_path": str(prediction_path),
                        "discovery_method": discovery_method,
                        "duplicate_count": duplicate_count,
                        "chain": parsed.chain_id,
                        "mapping_method": parsed.mapping_method,
                        "mapping_identity": parsed.mapping_identity,
                        "prediction_residues": parsed.prediction_residue_count,
                        "mapped_residues": parsed.mapped_residue_count,
                    }
                )

            except Exception as exc:
                error_rows.append(
                    {
                        "model": model_name,
                        "pair_id": pair_id,
                        "event_id": event_id,
                        "stage": "prediction_mapping_or_metrics",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    results = pd.DataFrame(result_rows)
    errors = pd.DataFrame(error_rows)
    mappings = pd.DataFrame(mapping_rows)

    results_path = args.output_dir / "pair_metrics.csv"
    results.to_csv(results_path, index=False)
    mappings.to_csv(args.output_dir / "prediction_mapping_audit.csv", index=False)
    if len(errors):
        errors.to_csv(args.output_dir / "errors.csv", index=False)

    if results.empty:
        raise RuntimeError(
            "No predictions were successfully evaluated. See errors.csv."
        )

    pair_summary = summarize(results, unit_name="pairs")
    pair_summary.to_csv(args.output_dir / "summary_pair_macro.csv", index=False)

    event_metrics = (
        results.groupby(["model", "event_id"], as_index=False, sort=False)[
            PRIMARY_METRICS
        ]
        .mean(numeric_only=True)
    )
    event_metrics.to_csv(args.output_dir / "event_metrics.csv", index=False)
    event_summary = summarize(event_metrics, unit_name="events")
    event_summary.to_csv(args.output_dir / "summary_event_macro.csv", index=False)

    main_table = manuscript_table(event_summary)
    main_table.to_csv(args.output_dir / "main_table_event_macro.csv", index=False)

    class_event = (
        results.groupby(
            ["model", "pair_class", "event_id"],
            as_index=False,
            sort=False,
        )[PRIMARY_METRICS]
        .mean(numeric_only=True)
    )
    class_rows: list[dict[str, Any]] = []
    for (model, pair_class_value), group in class_event.groupby(
        ["model", "pair_class"], sort=False
    ):
        row_out: dict[str, Any] = {
            "model": model,
            "pair_class": pair_class_value,
            "n_events": int(group["event_id"].nunique()),
        }
        for metric in PRIMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            row_out[f"{metric}_mean"] = (
                float(finite.mean()) if len(finite) else np.nan
            )
            row_out[f"{metric}_median"] = (
                float(finite.median()) if len(finite) else np.nan
            )
            row_out[f"{metric}_n"] = int(len(finite))
        class_rows.append(row_out)
    pd.DataFrame(class_rows).to_csv(
        args.output_dir / "summary_by_pair_class_event_macro.csv",
        index=False,
    )

    coverage_rows: list[dict[str, Any]] = []
    requested_pairs = len(dataset)
    for model_name, _ in model_specs:
        model_results = results[results["model"] == model_name]
        model_errors = errors[errors["model"] == model_name] if len(errors) else errors
        coverage_rows.append(
            {
                "model": model_name,
                "requested_pairs": requested_pairs,
                "successfully_evaluated_pairs": int(len(model_results)),
                "failed_pairs": int(len(model_errors)),
                "unique_prediction_files": int(
                    model_results["prediction_path"].nunique()
                ),
                "sidechain_rmsd_evaluable_pairs": int(
                    pd.to_numeric(
                        model_results["mutation_sidechain_rmsd"], errors="coerce"
                    ).notna().sum()
                ),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.output_dir / "prediction_coverage.csv", index=False)

    metadata = {
        "requested_test_pairs": requested_pairs,
        "models": [name for name, _ in model_specs],
        "manifest": str(args.manifest.resolve()),
        "structures_dir": str(args.structures_dir.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "esmfold_root": str(args.esmfold_root.resolve()) if "esmfold" in args.models else "",
        "premut_root": str(args.premut_root.resolve()) if "premut" in args.models else "",
        "foldx_root": (
            str(args.foldx_root.resolve()) if "foldx" in args.models else ""
        ),
        "rosetta_root": (
            str(args.rosetta_root.resolve()) if "rosetta" in args.models else ""
        ),
        "primary_metrics": PRIMARY_METRICS,
        "recommended_primary_aggregation": "event-macro mean",
        "local_radius_angstrom": float(args.local_radius),
        "lddt_radius_angstrom": float(args.lddt_radius),
        "lddt_thresholds_angstrom": [0.5, 1.0, 2.0, 4.0],
        "evaluation_universe": (
            "benchmark model-represented atoms intersected with experimentally "
            "observed target atoms and atoms present in each baseline prediction"
        ),
        "elapsed_seconds": float(time.time() - started),
    }
    (args.output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 96)
    print("Recommended manuscript table — event-macro mean")
    print("=" * 96)
    display = main_table.copy()
    for column in display.columns:
        if column != "Model":
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda value: f"{value:.4f}" if np.isfinite(value) else "NA"
            )
    print(display.to_string(index=False))
    print()
    print("Prediction coverage")
    print(coverage.to_string(index=False))
    print()
    print(f"Pair metrics:       {results_path}")
    print(f"Main table:         {args.output_dir / 'main_table_event_macro.csv'}")
    print(f"Mapping audit:      {args.output_dir / 'prediction_mapping_audit.csv'}")
    if len(errors):
        print(f"Errors:             {args.output_dir / 'errors.csv'}")
    print("=" * 96)


if __name__ == "__main__":
    main()
