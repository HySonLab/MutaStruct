#!/usr/bin/env python3
"""Run the released PreMut checkpoint on the locked MutaStruct Tier-A test split.

Key design choices
------------------
* Model input is ONLY the experimental reference structure + mutation.
  The experimental target structure is never used to construct the PreMut graph.
* Reads MutaStruct mmCIF files directly instead of routing them through a lossy
  CIF->PDB conversion.
* Reproduces the released PreMut mutant-topology initialization:
    - non-mutated residues use the reference residue topology;
    - the mutation site uses the TARGET residue atom topology;
    - atoms shared with the reference residue keep their reference coordinates;
    - target-only atoms are randomly initialized 1.54 Å from the reference Cα;
    - missing expected atoms elsewhere are initialized at (0,0,0).
* Reproduces the released PreMut graph construction: 30 nearest neighbors and
  edge distance as the edge feature.
* Reproduces the upstream quirk where nearest-neighbor edges are built from a
  second stochastic mutant-side-chain initialization.
* Uses a deterministic seed derived from the UNIQUE model input, so repeated
  coordinate-pair rows with the same reference structure + mutation receive the
  exact same prediction.
* Predictions are deduplicated by model input, while a CSV maps every test pair
  to its prediction.

Defaults are relative to the MutaStruct checkout, and upstream PreMut
code and checkpoint must be installed separately.

Example smoke run (from the repository root):
    python baselines/premut/run.py \
      --premut-repo /path/to/PreMut \
      --checkpoint /path/to/PreMut/Saved_Model/model.ckpt \
      --manifest /path/to/tier_a_pairs_with_splits.parquet \
      --structures-dir /path/to/structures \
      --max-cases 20

Omit --max-cases for the full test split.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREMUT_REPO = PROJECT_ROOT / "external" / "PreMut"
DEFAULT_MANIFEST = Path(os.environ.get("MUTASTRUCT_MANIFEST", PROJECT_ROOT / "data" / "tier_a_pairs_with_splits.parquet"))
DEFAULT_STRUCTURES = Path(os.environ.get("MUTASTRUCT_STRUCTURES", PROJECT_ROOT / "data" / "structures"))
DEFAULT_CHECKPOINT = None
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions" / "PreMut" / "mutastruct_test_predictions"
AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def load_upstream(premut_repo: Path) -> None:
    """Import upstream PreMut lazily, so --help works without its dependencies."""
    global AMINO_ACID_ATOM, AMINO_ACIDS_1_TO_3, ATOMS, BASE_AMINO_ACIDS_3_LETTER
    global egnn_ablation_atom_types_only, ATOM_TO_INDEX, AA3_TO_INDEX, NearestNeighbors
    from sklearn.neighbors import NearestNeighbors
    src = premut_repo / "src"
    if not (src / "constants.py").is_file() or not (src / "model.py").is_file():
        raise FileNotFoundError("--premut-repo must point to the separately installed PreMut repository")
    sys.path.insert(0, str(src))
    from constants import AMINO_ACID_ATOM, AMINO_ACIDS_1_TO_3, ATOMS, BASE_AMINO_ACIDS_3_LETTER
    from model import egnn_ablation_atom_types_only
    ATOM_TO_INDEX = {name: i for i, name in enumerate(ATOMS)}
    AA3_TO_INDEX = {name: i for i, name in enumerate(BASE_AMINO_ACIDS_3_LETTER)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--premut-repo", type=Path, default=DEFAULT_PREMUT_REPO)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--structures-dir", type=Path, default=DEFAULT_STRUCTURES)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument(
        "--pair-class",
        choices=[
            "all",
            "canonical_wt",
            "background_matched",
            "mapped_core_length_mismatch",
        ],
        default="all",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--save-pair-copies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also create pair_predictions/<pair_id>.pdb for convenient evaluation.",
    )
    return p.parse_args()


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


def clean_sequence(value: Any) -> str:
    return "".join(clean(value).split()).upper()


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**31 - 1)


def prediction_id(key: str) -> str:
    return "premut_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def resolve_structure_path(
    path_value: Any,
    pdb_id: Any,
    structures_dir: Path,
) -> Path:
    candidates: list[Path] = []

    path_str = clean(path_value)
    if path_str:
        path = Path(path_str)
        candidates.extend([path, structures_dir / path.name])

    pdb = clean(pdb_id)
    if pdb:
        for stem in (pdb, pdb.lower(), pdb.upper()):
            candidates.extend(
                [
                    structures_dir / f"{stem}.cif.gz",
                    structures_dir / f"{stem}.cif",
                ]
            )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find structure for PDB {pdb!r} under {structures_dir}"
    )


def read_cif(path: Path) -> gemmi.cif.Block:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            doc = gemmi.cif.read_string(handle.read())
    else:
        doc = gemmi.cif.read_file(str(path))
    return doc.sole_block()


def column(block: gemmi.cif.Block, key: str) -> list[str]:
    values = list(block.find_values(key))
    return [gemmi.cif.as_string(x) for x in values]


def normalized_entity_id(value: Any) -> str:
    text = clean(value)
    if not text:
        return ""
    try:
        f = float(text)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return text


def resolve_label_asym_id(
    block: gemmi.cif.Block,
    explicit_label: str,
    entity_id: str,
    auth_asym_id: str,
) -> str:
    """Resolve the polymer chain without ever using the target structure."""
    labels = column(block, "_atom_site.label_asym_id")
    seq_ids = column(block, "_atom_site.label_seq_id")
    atom_ids = column(block, "_atom_site.label_atom_id")
    entity_ids = column(block, "_atom_site.label_entity_id")
    auth_ids = column(block, "_atom_site.auth_asym_id")
    models = column(block, "_atom_site.pdbx_PDB_model_num")

    n = len(labels)
    if not (len(seq_ids) == len(atom_ids) == len(entity_ids) == len(auth_ids) == len(models) == n):
        raise ValueError("Inconsistent _atom_site column lengths")

    chain_positions: dict[str, set[int]] = defaultdict(set)
    chain_entities: dict[str, set[str]] = defaultdict(set)
    chain_auth: dict[str, set[str]] = defaultdict(set)

    for i in range(n):
        model = clean(models[i])
        if model and model not in {"1", "1.0"}:
            continue
        if clean(atom_ids[i]).upper() != "CA":
            continue

        seq_text = clean(seq_ids[i])
        if not seq_text:
            continue
        try:
            seq_id = int(float(seq_text))
        except ValueError:
            continue

        label = clean(labels[i])
        if not label:
            continue

        chain_positions[label].add(seq_id)
        ent = normalized_entity_id(entity_ids[i])
        auth = clean(auth_ids[i])
        if ent:
            chain_entities[label].add(ent)
        if auth:
            chain_auth[label].add(auth)

    if not chain_positions:
        raise ValueError("No protein-like polymer Cα chains found in mmCIF")

    if explicit_label:
        if explicit_label in chain_positions:
            return explicit_label
        raise ValueError(
            f"Requested label_asym_id={explicit_label!r} not found; "
            f"available={sorted(chain_positions)}"
        )

    candidates = list(chain_positions)

    if entity_id:
        entity_id = normalized_entity_id(entity_id)
        by_entity = [
            c for c in candidates if entity_id in chain_entities.get(c, set())
        ]
        if by_entity:
            candidates = by_entity

    if auth_asym_id:
        by_auth = [
            c for c in candidates if auth_asym_id in chain_auth.get(c, set())
        ]
        if by_auth:
            candidates = by_auth

    # Deterministic choice: greatest observed CA coverage, then lexical ID.
    candidates.sort(key=lambda c: (-len(chain_positions[c]), c))
    if not candidates:
        raise ValueError("Unable to resolve reference chain")
    return candidates[0]


def choose_altloc(
    rows: list[tuple[float, int, np.ndarray]],
) -> np.ndarray:
    """Highest occupancy; blank altloc preferred over A over others on a tie."""
    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return rows[0][2]


def parse_reference_atoms(
    cif_path: Path,
    label_asym_id: str,
) -> dict[int, dict[str, np.ndarray]]:
    """Observed model-1 protein atom coordinates keyed by label_seq_id and atom name."""
    block = read_cif(cif_path)

    labels = column(block, "_atom_site.label_asym_id")
    seq_ids = column(block, "_atom_site.label_seq_id")
    atom_ids = column(block, "_atom_site.label_atom_id")
    xs = column(block, "_atom_site.Cartn_x")
    ys = column(block, "_atom_site.Cartn_y")
    zs = column(block, "_atom_site.Cartn_z")
    models = column(block, "_atom_site.pdbx_PDB_model_num")
    alt_ids = column(block, "_atom_site.label_alt_id")
    occupancies = column(block, "_atom_site.occupancy")

    n = len(labels)
    expected_lengths = [
        len(seq_ids), len(atom_ids), len(xs), len(ys), len(zs),
        len(models), len(alt_ids), len(occupancies),
    ]
    if any(length != n for length in expected_lengths):
        raise ValueError(f"{cif_path.name}: inconsistent _atom_site columns")

    candidates: dict[int, dict[str, list[tuple[float, int, np.ndarray]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for i in range(n):
        if clean(labels[i]) != label_asym_id:
            continue

        model = clean(models[i])
        if model and model not in {"1", "1.0"}:
            continue

        atom_name = clean(atom_ids[i]).upper()
        if atom_name not in ATOM_TO_INDEX:
            continue

        seq_text = clean(seq_ids[i])
        if not seq_text:
            continue
        try:
            seq_id = int(float(seq_text))
        except ValueError:
            continue

        try:
            xyz = np.asarray(
                [float(xs[i]), float(ys[i]), float(zs[i])],
                dtype=np.float64,
            )
        except ValueError:
            continue
        if not np.isfinite(xyz).all():
            continue

        try:
            occ = float(clean(occupancies[i]) or "1")
        except ValueError:
            occ = 1.0

        alt = clean(alt_ids[i])
        alt_rank = 2 if not alt else 1 if alt == "A" else 0

        # Match released PreMut's acceptable altloc policy: blank or A.
        if alt not in {"", "A"}:
            continue

        candidates[seq_id][atom_name].append((occ, alt_rank, xyz))

    output: dict[int, dict[str, np.ndarray]] = {}
    for seq_id, atoms in candidates.items():
        output[seq_id] = {
            atom: choose_altloc(options)
            for atom, options in atoms.items()
        }

    if not output:
        raise ValueError(
            f"{cif_path.name}: no accepted atoms found for label_asym_id={label_asym_id!r}"
        )

    return output


def random_point_on_sphere(
    center: np.ndarray,
    radius: float,
    rng: random.Random,
) -> np.ndarray:
    """Reproduce PreMut's released initialization, including its phi sampling."""
    theta = rng.uniform(0.0, 2.0 * math.pi)
    phi = rng.uniform(0.0, math.pi)
    return np.asarray(
        [
            center[0] + radius * math.sin(phi) * math.cos(theta),
            center[1] + radius * math.sin(phi) * math.sin(theta),
            center[2] + radius * math.cos(phi),
        ],
        dtype=np.float64,
    )


def validate_pair_sequences(row: pd.Series) -> tuple[str, str, int, str, str]:
    ref_seq = clean_sequence(row.get("reference_sequence"))
    mut_seq = clean_sequence(row.get("target_sequence"))

    if not ref_seq or not mut_seq:
        raise ValueError("Manifest is missing reference_sequence or target_sequence")
    if len(ref_seq) != len(mut_seq):
        raise ValueError(
            f"Reference/target sequence length mismatch: {len(ref_seq)} vs {len(mut_seq)}"
        )

    diffs = [i for i, (a, b) in enumerate(zip(ref_seq, mut_seq)) if a != b]
    if len(diffs) != 1:
        raise ValueError(f"Expected one substitution, found {len(diffs)}")

    idx = diffs[0]

    annotated = clean(row.get("mutation_index_0"))
    if annotated:
        if int(float(annotated)) != idx:
            raise ValueError(
                f"Mutation index mismatch: sequences={idx}, manifest={annotated}"
            )

    ref_aa = ref_seq[idx]
    mut_aa = mut_seq[idx]
    if ref_aa not in AA20 or mut_aa not in AA20:
        raise ValueError(f"Non-standard mutation {ref_aa}{idx+1}{mut_aa}")

    return ref_seq, mut_seq, idx, ref_aa, mut_aa


def build_premut_input(
    observed_atoms: dict[int, dict[str, np.ndarray]],
    reference_sequence: str,
    mutation_index_0: int,
    reference_aa: str,
    target_aa: str,
    rng: random.Random,
) -> dict[str, Any]:
    """Build PreMut's mutant-topology all-atom input from the reference structure."""
    mutation_seq_id = mutation_index_0 + 1

    if mutation_seq_id not in observed_atoms:
        raise ValueError(
            f"Mutation position {mutation_seq_id} has no observed reference coordinates"
        )

    # PreMut centers the observed reference atoms before it fills missing atoms.
    observed_xyz = np.vstack(
        [
            xyz
            for atoms in observed_atoms.values()
            for xyz in atoms.values()
        ]
    )
    centroid = observed_xyz.mean(axis=0)

    centered: dict[int, dict[str, np.ndarray]] = {
        seq_id: {
            atom: coord - centroid
            for atom, coord in atoms.items()
        }
        for seq_id, atoms in observed_atoms.items()
    }

    max_observed_seq_id = max(centered)

    coords: list[np.ndarray] = []
    atom_features: list[list[float]] = []
    residue_features: list[list[float]] = []
    node_seq_ids: list[int] = []
    node_atom_names: list[str] = []
    node_residue_names: list[str] = []

    for seq_id in sorted(centered):
        if not (1 <= seq_id <= len(reference_sequence)):
            continue

        wt_aa = reference_sequence[seq_id - 1]
        if wt_aa not in AA20:
            continue

        is_mutation = seq_id == mutation_seq_id
        output_aa = target_aa if is_mutation else wt_aa
        residue_name = AMINO_ACIDS_1_TO_3[output_aa]

        expected_atoms = list(AMINO_ACID_ATOM[residue_name])
        if seq_id == max_observed_seq_id:
            expected_atoms = expected_atoms + ["OXT"]

        available = centered[seq_id]

        if is_mutation:
            if "CA" not in available:
                raise ValueError("Mutation-site Cα is missing in reference structure")
            ca = available["CA"]

        for atom_name in expected_atoms:
            if atom_name in available:
                xyz = available[atom_name]
            elif is_mutation:
                # Released PreMut randomly initializes target-only / missing
                # mutation-site atoms around reference Cα.
                xyz = random_point_on_sphere(ca, 1.54, rng)
            else:
                # Released PreMut uses zeros for expected but unresolved atoms.
                xyz = np.zeros(3, dtype=np.float64)

            atom_onehot = [0.0] * len(ATOMS)
            atom_onehot[ATOM_TO_INDEX[atom_name]] = 1.0

            residue_onehot = [0.0] * (len(BASE_AMINO_ACIDS_3_LETTER) + 1)
            if is_mutation:
                mut_name = AMINO_ACIDS_1_TO_3[target_aa]
                wt_name = AMINO_ACIDS_1_TO_3[reference_aa]
                if mut_name in AA3_TO_INDEX:
                    residue_onehot[AA3_TO_INDEX[mut_name]] += 1.0
                else:
                    residue_onehot[-1] += 1.0
                if wt_name in AA3_TO_INDEX:
                    residue_onehot[AA3_TO_INDEX[wt_name]] = -1.0
                else:
                    residue_onehot[-1] = -1.0
            else:
                wt_name = AMINO_ACIDS_1_TO_3[wt_aa]
                if wt_name in AA3_TO_INDEX:
                    residue_onehot[AA3_TO_INDEX[wt_name]] = 1.0
                else:
                    residue_onehot[-1] = 1.0

            coords.append(np.asarray(xyz, dtype=np.float64))
            atom_features.append(atom_onehot)
            residue_features.append(residue_onehot)
            node_seq_ids.append(seq_id)
            node_atom_names.append(atom_name)
            node_residue_names.append(residue_name)

    coords_np = np.asarray(coords, dtype=np.float64)
    if len(coords_np) < 2:
        raise ValueError("Too few PreMut graph nodes")

    return {
        "coords": coords_np,
        "atom_features": np.asarray(atom_features, dtype=np.float32),
        "residue_features": np.asarray(residue_features, dtype=np.float32),
        "seq_ids": np.asarray(node_seq_ids, dtype=np.int64),
        "atom_names": node_atom_names,
        "residue_names": node_residue_names,
    }


def build_knn_edges(coords: np.ndarray, k: int = 30) -> tuple[np.ndarray, np.ndarray]:
    n = len(coords)
    if n < 2:
        raise ValueError("Need at least two graph nodes")
    n_neighbors = min(int(k), n)

    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(coords)
    distances, neighbors = knn.kneighbors(coords)

    source = np.repeat(np.arange(n, dtype=np.int64), n_neighbors)
    target = neighbors.reshape(-1).astype(np.int64)
    edge_index = np.stack([source, target], axis=0)
    edge_attr = distances.reshape(-1, 1).astype(np.float32)
    return edge_index, edge_attr


def load_premut_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = egnn_ablation_atom_types_only(
        in_channels=[3, 37, 21],
        hidden_channels=32,
        num_classes=3,
        num_hidden_layers=4,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state = checkpoint.get("state_dict", checkpoint)

    # Reproduce upstream prediction.py's Lightning -> bare-network key mapping.
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if "model." in key:
            new_key = "model." + key.split("model.")[-1]
        else:
            new_key = key
        remapped[new_key] = value

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            f"Missing keys: {missing}\nUnexpected keys: {unexpected}"
        )

    model.to(device).eval()
    return model


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    graph_input: dict[str, Any],
    edge_coords: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    edge_index_np, edge_attr_np = build_knn_edges(edge_coords, k=30)

    node_coords = torch.from_numpy(graph_input["coords"]).float().to(device)
    atom_features = torch.from_numpy(graph_input["atom_features"]).float().to(device)
    residue_features = (
        torch.from_numpy(graph_input["residue_features"]).float().to(device)
    )
    edge_index = torch.from_numpy(edge_index_np).long().to(device)
    edge_attr = torch.from_numpy(edge_attr_np).float().to(device)
    batch = torch.zeros(len(node_coords), dtype=torch.long, device=device)

    features = (node_coords, atom_features, residue_features)
    delta = model(
        node_feats=features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
    )
    predicted = delta + node_coords
    return predicted.detach().cpu().numpy().astype(np.float64)


def write_prediction_pdb(
    path: Path,
    coords: np.ndarray,
    graph_input: dict[str, Any],
) -> None:
    """Write model coordinates while preserving polymer sequence position."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for serial, (xyz, seq_id, atom_name, residue_name) in enumerate(
        zip(
            coords,
            graph_input["seq_ids"],
            graph_input["atom_names"],
            graph_input["residue_names"],
        ),
        start=1,
    ):
        x, y, z = map(float, xyz)
        # Protein atom names all begin with their element in this topology.
        element = atom_name[0]
        lines.append(
            f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} A"
            f"{int(seq_id):4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}"
        )

    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    load_upstream(args.premut_repo.resolve())
    if args.checkpoint is None:
        args.checkpoint = args.premut_repo / "Saved_Model" / "model.ckpt"

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Use --device cpu or activate the CUDA-enabled PreMut environment."
        )
    device = torch.device(args.device)

    for path in (args.manifest, args.structures_dir, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    unique_dir = args.output_dir / "unique_predictions"
    pair_dir = args.output_dir / "pair_predictions"
    unique_dir.mkdir(exist_ok=True)
    if args.save_pair_copies:
        pair_dir.mkdir(exist_ok=True)

    if args.manifest.suffix == ".parquet":
        frame = pd.read_parquet(args.manifest)
    else:
        frame = pd.read_csv(args.manifest, dtype=str)

    if "split" not in frame.columns:
        raise ValueError("Manifest has no 'split' column")
    frame = frame[frame["split"].astype(str).str.lower().eq("test")].copy()

    if args.pair_class != "all":
        frame = frame[frame["pair_class"].eq(args.pair_class)].copy()

    if args.max_cases is not None:
        frame = frame.head(args.max_cases).copy()

    required = [
        "pair_id",
        "reference_pdb_id",
        "reference_structure_path",
        "reference_sequence",
        "target_sequence",
    ]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    model = load_premut_model(args.checkpoint, device)

    print("=" * 88)
    print("PreMut — MutaStruct Tier-A test prediction")
    print("=" * 88)
    print(f"Repository:       {args.premut_repo}")
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Manifest:         {args.manifest}")
    print(f"Structures:       {args.structures_dir}")
    print(f"Device:           {device}")
    print(f"Test pairs:       {len(frame):,}")
    print(f"Output:           {args.output_dir}")
    print("=" * 88)

    # We first construct deterministic unique-input keys from manifest metadata.
    planned_rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        s = pd.Series(row._asdict())
        ref_seq, mut_seq, mut_idx, ref_aa, mut_aa = validate_pair_sequences(s)

        ref_pdb = clean(s.get("reference_pdb_id")).upper()
        ref_entity = clean(s.get("reference_entity_id"))
        ref_label = clean(s.get("reference_chain_label_asym_id"))
        ref_auth = clean(s.get("reference_chain_auth_asym_id"))

        input_key = "|".join(
            [
                ref_pdb,
                ref_entity,
                ref_label,
                ref_auth,
                hashlib.sha256(ref_seq.encode()).hexdigest()[:16],
                f"{ref_aa}{mut_idx}{mut_aa}",
            ]
        )
        pred_id = prediction_id(input_key)

        planned_rows.append(
            {
                **s.to_dict(),
                "_reference_sequence": ref_seq,
                "_target_sequence": mut_seq,
                "_mutation_index_0": mut_idx,
                "_reference_aa": ref_aa,
                "_target_aa": mut_aa,
                "prediction_input_key": input_key,
                "prediction_id": pred_id,
            }
        )

    planned = pd.DataFrame(planned_rows)
    unique_inputs = (
        planned.sort_values("pair_id")
        .drop_duplicates("prediction_id", keep="first")
        .copy()
    )

    print(f"Unique PreMut inputs: {len(unique_inputs):,}")
    print(
        f"Redundant inference avoided: {len(planned) - len(unique_inputs):,} "
        f"pair rows"
    )

    prediction_records: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    started = time.time()

    for number, (_, row) in enumerate(unique_inputs.iterrows(), start=1):
        pred_id = row["prediction_id"]
        output_path = unique_dir / f"{pred_id}.pdb"

        if output_path.exists() and not args.overwrite:
            prediction_records.append(
                {
                    "prediction_id": pred_id,
                    "prediction_path": str(output_path.resolve()),
                    "status": "existing",
                    "error": "",
                }
            )
            continue

        try:
            cif_path = resolve_structure_path(
                row.get("reference_structure_path"),
                row.get("reference_pdb_id"),
                args.structures_dir,
            )
            block = read_cif(cif_path)

            label_asym_id = resolve_label_asym_id(
                block=block,
                explicit_label=clean(row.get("reference_chain_label_asym_id")),
                entity_id=clean(row.get("reference_entity_id")),
                auth_asym_id=clean(row.get("reference_chain_auth_asym_id")),
            )

            observed = parse_reference_atoms(cif_path, label_asym_id)

            seed = stable_seed(args.seed, row["prediction_input_key"])
            rng = random.Random(seed)

            # First stochastic initialization -> actual model node coordinates.
            graph_input = build_premut_input(
                observed_atoms=observed,
                reference_sequence=row["_reference_sequence"],
                mutation_index_0=int(row["_mutation_index_0"]),
                reference_aa=row["_reference_aa"],
                target_aa=row["_target_aa"],
                rng=rng,
            )

            # Released PDBReader_All_Atom calls get_xyz() a second time when making
            # kNN edges. Reproduce that stochastic quirk for baseline fidelity.
            edge_input = build_premut_input(
                observed_atoms=observed,
                reference_sequence=row["_reference_sequence"],
                mutation_index_0=int(row["_mutation_index_0"]),
                reference_aa=row["_reference_aa"],
                target_aa=row["_target_aa"],
                rng=rng,
            )

            if len(graph_input["coords"]) != len(edge_input["coords"]):
                raise RuntimeError("Node topology changed between PreMut initializations")

            predicted = predict(
                model=model,
                graph_input=graph_input,
                edge_coords=edge_input["coords"],
                device=device,
            )

            write_prediction_pdb(output_path, predicted, graph_input)

            prediction_records.append(
                {
                    "prediction_id": pred_id,
                    "prediction_path": str(output_path.resolve()),
                    "status": "ok",
                    "error": "",
                    "reference_pdb_id": row.get("reference_pdb_id"),
                    "reference_entity_id": row.get("reference_entity_id"),
                    "reference_chain_label_asym_id": label_asym_id,
                    "mutation_index_0": int(row["_mutation_index_0"]),
                    "reference_aa": row["_reference_aa"],
                    "target_aa": row["_target_aa"],
                    "seed": seed,
                    "nodes": int(len(graph_input["coords"])),
                }
            )

        except Exception as exc:
            error_records.append(
                {
                    "prediction_id": pred_id,
                    "representative_pair_id": row.get("pair_id"),
                    "reference_pdb_id": row.get("reference_pdb_id"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

        if number % 25 == 0 or number == len(unique_inputs):
            print(
                f"[{number}/{len(unique_inputs)}] "
                f"predictions={len(prediction_records)} "
                f"errors={len(error_records)}",
                flush=True,
            )

    prediction_manifest = pd.DataFrame(prediction_records)
    prediction_manifest.to_csv(
        args.output_dir / "unique_prediction_manifest.csv",
        index=False,
    )

    error_ids = {x["prediction_id"] for x in error_records}
    path_by_id = {}
    for row in prediction_records:
        if row["status"] in {"ok", "existing"}:
            path_by_id[row["prediction_id"]] = row["prediction_path"]

    pair_map_rows: list[dict[str, Any]] = []
    for _, row in planned.iterrows():
        pred_id = row["prediction_id"]
        source = Path(path_by_id[pred_id]) if pred_id in path_by_id else None

        pair_path = ""
        status = "error" if pred_id in error_ids else "ok"

        if source is not None and args.save_pair_copies:
            destination = pair_dir / f"{row['pair_id']}.pdb"
            if args.overwrite or not destination.exists():
                hardlink_or_copy(source, destination)
            pair_path = str(destination.resolve())

        pair_map_rows.append(
            {
                "pair_id": row["pair_id"],
                "event_id": row.get("event_id"),
                "pair_class": row.get("pair_class"),
                "reference_pdb_id": row.get("reference_pdb_id"),
                "target_pdb_id": row.get("target_pdb_id"),
                "mutation": row.get("mutation"),
                "mutation_index_0": int(row["_mutation_index_0"]),
                "reference_aa": row["_reference_aa"],
                "target_aa": row["_target_aa"],
                "prediction_id": pred_id,
                "prediction_path": (
                    pair_path
                    if pair_path
                    else path_by_id.get(pred_id, "")
                ),
                "status": status,
            }
        )

    pair_map = pd.DataFrame(pair_map_rows)
    pair_map.to_csv(
        args.output_dir / "pair_to_prediction.csv",
        index=False,
    )

    if error_records:
        pd.DataFrame(error_records).to_csv(
            args.output_dir / "errors.csv",
            index=False,
        )

    successful_unique = len(path_by_id)
    successful_pairs = int(pair_map["status"].eq("ok").sum())

    summary = {
        "test_pairs_requested": int(len(frame)),
        "unique_model_inputs": int(len(unique_inputs)),
        "successful_unique_predictions": int(successful_unique),
        "failed_unique_predictions": int(len(error_records)),
        "successful_pair_mappings": successful_pairs,
        "failed_pair_mappings": int(len(pair_map) - successful_pairs),
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "structures_dir": str(args.structures_dir.resolve()),
        "device": str(device),
        "base_seed": int(args.seed),
        "inference_protocol": (
            "Released PreMut atom-topology/feature/random-initialization logic "
            "with direct MutaStruct mmCIF parsing; experimental target structures "
            "are not used for model input."
        ),
        "elapsed_seconds": float(time.time() - started),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("Finished")
    print("=" * 88)
    print(f"Unique inputs:             {len(unique_inputs):,}")
    print(f"Successful predictions:   {successful_unique:,}")
    print(f"Failed unique inputs:      {len(error_records):,}")
    print(f"Test pairs mapped:         {successful_pairs:,}/{len(pair_map):,}")
    print(f"Unique predictions:        {unique_dir}")
    if args.save_pair_copies:
        print(f"Pair prediction copies:    {pair_dir}")
    print(f"Pair mapping:              {args.output_dir / 'pair_to_prediction.csv'}")
    if error_records:
        print(f"Errors:                    {args.output_dir / 'errors.csv'}")
    print("=" * 88)


if __name__ == "__main__":
    main()
