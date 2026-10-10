from __future__ import annotations

import gzip
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import gemmi
import numpy as np

from .constants import AA3_TO_AA1, BACKBONE_ATOMS, VDW_RADII
from .geometry import kabsch_mobile_to_reference, transform_points
from .templates import atom_order, load_residue_templates


@dataclass
class AtomRecord:
    name: str
    element: str
    coord: np.ndarray
    occupancy: float
    b_factor: float | None
    alt_id: str
    comp_id: str


@dataclass
class ResidueRecord:
    label_seq_id: int
    comp_id: str
    atoms: dict[str, AtomRecord] = field(default_factory=dict)


@dataclass
class ChainRecord:
    entity_id: str
    label_asym_id: str
    residues: dict[int, ResidueRecord] = field(default_factory=dict)

    def ca_positions(self) -> dict[int, np.ndarray]:
        return {
            seq_id: residue.atoms["CA"].coord
            for seq_id, residue in self.residues.items()
            if "CA" in residue.atoms
        }


@dataclass
class NodeRecord:
    seq_id: int
    atom_name: str
    element: str
    pos_init: np.ndarray
    wt_aa: str
    mt_aa: str
    is_mutation_site: bool
    is_new_atom: bool


@dataclass
class PairStructure:
    nodes: list[NodeRecord]
    target_positions: dict[tuple[int, str], np.ndarray]
    target_alignment_rmsd: float
    reference_chain: ChainRecord
    target_chain: ChainRecord
    mutation_position: int
    sequence_length: int


def _parse_float(value: str | None, default: float | None = None) -> float | None:
    if value in (None, "", ".", "?"):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_int(value: str | None) -> int | None:
    if value in (None, "", ".", "?"):
        return None
    try:
        return int(value)
    except ValueError:
        try:
            numeric = float(value)
            return int(numeric) if numeric.is_integer() else None
        except ValueError:
            return None


def _alt_score(atom: AtomRecord) -> tuple[float, int, int]:
    return (
        atom.occupancy,
        int(atom.alt_id in ("", ".", "?")),
        int(atom.alt_id == "A"),
    )


def _read_atom_site_rows(path: str | Path) -> Iterable[dict[str, str]]:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            document = gemmi.cif.read_string(handle.read())
    else:
        document = gemmi.cif.read(str(path))
    block = document.sole_block()
    column = block.find_loop("_atom_site.group_PDB")
    if not column:
        raise ValueError(f"No _atom_site loop in {path}")
    loop = column.get_loop()
    tags = [tag.split(".", 1)[-1] for tag in loop.tags]
    width = loop.width()
    values = loop.values
    for offset in range(0, len(values), width):
        yield dict(zip(tags, values[offset : offset + width], strict=True))


@lru_cache(maxsize=32)
def parse_cif_chains(path: str | Path) -> dict[tuple[str, str], ChainRecord]:
    """Parse model-1 polymer heavy atoms keyed by (entity_id, label_asym_id)."""
    atom_candidates: dict[tuple[str, str, int, str], AtomRecord] = {}
    residue_comp_ids: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    chain_ids: set[tuple[str, str]] = set()

    for row in _read_atom_site_rows(path):
        model = row.get("pdbx_PDB_model_num", "1")
        if model not in ("1", "1.0", ".", "?"):
            continue
        seq_id = _parse_int(row.get("label_seq_id"))
        entity_id = str(row.get("label_entity_id", ""))
        asym_id = str(row.get("label_asym_id", ""))
        if seq_id is None or entity_id in ("", ".", "?") or asym_id in ("", ".", "?"):
            continue

        element = str(row.get("type_symbol", "")).strip().upper()
        if element in {"H", "D"}:
            continue
        atom_name = str(row.get("label_atom_id", "")).strip().upper()
        if not atom_name:
            continue
        comp_id = str(row.get("label_comp_id", "")).strip().upper()
        try:
            coord = np.asarray(
                [float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"])],
                dtype=float,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid coordinates in {path}: {exc}") from exc

        atom = AtomRecord(
            name=atom_name,
            element=element or "OTHER",
            coord=coord,
            occupancy=float(_parse_float(row.get("occupancy"), 1.0) or 0.0),
            b_factor=_parse_float(row.get("B_iso_or_equiv")),
            alt_id=str(row.get("label_alt_id", ".")),
            comp_id=comp_id,
        )
        key = (entity_id, asym_id, seq_id, atom_name)
        previous = atom_candidates.get(key)
        if previous is None or _alt_score(atom) > _alt_score(previous):
            atom_candidates[key] = atom
        residue_comp_ids[(entity_id, asym_id, seq_id)].append(comp_id)
        chain_ids.add((entity_id, asym_id))

    chains = {
        key: ChainRecord(entity_id=key[0], label_asym_id=key[1])
        for key in chain_ids
    }
    for (entity_id, asym_id, seq_id, atom_name), atom in atom_candidates.items():
        residue_key = (entity_id, asym_id, seq_id)
        comp_ids = residue_comp_ids[residue_key]
        comp_id = Counter(comp_ids).most_common(1)[0][0]
        residue = chains[(entity_id, asym_id)].residues.setdefault(
            seq_id,
            ResidueRecord(label_seq_id=seq_id, comp_id=comp_id),
        )
        residue.atoms[atom_name] = atom

    if not chains:
        raise ValueError(f"No polymer chains parsed from {path}")
    return chains


def select_chain(
    chains: dict[tuple[str, str], ChainRecord],
    entity_id: str,
    label_asym_id: str,
) -> ChainRecord:
    explicit = chains.get((str(entity_id), str(label_asym_id)))
    if explicit is not None:
        return explicit
    candidates = [
        chain for (candidate_entity, _), chain in chains.items()
        if candidate_entity == str(entity_id)
    ]
    if not candidates:
        available = sorted(f"{entity}:{asym}" for entity, asym in chains)
        raise ValueError(
            f"Entity {entity_id!r}/chain {label_asym_id!r} not found; available={available}"
        )
    candidates.sort(
        key=lambda chain: (len(chain.ca_positions()), chain.label_asym_id),
        reverse=True,
    )
    return candidates[0]


def resolve_structure_path(
    path_value: Any,
    pdb_id: str,
    structures_dir: Path,
) -> Path:
    candidates: list[Path] = []
    text = "" if path_value is None else str(path_value).strip()
    if text and text.lower() not in {"nan", "none", "<na>"}:
        path = Path(text)
        candidates.extend([path, structures_dir / path.name])
    for stem in (pdb_id, pdb_id.lower(), pdb_id.upper()):
        candidates.extend([
            structures_dir / f"{stem}.cif.gz",
            structures_dir / f"{stem}.cif",
        ])
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve mmCIF for PDB {pdb_id}; searched {structures_dir}"
    )


def _fit_template_to_reference(
    template_coords: dict[str, list[float]],
    reference_residue: ResidueRecord,
) -> tuple[np.ndarray, np.ndarray]:
    preferred = ["N", "CA", "C", "CB"]
    anchors = [
        name for name in preferred
        if name in template_coords and name in reference_residue.atoms
    ]
    if len(anchors) < 3:
        raise ValueError(
            f"Mutation residue {reference_residue.label_seq_id} lacks three template anchors"
        )
    mobile = np.asarray([template_coords[name] for name in anchors], dtype=float)
    reference = np.asarray([reference_residue.atoms[name].coord for name in anchors], dtype=float)
    rotation, translation, _ = kabsch_mobile_to_reference(mobile, reference)
    return rotation, translation


def _clash_score(
    candidate: dict[str, np.ndarray],
    candidate_elements: dict[str, str],
    new_atom_names: set[str],
    environment_atoms: list[AtomRecord],
) -> float:
    if not new_atom_names or not environment_atoms:
        return 0.0
    env_coords = np.asarray([atom.coord for atom in environment_atoms], dtype=float)
    env_radii = np.asarray(
        [VDW_RADII.get(atom.element, VDW_RADII["OTHER"]) for atom in environment_atoms],
        dtype=float,
    )
    score = 0.0
    for atom_name in sorted(new_atom_names):
        coord = candidate[atom_name]
        element = candidate_elements.get(atom_name, "OTHER")
        radius = VDW_RADII.get(element, VDW_RADII["OTHER"])
        distances = np.linalg.norm(env_coords - coord, axis=1)
        # A permissive fraction of the van der Waals sum is sufficient for
        # selecting among ideal conformers; the neural model performs refinement.
        minimum = 0.70 * (radius + env_radii)
        penetration = np.maximum(minimum - distances, 0.0)
        score += float(np.sum(penetration * penetration))
    return score


def initialize_mutant_residue(
    reference_chain: ChainRecord,
    mutation_position: int,
    target_aa: str,
    is_terminal: bool,
) -> dict[str, tuple[np.ndarray, str, bool]]:
    """Return target atom name -> (position, element, is_new_atom)."""
    templates = load_residue_templates()
    if target_aa not in templates:
        raise ValueError(f"Unsupported target amino acid {target_aa!r}")
    reference_residue = reference_chain.residues.get(mutation_position)
    if reference_residue is None:
        raise ValueError(f"Mutation position {mutation_position} is absent from reference chain")
    if not BACKBONE_ATOMS <= set(reference_residue.atoms):
        raise ValueError(f"Mutation position {mutation_position} lacks complete backbone atoms")

    template = templates[target_aa]
    allowed_names = set(atom_order(target_aa, include_oxt=is_terminal))
    reference_names = set(reference_residue.atoms)
    new_names = allowed_names - reference_names

    environment_atoms = [
        atom
        for seq_id, residue in reference_chain.residues.items()
        if seq_id != mutation_position
        for atom in residue.atoms.values()
        if atom.element not in {"H", "D"}
    ]

    best_score: float | None = None
    best_positions: dict[str, np.ndarray] | None = None
    for conformer in template["conformers"]:
        rotation, translation = _fit_template_to_reference(conformer, reference_residue)
        transformed = {
            name: np.asarray(coord, dtype=float) @ rotation + translation
            for name, coord in conformer.items()
            if name in allowed_names
        }
        # Preserve all observed atoms with names shared by WT and mutant topology.
        for name in allowed_names & reference_names:
            transformed[name] = reference_residue.atoms[name].coord.copy()
        score = _clash_score(
            transformed,
            template["elements"],
            new_names,
            environment_atoms,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_positions = transformed

    if best_positions is None:
        raise RuntimeError(f"No conformer could initialize mutation at {mutation_position}")

    result: dict[str, tuple[np.ndarray, str, bool]] = {}
    for name in atom_order(target_aa, include_oxt=is_terminal):
        if name not in best_positions:
            continue
        result[name] = (
            best_positions[name],
            str(template["elements"].get(name, "OTHER")).upper(),
            name not in reference_names,
        )
    return result


def align_target_chain_to_reference(
    reference_chain: ChainRecord,
    target_chain: ChainRecord,
) -> tuple[dict[tuple[int, str], np.ndarray], float]:
    reference_ca = reference_chain.ca_positions()
    target_ca = target_chain.ca_positions()
    common = sorted(set(reference_ca) & set(target_ca))
    if len(common) < 3:
        raise ValueError("Fewer than three common C-alpha positions")
    target_points = np.asarray([target_ca[pos] for pos in common], dtype=float)
    reference_points = np.asarray([reference_ca[pos] for pos in common], dtype=float)
    rotation, translation, rmsd = kabsch_mobile_to_reference(target_points, reference_points)
    aligned: dict[tuple[int, str], np.ndarray] = {}
    for seq_id, residue in target_chain.residues.items():
        for atom_name, atom in residue.atoms.items():
            aligned[(seq_id, atom_name)] = atom.coord @ rotation + translation
    return aligned, rmsd


def build_pair_structure(row: dict[str, Any], structures_dir: Path) -> PairStructure:
    reference_pdb = str(row["reference_pdb_id"]).upper()
    target_pdb = str(row["target_pdb_id"]).upper()
    reference_path = resolve_structure_path(
        row.get("reference_structure_path"), reference_pdb, structures_dir
    )
    target_path = resolve_structure_path(
        row.get("target_structure_path"), target_pdb, structures_dir
    )

    reference_chains = parse_cif_chains(reference_path)
    target_chains = parse_cif_chains(target_path)
    reference_chain = select_chain(
        reference_chains,
        str(row["reference_entity_id"]),
        str(row["reference_chain_label_asym_id"]),
    )
    target_chain = select_chain(
        target_chains,
        str(row["target_entity_id"]),
        str(row["target_chain_label_asym_id"]),
    )

    reference_sequence = "".join(str(row["reference_sequence"]).split()).upper()
    target_sequence = "".join(str(row["target_sequence"]).split()).upper()
    if len(reference_sequence) != len(target_sequence):
        raise ValueError(
            f"Sequence length mismatch: {len(reference_sequence)} vs {len(target_sequence)}"
        )
    mutation_index = int(row["mutation_index_0"])
    mutation_position = mutation_index + 1
    if not (0 <= mutation_index < len(reference_sequence)):
        raise ValueError(f"Mutation index {mutation_index} out of range")
    wt_aa = reference_sequence[mutation_index]
    mt_aa = target_sequence[mutation_index]
    if wt_aa == mt_aa:
        raise ValueError("Reference and target residues are identical at mutation position")

    target_positions, alignment_rmsd = align_target_chain_to_reference(
        reference_chain, target_chain
    )
    mutation_atoms = initialize_mutant_residue(
        reference_chain=reference_chain,
        mutation_position=mutation_position,
        target_aa=mt_aa,
        is_terminal=mutation_position == len(reference_sequence),
    )

    nodes: list[NodeRecord] = []
    for seq_id in sorted(reference_chain.residues):
        if not (1 <= seq_id <= len(reference_sequence)):
            continue
        reference_residue = reference_chain.residues[seq_id]
        residue_wt = reference_sequence[seq_id - 1]
        residue_mt = target_sequence[seq_id - 1]
        if seq_id == mutation_position:
            for atom_name in atom_order(
                residue_mt,
                include_oxt=seq_id == len(reference_sequence),
            ):
                if atom_name not in mutation_atoms:
                    continue
                position, element, is_new = mutation_atoms[atom_name]
                nodes.append(
                    NodeRecord(
                        seq_id=seq_id,
                        atom_name=atom_name,
                        element=element,
                        pos_init=np.asarray(position, dtype=float),
                        wt_aa=residue_wt,
                        mt_aa=residue_mt,
                        is_mutation_site=True,
                        is_new_atom=is_new,
                    )
                )
        else:
            # Keep observed WT heavy atoms outside the mutation site. Missing
            # target atoms are handled by the supervision mask rather than by
            # fabricating coordinates throughout the protein.
            preferred = atom_order(
                residue_mt,
                include_oxt=seq_id == len(reference_sequence),
            )
            ordered_names = [name for name in preferred if name in reference_residue.atoms]
            ordered_names += sorted(set(reference_residue.atoms) - set(ordered_names))
            for atom_name in ordered_names:
                atom = reference_residue.atoms[atom_name]
                if atom.element in {"H", "D"}:
                    continue
                nodes.append(
                    NodeRecord(
                        seq_id=seq_id,
                        atom_name=atom_name,
                        element=atom.element or "OTHER",
                        pos_init=atom.coord.copy(),
                        wt_aa=residue_wt,
                        mt_aa=residue_mt,
                        is_mutation_site=False,
                        is_new_atom=False,
                    )
                )

    if not nodes:
        raise ValueError("No graph nodes constructed")
    if not any(node.is_mutation_site and node.atom_name == "CA" for node in nodes):
        raise ValueError("Mutation-site CA is missing after initialization")

    return PairStructure(
        nodes=nodes,
        target_positions=target_positions,
        target_alignment_rmsd=alignment_rmsd,
        reference_chain=reference_chain,
        target_chain=target_chain,
        mutation_position=mutation_position,
        sequence_length=len(reference_sequence),
    )
