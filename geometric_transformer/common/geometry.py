from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def normalize_np(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm


def kabsch_mobile_to_reference(
    mobile: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return R, t such that ``mobile @ R + t`` best matches reference."""
    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(
            f"Expected matching Nx3 arrays, got {mobile.shape} and {reference.shape}"
        )
    if len(mobile) < 3:
        raise ValueError("At least three points are required for Kabsch alignment")

    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    x = mobile - mobile_center
    y = reference - reference_center
    covariance = x.T @ y
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    translation = reference_center - mobile_center @ rotation
    aligned = mobile @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=1))))
    return rotation, translation, rmsd


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float) @ rotation + translation


def residue_frame_np(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Construct a right-handed frame with columns [x, y, z] and origin at CA."""
    x_axis = normalize_np(np.asarray(c, dtype=float) - np.asarray(ca, dtype=float))
    n_direction = normalize_np(np.asarray(n, dtype=float) - np.asarray(ca, dtype=float))
    z_axis = normalize_np(np.cross(x_axis, n_direction))
    y_axis = normalize_np(np.cross(z_axis, x_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def dihedral_np(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in (p0, p1, p2, p3)]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = normalize_np(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert Zhou et al. 6D rotation representation to matrices.

    The output matrices store basis vectors in columns.
    """
    a1 = rotation_6d[..., 0:3]
    a2 = rotation_6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    a2_orthogonal = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2_orthogonal, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rotation_geodesic_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.numel() == 0 or not torch.any(mask):
        return predicted.sum() * 0.0
    pred = predicted[mask]
    true = target[mask]
    relative = pred.transpose(-1, -2) @ true
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cosine)
    return torch.mean(angle * angle)


def apply_residue_frame_transform(
    positions: torch.Tensor,
    node_residue: torch.Tensor,
    frame_init: torch.Tensor,
    ca_init: torch.Tensor,
    relative_rotation: torch.Tensor,
    translation_local: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply an equivariant residue-wise rigid transform.

    ``relative_rotation`` and ``translation_local`` are expressed in each
    residue's input local frame, so the operation is equivariant to a global
    rotation/translation of the protein.
    """
    result = positions.clone()
    valid_node = frame_mask[node_residue]
    if not torch.any(valid_node):
        return result

    ridx = node_residue[valid_node]
    r0 = frame_init[ridx]
    origin = ca_init[ridx]
    q = relative_rotation[ridx]
    t_local = translation_local[ridx]

    centered = positions[valid_node] - origin
    local = torch.einsum("nij,nj->ni", r0.transpose(-1, -2), centered)
    transformed_local = torch.einsum("nij,nj->ni", q, local) + t_local
    transformed_global = origin + torch.einsum("nij,nj->ni", r0, transformed_local)
    result[valid_node] = transformed_global
    return result


def masked_rmsd_no_alignment(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(mask):
        return predicted.sum() * 0.0
    squared = torch.sum((predicted[mask] - target[mask]) ** 2, dim=-1)
    return torch.sqrt(torch.mean(squared).clamp_min(1e-12))
