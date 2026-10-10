from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .constants import AA_ORDER, ATOM_NAMES, BACKBONE_ATOMS
from .geometry import rotation_geodesic_loss


# Chemically equivalent side-chain atom labels.  For PHE/TYR, both ring
# swaps must happen together.
_SIMPLE_SYMMETRY = {
    "D": [(("OD1", "OD2"),)],
    "E": [(("OE1", "OE2"),)],
    "R": [(("NH1", "NH2"),)],
    "L": [(("CD1", "CD2"),)],
    "V": [(("CG1", "CG2"),)],
}
_COUPLED_SYMMETRY = {
    "F": [(("CD1", "CD2"), ("CE1", "CE2"))],
    "Y": [(("CD1", "CD2"), ("CE1", "CE2"))],
}
_SIDECHAIN_BACKBONE = set(BACKBONE_ATOMS) | {"OXT"}


@dataclass
class LossWeights:
    # Original losses.  Defaults reproduce the previous training regime.
    coordinate: float = 1.0
    edge_distance: float = 0.2
    clash: float = 0.02
    far_delta: float = 0.0
    frame_rotation: float = 0.0
    frame_translation: float = 0.0
    torsion: float = 0.0

    local_boost: float = 3.0
    local_scale: float = 6.0
    new_atom_boost: float = 1.0
    clash_distance: float = 1.2
    far_radius: float = 12.0

    # Response-focused fine-tuning losses.  They default to zero so old runs
    # are unchanged unless explicitly enabled.
    response_ca: float = 0.0
    mutation_sidechain: float = 0.0

    # Conservative response fine-tuning. These losses directly target the
    # benchmark failure mode: avoid moving low-response backbones, while
    # requiring improvement over Copy-WT on clearly responsive examples.
    mr_margin: float = 0.0
    low_response_anchor: float = 0.0

    mr_radius: float = 8.0
    mr_high_threshold: float = 0.55
    mr_low_threshold: float = 0.32
    mr_temperature: float = 0.05
    mr_improvement_fraction: float = 0.02
    mr_hinge_softness: float = 0.01
    anchor_beta: float = 0.10

    # Native-response weighting for local CA displacement.
    response_radius: float = 8.0
    response_boost: float = 4.0
    response_scale: float = 0.25
    response_max_weight: float = 5.0
    response_beta: float = 0.25

    # Mutation-site side-chain coordinate loss.
    sidechain_beta: float = 0.25


def _smooth_l1_vector(
    predicted: torch.Tensor,
    target: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Smooth-L1 per 3D point, averaged over xyz."""
    return F.smooth_l1_loss(
        predicted,
        target,
        reduction="none",
        beta=float(beta),
    ).mean(dim=-1)


def masked_coordinate_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    distance_to_mutation: torch.Tensor,
    is_new_atom: torch.Tensor,
    local_boost: float,
    local_scale: float,
    new_atom_boost: float,
) -> torch.Tensor:
    if not torch.any(mask):
        return predicted.sum() * 0.0

    error = F.smooth_l1_loss(
        predicted,
        target,
        reduction="none",
    ).mean(dim=-1)

    weights = 1.0 + float(local_boost) * torch.exp(
        -distance_to_mutation / float(local_scale)
    )
    weights = weights + float(new_atom_boost) * is_new_atom.float()

    weights = weights[mask]
    return torch.sum(error[mask] * weights) / weights.sum().clamp_min(1e-8)


def response_weighted_ca_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    initial: torch.Tensor,
    target_mask: torch.Tensor,
    is_ca: torch.Tensor,
    distance_to_mutation: torch.Tensor,
    node_graph: torch.Tensor,
    radius: float,
    boost: float,
    scale: float,
    max_weight: float,
    beta: float,
) -> torch.Tensor:
    """Local CA loss that gives extra weight to atoms that truly moved.

    Note that (predicted-initial) - (target-initial) algebraically equals
    predicted-target.  The useful difference from the ordinary coordinate loss
    is therefore NOT the subtraction itself.  It is:
      * CA-only supervision,
      * a fixed mutation-local neighborhood,
      * per-graph normalization,
      * up-weighting by native displacement magnitude.

    This prevents the many nearly stationary atoms from dominating the gradient.
    """
    mask = (
        target_mask
        & is_ca
        & (distance_to_mutation <= float(radius))
    )

    if not torch.any(mask):
        return predicted.sum() * 0.0

    native_delta = target - initial
    predicted_delta = predicted - initial

    native_magnitude = torch.linalg.norm(
        native_delta,
        dim=-1,
    )

    # Smooth bounded weighting:
    #   response=0   -> weight 1
    #   response=scale -> approximately 1 + boost/2
    #   large response -> approaches 1 + boost, capped by max_weight.
    scale_value = max(float(scale), 1e-6)
    response_weight = 1.0 + float(boost) * (
        native_magnitude
        / (native_magnitude + scale_value)
    )
    response_weight = response_weight.clamp(
        max=float(max_weight)
    )

    point_error = _smooth_l1_vector(
        predicted_delta,
        native_delta,
        beta=float(beta),
    )

    # Equalize graph contribution so a graph with many local CA atoms does not
    # dominate another graph purely because of chain/crop size.
    graph_losses = []
    graph_ids = torch.unique(node_graph[mask])

    for graph_id in graph_ids:
        graph_mask = mask & (node_graph == graph_id)
        weights = response_weight[graph_mask]
        errors = point_error[graph_mask]

        graph_losses.append(
            torch.sum(errors * weights)
            / weights.sum().clamp_min(1e-8)
        )

    if not graph_losses:
        return predicted.sum() * 0.0

    return torch.stack(graph_losses).mean()


def _atom_name(atom_type_value: int) -> str:
    if 0 <= int(atom_type_value) < len(ATOM_NAMES):
        return ATOM_NAMES[int(atom_type_value)]
    return "UNK"


def _mutation_aa(
    res_type_mt: torch.Tensor,
    mutation_indices: torch.Tensor,
) -> str | None:
    if mutation_indices.numel() == 0:
        return None

    aa_id = int(
        res_type_mt[mutation_indices[0]].detach().item()
    )

    if 0 <= aa_id < len(AA_ORDER):
        return AA_ORDER[aa_id]

    return None



def _detached_kabsch_mobile_to_reference(
    mobile: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detached Kabsch transform for row-vector coordinates.

    The transform is recomputed from the current prediction but gradients do
    not flow through the SVD itself. Gradients still flow through the aligned
    predicted coordinates. FP32 is used internally because CUDA SVD does not
    support all low-precision dtypes reliably under AMP.
    """
    with torch.no_grad():
        m = mobile.detach().float()
        r = reference.detach().float()

        cm = m.mean(dim=0)
        cr = r.mean(dim=0)
        x = m - cm
        y = r - cr
        covariance = x.transpose(0, 1) @ y
        u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
        rotation = u @ vh
        if torch.det(rotation) < 0:
            u = u.clone()
            u[:, -1] *= -1.0
            rotation = u @ vh
        translation = cr - cm @ rotation

    return (
        rotation.to(device=mobile.device, dtype=mobile.dtype),
        translation.to(device=mobile.device, dtype=mobile.dtype),
    )


def _aligned_to_reference_detached(
    mobile: torch.Tensor,
    reference: torch.Tensor,
    fit_mask: torch.Tensor,
) -> torch.Tensor:
    rotation, translation = _detached_kabsch_mobile_to_reference(
        mobile[fit_mask],
        reference[fit_mask].to(dtype=mobile.dtype),
    )
    return mobile @ rotation + translation


def mr_margin_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    initial: torch.Tensor,
    target_mask: torch.Tensor,
    is_ca: torch.Tensor,
    distance_to_mutation: torch.Tensor,
    node_graph: torch.Tensor,
    radius: float,
    high_threshold: float,
    temperature: float,
    improvement_fraction: float,
    hinge_softness: float,
) -> torch.Tensor:
    """Soft hinge against the Copy-WT MR-RMSD baseline on responsive graphs.

    The benchmark removes rigid-body motion relative to WT before calculating
    MR-RMSD. This loss mirrors that operation. It gives substantial weight only
    when the native local response is clearly above `high_threshold`.

    Desired model error = (1 - improvement_fraction) * Copy-WT error.
    """
    graph_losses = []
    graph_weights = []
    eps = 1e-8
    tau = max(float(temperature), 1e-6)
    softness = max(float(hinge_softness), 1e-6)

    for graph_id in torch.unique(node_graph):
        graph = node_graph == graph_id
        ca = graph & target_mask & is_ca
        local = ca & (distance_to_mutation <= float(radius))

        if int(ca.sum().detach().item()) < 3 or not torch.any(local):
            continue

        pred_aligned = _aligned_to_reference_detached(
            predicted,
            initial.to(dtype=predicted.dtype),
            ca,
        )
        target_aligned = _aligned_to_reference_detached(
            target.to(dtype=predicted.dtype),
            initial.to(dtype=predicted.dtype),
            ca,
        )

        init_local = initial[local].to(dtype=predicted.dtype)
        model_delta_error = pred_aligned[local] - target_aligned[local]
        native_delta = target_aligned[local] - init_local

        model_rms = torch.sqrt(
            torch.mean(torch.sum(model_delta_error.float() ** 2, dim=-1)) + eps
        )
        copy_rms = torch.sqrt(
            torch.mean(torch.sum(native_delta.float() ** 2, dim=-1)) + eps
        ).detach()

        high_weight = torch.sigmoid(
            (copy_rms - float(high_threshold)) / tau
        ).detach()

        desired = copy_rms * max(0.0, 1.0 - float(improvement_fraction))
        # Smooth ReLU keeps a useful gradient close to the baseline boundary.
        hinge = softness * F.softplus((model_rms - desired) / softness)

        graph_losses.append(hinge * high_weight)
        graph_weights.append(high_weight)

    if not graph_losses:
        return predicted.sum() * 0.0

    losses = torch.stack(graph_losses)
    weights = torch.stack(graph_weights)
    return losses.sum() / weights.sum().clamp_min(1e-6)


def low_response_anchor_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    initial: torch.Tensor,
    target_mask: torch.Tensor,
    is_ca: torch.Tensor,
    distance_to_mutation: torch.Tensor,
    node_graph: torch.Tensor,
    radius: float,
    low_threshold: float,
    temperature: float,
    beta: float,
) -> torch.Tensor:
    """Do-no-harm regularizer for low-response mutation events.

    For graphs where Copy-WT is already very close to the experimental target,
    penalize non-rigid local CA movement away from the initialization. The
    weight decreases smoothly for larger native responses.
    """
    graph_losses = []
    graph_weights = []
    tau = max(float(temperature), 1e-6)

    for graph_id in torch.unique(node_graph):
        graph = node_graph == graph_id
        ca = graph & target_mask & is_ca
        local = ca & (distance_to_mutation <= float(radius))

        if int(ca.sum().detach().item()) < 3 or not torch.any(local):
            continue

        pred_aligned = _aligned_to_reference_detached(
            predicted,
            initial.to(dtype=predicted.dtype),
            ca,
        )
        target_aligned = _aligned_to_reference_detached(
            target.to(dtype=predicted.dtype),
            initial.to(dtype=predicted.dtype),
            ca,
        )

        init_local = initial[local].to(dtype=predicted.dtype)
        native_delta = target_aligned[local] - init_local
        copy_rms = torch.sqrt(
            torch.mean(torch.sum(native_delta.float() ** 2, dim=-1)) + 1e-8
        ).detach()

        low_weight = torch.sigmoid(
            (float(low_threshold) - copy_rms) / tau
        ).detach()

        anchor = F.smooth_l1_loss(
            pred_aligned[local],
            init_local,
            reduction="mean",
            beta=float(beta),
        )

        graph_losses.append(anchor * low_weight)
        graph_weights.append(low_weight)

    if not graph_losses:
        return predicted.sum() * 0.0

    losses = torch.stack(graph_losses)
    weights = torch.stack(graph_weights)
    return losses.sum() / weights.sum().clamp_min(1e-6)


def mutation_sidechain_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    is_mutation_site: torch.Tensor,
    atom_type: torch.Tensor,
    res_type_mt: torch.Tensor,
    node_graph: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Symmetry-aware mutation-site side-chain Smooth-L1 loss.

    Each graph receives equal weight.  For residues with equivalent atom names,
    the lower-loss chemically equivalent assignment is used.
    """
    graph_losses = []

    for graph_id in torch.unique(node_graph):
        graph_nodes = node_graph == graph_id
        mutation_nodes = torch.where(
            graph_nodes
            & is_mutation_site
            & target_mask
        )[0]

        if mutation_nodes.numel() == 0:
            continue

        names = {
            int(index): _atom_name(
                int(atom_type[index].detach().item())
            )
            for index in mutation_nodes
        }

        sidechain_indices = torch.tensor(
            [
                index
                for index in mutation_nodes.tolist()
                if (
                    names[int(index)]
                    not in _SIDECHAIN_BACKBONE
                    and names[int(index)] != "UNK"
                )
            ],
            device=predicted.device,
            dtype=torch.long,
        )

        if sidechain_indices.numel() == 0:
            continue

        identity_loss = F.smooth_l1_loss(
            predicted[sidechain_indices],
            target[sidechain_indices],
            reduction="mean",
            beta=float(beta),
        )

        candidates = [identity_loss]

        aa = _mutation_aa(
            res_type_mt,
            mutation_nodes,
        )

        if aa is not None:
            by_name = {
                names[int(index)]: int(index)
                for index in mutation_nodes.tolist()
                if names[int(index)] != "UNK"
            }

            permutations = []
            permutations.extend(
                _SIMPLE_SYMMETRY.get(aa, [])
            )
            permutations.extend(
                _COUPLED_SYMMETRY.get(aa, [])
            )

            for swaps in permutations:
                # Only use a swapped candidate when every atom involved is
                # observed in the target.  Otherwise the ambiguity is not
                # safely resolvable.
                if not all(
                    a in by_name and b in by_name
                    for a, b in swaps
                ):
                    continue

                target_for_pred = target[
                    sidechain_indices
                ].clone()

                local_lookup = {
                    int(global_index): local_index
                    for local_index, global_index
                    in enumerate(sidechain_indices.tolist())
                }

                valid = True

                for name_a, name_b in swaps:
                    ia = by_name[name_a]
                    ib = by_name[name_b]

                    if (
                        ia not in local_lookup
                        or ib not in local_lookup
                    ):
                        valid = False
                        break

                    la = local_lookup[ia]
                    lb = local_lookup[ib]

                    original_a = target_for_pred[la].clone()
                    original_b = target_for_pred[lb].clone()

                    target_for_pred[la] = original_b
                    target_for_pred[lb] = original_a

                if not valid:
                    continue

                candidates.append(
                    F.smooth_l1_loss(
                        predicted[sidechain_indices],
                        target_for_pred,
                        reduction="mean",
                        beta=float(beta),
                    )
                )

        graph_losses.append(
            torch.stack(candidates).min()
        )

    if not graph_losses:
        return predicted.sum() * 0.0

    return torch.stack(graph_losses).mean()


def edge_distance_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    edge_index: torch.Tensor,
    edge_static: torch.Tensor,
) -> torch.Tensor:
    src, dst = edge_index
    valid = target_mask[src] & target_mask[dst]

    if not torch.any(valid):
        return predicted.sum() * 0.0

    pred_distance = torch.linalg.norm(
        predicted[src] - predicted[dst],
        dim=-1,
    )
    target_distance = torch.linalg.norm(
        target[src] - target[dst],
        dim=-1,
    )

    covalent = edge_static[:, 0]
    weights = 1.0 + 2.0 * covalent

    error = F.smooth_l1_loss(
        pred_distance[valid],
        target_distance[valid],
        reduction="none",
    )

    return (
        torch.sum(error * weights[valid])
        / weights[valid].sum().clamp_min(1e-8)
    )


def clash_loss(
    predicted: torch.Tensor,
    edge_index: torch.Tensor,
    edge_static: torch.Tensor,
    minimum_distance: float,
) -> torch.Tensor:
    src, dst = edge_index
    noncovalent = edge_static[:, 0] < 0.5
    unique = src < dst
    mask = noncovalent & unique

    if not torch.any(mask):
        return predicted.sum() * 0.0

    distances = torch.linalg.norm(
        predicted[src[mask]]
        - predicted[dst[mask]],
        dim=-1,
    )
    penetration = F.relu(
        float(minimum_distance)
        - distances
    )

    return torch.mean(
        penetration * penetration
    )


def far_field_delta_loss(
    predicted: torch.Tensor,
    initial: torch.Tensor,
    distance_to_mutation: torch.Tensor,
    far_radius: float,
) -> torch.Tensor:
    far = (
        distance_to_mutation
        > float(far_radius)
    )

    if not torch.any(far):
        return predicted.sum() * 0.0

    return F.smooth_l1_loss(
        predicted[far],
        initial[far],
    )


def frame_translation_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(mask):
        return predicted.sum() * 0.0

    return F.smooth_l1_loss(
        predicted[mask],
        target[mask],
    )


def torsion_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(mask):
        return predicted.sum() * 0.0

    pred = F.normalize(
        predicted[mask],
        dim=-1,
        eps=1e-8,
    )
    true = F.normalize(
        target[mask],
        dim=-1,
        eps=1e-8,
    )

    return torch.mean(
        1.0
        - torch.sum(
            pred * true,
            dim=-1,
        )
    )


def compute_total_loss(
    outputs: dict[str, torch.Tensor],
    batch,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the configured objective without evaluating disabled auxiliaries.

    The response-focused losses involve per-graph Kabsch fits and symmetry-aware
    side-chain matching.  They are optional and default to zero for the clean
    architecture comparison, so they are skipped entirely when disabled.
    """
    predicted = outputs["pos"]
    zero = predicted.sum() * 0.0
    terms: dict[str, torch.Tensor] = {}

    terms["coordinate"] = masked_coordinate_loss(
        predicted=predicted,
        target=batch.pos_target,
        mask=batch.target_mask,
        distance_to_mutation=batch.distance_to_mutation,
        is_new_atom=batch.is_new_atom,
        local_boost=weights.local_boost,
        local_scale=weights.local_scale,
        new_atom_boost=weights.new_atom_boost,
    )

    terms["response_ca"] = (
        response_weighted_ca_loss(
            predicted=predicted,
            target=batch.pos_target,
            initial=batch.pos_init,
            target_mask=batch.target_mask,
            is_ca=batch.is_ca,
            distance_to_mutation=batch.distance_to_mutation,
            node_graph=batch.node_graph,
            radius=weights.response_radius,
            boost=weights.response_boost,
            scale=weights.response_scale,
            max_weight=weights.response_max_weight,
            beta=weights.response_beta,
        )
        if weights.response_ca != 0.0
        else zero
    )

    terms["mr_margin"] = (
        mr_margin_loss(
            predicted=predicted,
            target=batch.pos_target,
            initial=batch.pos_init,
            target_mask=batch.target_mask,
            is_ca=batch.is_ca,
            distance_to_mutation=batch.distance_to_mutation,
            node_graph=batch.node_graph,
            radius=weights.mr_radius,
            high_threshold=weights.mr_high_threshold,
            temperature=weights.mr_temperature,
            improvement_fraction=weights.mr_improvement_fraction,
            hinge_softness=weights.mr_hinge_softness,
        )
        if weights.mr_margin != 0.0
        else zero
    )

    terms["low_response_anchor"] = (
        low_response_anchor_loss(
            predicted=predicted,
            target=batch.pos_target,
            initial=batch.pos_init,
            target_mask=batch.target_mask,
            is_ca=batch.is_ca,
            distance_to_mutation=batch.distance_to_mutation,
            node_graph=batch.node_graph,
            radius=weights.mr_radius,
            low_threshold=weights.mr_low_threshold,
            temperature=weights.mr_temperature,
            beta=weights.anchor_beta,
        )
        if weights.low_response_anchor != 0.0
        else zero
    )

    terms["mutation_sidechain"] = (
        mutation_sidechain_loss(
            predicted=predicted,
            target=batch.pos_target,
            target_mask=batch.target_mask,
            is_mutation_site=batch.is_mutation_site,
            atom_type=batch.atom_type,
            res_type_mt=batch.res_type_mt,
            node_graph=batch.node_graph,
            beta=weights.sidechain_beta,
        )
        if weights.mutation_sidechain != 0.0
        else zero
    )

    terms["edge_distance"] = edge_distance_loss(
        predicted=predicted,
        target=batch.pos_target,
        target_mask=batch.target_mask,
        edge_index=batch.edge_index,
        edge_static=batch.edge_static,
    )
    terms["clash"] = clash_loss(
        predicted=predicted,
        edge_index=batch.edge_index,
        edge_static=batch.edge_static,
        minimum_distance=weights.clash_distance,
    )
    terms["far_delta"] = (
        far_field_delta_loss(
            predicted=predicted,
            initial=batch.pos_init,
            distance_to_mutation=batch.distance_to_mutation,
            far_radius=weights.far_radius,
        )
        if weights.far_delta != 0.0
        else zero
    )

    terms["frame_rotation"] = zero
    terms["frame_translation"] = zero
    terms["torsion"] = zero

    if "frame_rotation_rel" in outputs and weights.frame_rotation != 0.0:
        terms["frame_rotation"] = rotation_geodesic_loss(
            predicted=outputs["frame_rotation_rel"],
            target=batch.residue_frame_target_rel,
            mask=batch.residue_frame_mask,
        )
    if "frame_translation_local" in outputs and weights.frame_translation != 0.0:
        terms["frame_translation"] = frame_translation_loss(
            predicted=outputs["frame_translation_local"],
            target=batch.residue_translation_target_local,
            mask=batch.residue_frame_mask,
        )
    if "torsion_sincos" in outputs and weights.torsion != 0.0:
        terms["torsion"] = torsion_loss(
            predicted=outputs["torsion_sincos"],
            target=batch.residue_torsion_target,
            mask=batch.residue_torsion_mask,
        )

    total = (
        weights.coordinate * terms["coordinate"]
        + weights.response_ca * terms["response_ca"]
        + weights.mr_margin * terms["mr_margin"]
        + weights.low_response_anchor * terms["low_response_anchor"]
        + weights.mutation_sidechain * terms["mutation_sidechain"]
        + weights.edge_distance * terms["edge_distance"]
        + weights.clash * terms["clash"]
        + weights.far_delta * terms["far_delta"]
        + weights.frame_rotation * terms["frame_rotation"]
        + weights.frame_translation * terms["frame_translation"]
        + weights.torsion * terms["torsion"]
    )
    terms["total"] = total
    return total, terms
