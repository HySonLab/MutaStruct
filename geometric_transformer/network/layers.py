from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.constants import AA_ORDER, ATOM_NAMES, ELEMENTS


def scatter_sum(
    values: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """Sum edge values into nodes along the first tensor dimension."""
    output = torch.zeros(
        (int(dim_size),) + tuple(values.shape[1:]),
        device=values.device,
        dtype=values.dtype,
    )
    output.index_add_(0, index, values)
    return output


def segment_softmax_multihead(
    logits: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Softmax independently for every destination-node/head segment.

    Args:
        logits: ``[E, H]`` attention logits.
        index: ``[E]`` destination node for each directed edge.
        dim_size: number of nodes.

    The reduction is accumulated in FP32 under autocast, then converted back to
    the input dtype.  This avoids the Half/Float ``index_add_`` mismatch that
    can otherwise appear under CUDA AMP.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits [E,H], received {tuple(logits.shape)}")

    output_dtype = logits.dtype
    work = logits.float()
    num_heads = int(work.shape[1])

    head_ids = torch.arange(
        num_heads,
        device=index.device,
        dtype=index.dtype,
    )
    composite = index.unsqueeze(1) * num_heads + head_ids.unsqueeze(0)
    composite_flat = composite.reshape(-1)
    work_flat = work.reshape(-1)
    total_segments = int(dim_size) * num_heads

    maxima = torch.full(
        (total_segments,),
        -torch.inf,
        device=work.device,
        dtype=work.dtype,
    )
    maxima.scatter_reduce_(
        0,
        composite_flat,
        work_flat,
        reduce="amax",
        include_self=True,
    )

    stabilized = work_flat - maxima[composite_flat]
    exponentials = torch.exp(stabilized)

    denominator = torch.zeros(
        total_segments,
        device=work.device,
        dtype=work.dtype,
    )
    denominator.index_add_(0, composite_flat, exponentials)

    probabilities = exponentials / denominator[composite_flat].clamp_min(eps)
    return probabilities.reshape_as(work).to(dtype=output_dtype)


class SinusoidalPositionalEmbedding(nn.Module):
    """Qin et al.-style sinusoidal embedding for continuous geometry indices."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if int(embedding_dim) % 2 != 0:
            raise ValueError(
                "Sinusoidal positional embedding requires an even dimension; "
                f"received {embedding_dim}."
            )

        self.embedding_dim = int(embedding_dim)
        indices = torch.arange(0, self.embedding_dim, 2).float()
        divisor = torch.exp(
            indices * (-math.log(10000.0) / float(self.embedding_dim))
        )
        self.register_buffer("divisor", divisor)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        original_shape = tuple(indices.shape)
        omega = (
            indices.float().reshape(-1, 1, 1)
            * self.divisor.reshape(1, -1, 1)
        )
        embeddings = torch.cat([torch.sin(omega), torch.cos(omega)], dim=-1)
        embeddings = embeddings.reshape(*original_shape, self.embedding_dim)
        # Qin et al. detach the geometry indices from coordinate gradients.
        return embeddings.detach()


@torch.no_grad()
def build_knn_by_graph(
    points: torch.Tensor,
    graph_ptr: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K nearest atom indices for every node, independently within each graph.

    The neighbor identities are discrete and therefore intentionally computed
    without autograd, matching the original GeoTransformer treatment of KNN
    indices.  Invalid padding entries point back to the query node and are
    masked by the returned boolean tensor.
    """
    num_nodes = int(points.shape[0])
    k = int(k)
    if k < 1:
        raise ValueError("angle_k must be at least 1")

    neighbors = torch.arange(
        num_nodes,
        device=points.device,
        dtype=torch.long,
    ).unsqueeze(1).expand(num_nodes, k).clone()
    valid = torch.zeros((num_nodes, k), device=points.device, dtype=torch.bool)

    ptr = graph_ptr.detach().cpu().tolist()
    for start, end in zip(ptr[:-1], ptr[1:]):
        start = int(start)
        end = int(end)
        n = end - start
        if n <= 1:
            continue

        effective_k = min(k, n - 1)
        graph_points = points[start:end].detach().float()
        distance = torch.cdist(graph_points, graph_points)
        distance.fill_diagonal_(torch.inf)
        local_indices = torch.topk(
            distance,
            k=effective_k,
            dim=1,
            largest=False,
        ).indices

        neighbors[start:end, :effective_k] = local_indices + start
        valid[start:end, :effective_k] = True

    return neighbors, valid


class SparseGeometricStructureEmbedding(nn.Module):
    """Sparse adaptation of Qin et al.'s geometric structure embedding.

    For every directed graph edge ``j -> i`` it encodes:

    * the pairwise distance ``||x_j - x_i||``;
    * the K triplet angles ``angle(k-i, j-i)`` where ``k`` is a nearest
      neighbor of the query node ``i``.

    Distance and angle values are converted to sinusoidal embeddings, projected
    separately and summed.  The only departure from the original dense point
    cloud implementation is that embeddings are constructed for sparse graph
    edges rather than all ``N^2`` point pairs.  The four existing biochemical
    edge flags are added through a learned projection.
    """

    def __init__(
        self,
        hidden_dim: int,
        sigma_distance: float,
        sigma_angle_degrees: float,
        angle_k: int,
        edge_static_dim: int = 4,
        angle_reduction: str = "max",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if sigma_distance <= 0:
            raise ValueError("sigma_distance must be positive")
        if sigma_angle_degrees <= 0:
            raise ValueError("sigma_angle_degrees must be positive")
        if angle_reduction not in {"max", "mean"}:
            raise ValueError("angle_reduction must be 'max' or 'mean'")

        self.hidden_dim = int(hidden_dim)
        self.sigma_distance = float(sigma_distance)
        self.sigma_angle_degrees = float(sigma_angle_degrees)
        self.angle_factor = 180.0 / (
            self.sigma_angle_degrees * math.pi
        )
        self.angle_k = int(angle_k)
        self.angle_reduction = angle_reduction

        self.embedding = SinusoidalPositionalEmbedding(self.hidden_dim)
        self.distance_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.angle_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.edge_static_projection = nn.Linear(
            int(edge_static_dim), self.hidden_dim, bias=False
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    @torch.no_grad()
    def geometry_indices(
        self,
        points: torch.Tensor,
        edge_index: torch.Tensor,
        graph_ptr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source, destination = edge_index
        points_fp32 = points.detach().float()

        pair_vector = points_fp32[source] - points_fp32[destination]
        pair_distance = torch.linalg.norm(pair_vector, dim=-1)
        distance_indices = pair_distance / self.sigma_distance

        knn_indices, knn_valid = build_knn_by_graph(
            points_fp32,
            graph_ptr,
            self.angle_k,
        )
        edge_knn = knn_indices[destination]
        edge_knn_valid = knn_valid[destination]

        reference_vectors = (
            points_fp32[edge_knn] - points_fp32[destination].unsqueeze(1)
        )
        anchor_vectors = pair_vector.unsqueeze(1).expand_as(reference_vectors)

        sine = torch.linalg.norm(
            torch.cross(reference_vectors, anchor_vectors, dim=-1),
            dim=-1,
        )
        cosine = torch.sum(reference_vectors * anchor_vectors, dim=-1)
        angles = torch.atan2(sine, cosine)
        angle_indices = angles * self.angle_factor

        return distance_indices, angle_indices, edge_knn_valid

    def forward(
        self,
        points: torch.Tensor,
        edge_index: torch.Tensor,
        edge_static: torch.Tensor,
        graph_ptr: torch.Tensor,
    ) -> torch.Tensor:
        distance_indices, angle_indices, angle_valid = self.geometry_indices(
            points,
            edge_index,
            graph_ptr,
        )

        distance_embedding = self.distance_projection(
            self.embedding(distance_indices)
        )
        angle_embedding = self.angle_projection(self.embedding(angle_indices))

        if self.angle_reduction == "max":
            minimum = torch.finfo(angle_embedding.dtype).min
            masked = torch.where(
                angle_valid.unsqueeze(-1),
                angle_embedding,
                torch.full_like(angle_embedding, minimum),
            )
            angle_embedding = masked.max(dim=1).values
            no_valid = ~angle_valid.any(dim=1)
            if torch.any(no_valid):
                angle_embedding = angle_embedding.clone()
                angle_embedding[no_valid] = 0.0
        else:
            mask = angle_valid.unsqueeze(-1).to(angle_embedding.dtype)
            angle_embedding = (angle_embedding * mask).sum(dim=1) / mask.sum(
                dim=1
            ).clamp_min(1.0)

        embedding = (
            distance_embedding
            + angle_embedding
            + self.edge_static_projection(edge_static)
        )
        return self.dropout(self.output_norm(embedding))


class ProteinMutationNodeEncoder(nn.Module):
    """Protein-specific invariant atom features for the GeoTransformer."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        atom_dim = 32
        element_dim = 8
        residue_dim = 32
        mutation_pair_dim = 24
        num_residue_types = len(AA_ORDER) + 1

        self.atom_embedding = nn.Embedding(len(ATOM_NAMES) + 1, atom_dim)
        self.element_embedding = nn.Embedding(len(ELEMENTS), element_dim)
        self.residue_embedding = nn.Embedding(
            num_residue_types, residue_dim
        )
        self.mutation_pair_embedding = nn.Embedding(
            num_residue_types * num_residue_types,
            mutation_pair_dim,
        )

        input_dim = (
            atom_dim
            + element_dim
            + 2 * residue_dim
            + mutation_pair_dim
            + 5
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.num_residue_types = num_residue_types

    def forward(self, batch) -> torch.Tensor:
        pair_id = (
            batch.res_type_wt * self.num_residue_types + batch.res_type_mt
        )
        continuous = torch.cat(
            [
                batch.sequence_position_norm,
                batch.is_mutation_site.float().unsqueeze(-1),
                batch.is_new_atom.float().unsqueeze(-1),
                batch.is_ca.float().unsqueeze(-1),
                torch.exp(-batch.distance_to_mutation / 8.0).unsqueeze(-1),
            ],
            dim=-1,
        )
        features = torch.cat(
            [
                self.atom_embedding(batch.atom_type),
                self.element_embedding(batch.element_type),
                self.residue_embedding(batch.res_type_wt),
                self.residue_embedding(batch.res_type_mt),
                self.mutation_pair_embedding(pair_id),
                continuous,
            ],
            dim=-1,
        )
        return self.network(features)


class SparseRPEMultiHeadAttention(nn.Module):
    """Qin-style relative-position attention on sparse directed edges.

    The attention score for edge ``j -> i`` is

    ``q_i^T k_j + q_i^T p_ij``

    where ``p_ij`` is the projected distance-plus-angle embedding.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by heads={num_heads}"
            )

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads

        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.position_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        node_state: torch.Tensor,
        edge_embedding: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, destination = edge_index
        num_nodes = int(node_state.shape[0])
        num_edges = int(source.shape[0])

        query = self.query_projection(node_state).reshape(
            num_nodes, self.num_heads, self.head_dim
        )
        key = self.key_projection(node_state).reshape(
            num_nodes, self.num_heads, self.head_dim
        )
        value = self.value_projection(node_state).reshape(
            num_nodes, self.num_heads, self.head_dim
        )
        position = self.position_projection(edge_embedding).reshape(
            num_edges, self.num_heads, self.head_dim
        )

        query_edge = query[destination]
        key_edge = key[source]

        logits = (
            torch.sum(query_edge * key_edge, dim=-1)
            + torch.sum(query_edge * position, dim=-1)
        ) / math.sqrt(float(self.head_dim))

        attention = segment_softmax_multihead(
            logits,
            destination,
            dim_size=num_nodes,
        )
        attention = self.dropout(attention)

        message = attention.unsqueeze(-1) * value[source]
        aggregated = scatter_sum(message, destination, num_nodes).reshape(
            num_nodes, self.hidden_dim
        )
        return self.output_projection(aggregated), attention


class SparseRPETransformerBlock(nn.Module):
    """Sparse self-attention block with Qin-style geometric RPE."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.attention = SparseRPEMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.attention_dropout = nn.Dropout(float(dropout))
        self.attention_norm = nn.LayerNorm(hidden_dim)

        inner_dim = int(hidden_dim) * int(ffn_multiplier)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(float(dropout)),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_state: torch.Tensor,
        edge_embedding: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update, attention = self.attention(
            node_state,
            edge_embedding,
            edge_index,
        )
        node_state = self.attention_norm(
            node_state + self.attention_dropout(update)
        )
        node_state = self.ffn_norm(node_state + self.ffn(node_state))
        return node_state, attention


class GaussianRBF(nn.Module):
    def __init__(self, num_basis: int, cutoff: float) -> None:
        super().__init__()
        centers = torch.linspace(0.0, float(cutoff), int(num_basis))
        spacing = (
            float(centers[1] - centers[0])
            if int(num_basis) > 1
            else float(cutoff)
        )
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / max(spacing * spacing, 1e-6)
        self.cutoff = float(cutoff)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.unsqueeze(-1)
        values = torch.exp(-self.gamma * (d - self.centers) ** 2)
        envelope = 0.5 * (
            torch.cos(math.pi * d / self.cutoff).clamp(-1.0, 1.0) + 1.0
        )
        envelope = torch.where(d <= self.cutoff, envelope, torch.zeros_like(d))
        return values * envelope


class EquivariantCoordinateDecoderBlock(nn.Module):
    """Invariant scalar messages followed by E(3)-equivariant coordinate motion."""

    def __init__(
        self,
        hidden_dim: int,
        edge_static_dim: int,
        dropout: float,
        distance_cutoff: float,
        num_rbf: int,
        coordinate_scale: float,
        movement_gate_bias: float,
    ) -> None:
        super().__init__()
        self.rbf = GaussianRBF(num_basis=num_rbf, cutoff=distance_cutoff)

        edge_input_dim = (
            hidden_dim * 3 + int(num_rbf) + int(edge_static_dim)
        )
        self.edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.attention_logit = nn.Linear(hidden_dim, 1)
        self.node_message = nn.Linear(hidden_dim, hidden_dim)
        self.coordinate_gate = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.coordinate_gate.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.coordinate_gate.bias)

        self.movement_gate = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.movement_gate.weight)
        nn.init.constant_(self.movement_gate.bias, float(movement_gate_bias))

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.node_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        initial_scale = min(max(float(coordinate_scale), 1e-4), 1.0 - 1e-4)
        self.coordinate_scale_logit = nn.Parameter(
            torch.tensor(math.log(initial_scale / (1.0 - initial_scale)))
        )

    def forward(
        self,
        node_state: torch.Tensor,
        coordinates: torch.Tensor,
        edge_embedding: torch.Tensor,
        edge_index: torch.Tensor,
        edge_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source, destination = edge_index
        relative = coordinates[source] - coordinates[destination]
        distance = torch.linalg.norm(relative, dim=-1).clamp_min(1e-8)

        edge_state = self.edge_network(
            torch.cat(
                [
                    node_state[source],
                    node_state[destination],
                    edge_embedding,
                    self.rbf(distance),
                    edge_static,
                ],
                dim=-1,
            )
        )

        logits = self.attention_logit(edge_state).squeeze(-1)
        # Reuse the multi-head implementation with one head.
        attention = segment_softmax_multihead(
            logits.unsqueeze(-1),
            destination,
            dim_size=node_state.shape[0],
        ).squeeze(-1)

        scalar_message = self.node_message(edge_state) * attention.unsqueeze(-1)
        aggregated = scatter_sum(
            scalar_message,
            destination,
            node_state.shape[0],
        )
        node_state = self.node_norm(node_state + aggregated)
        node_state = node_state + self.node_ffn(node_state)

        edge_gate = torch.tanh(self.coordinate_gate(edge_state)).squeeze(-1)
        bounded_vector = relative / (1.0 + distance.unsqueeze(-1))
        vector_message = (
            attention.unsqueeze(-1)
            * edge_gate.unsqueeze(-1)
            * bounded_vector
        )
        displacement = scatter_sum(
            vector_message,
            destination,
            coordinates.shape[0],
        )

        node_gate = torch.sigmoid(self.movement_gate(node_state))
        scale = torch.sigmoid(self.coordinate_scale_logit)
        coordinates = coordinates + scale * node_gate * displacement
        return node_state, coordinates, node_gate.squeeze(-1)


class EquivariantCoordinateDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        distance_cutoff: float,
        num_rbf: int,
        coordinate_scale: float,
        movement_gate_bias: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EquivariantCoordinateDecoderBlock(
                    hidden_dim=hidden_dim,
                    edge_static_dim=4,
                    dropout=dropout,
                    distance_cutoff=distance_cutoff,
                    num_rbf=num_rbf,
                    coordinate_scale=coordinate_scale,
                    movement_gate_bias=movement_gate_bias,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        node_state: torch.Tensor,
        coordinates: torch.Tensor,
        edge_embedding: torch.Tensor,
        edge_index: torch.Tensor,
        edge_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        movement_gate = torch.zeros(
            coordinates.shape[0],
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        for layer in self.layers:
            node_state, coordinates, movement_gate = layer(
                node_state,
                coordinates,
                edge_embedding,
                edge_index,
                edge_static,
            )
        return node_state, coordinates, movement_gate
