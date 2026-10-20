"""Prior-guided prototype and graph-context survival model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .graph import coordinate_knn_edges


@dataclass
class ModelOutput:
    risk: Tensor
    slide_embedding: Tensor
    prototypes: Tensor
    attention: Tensor
    alignment_loss: Tensor
    orthogonality_loss: Tensor


class GraphConv(nn.Module):
    """Sparse normalized graph convolution without a PyG dependency."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        n = x.shape[0]
        device = x.device
        loops = torch.arange(n, device=device, dtype=torch.long).repeat(2, 1)
        edges = torch.cat((edge_index.to(device=device, dtype=torch.long), loops), dim=1)
        source, target = edges[0], edges[1]
        degree = torch.zeros(n, device=device, dtype=x.dtype)
        degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
        normalizer = degree[source].clamp_min(1).rsqrt() * degree[target].clamp_min(1).rsqrt()
        messages = x[source] * normalizer.unsqueeze(-1)
        aggregate = torch.zeros_like(x)
        aggregate.index_add_(0, target, messages)
        return self.dropout(F.gelu(self.norm(self.linear(aggregate))))


class PriorGuidedSurvival(nn.Module):
    """Soft-prior prototype encoder followed by graph-context readout.

    The first ``num_classes - 1`` classes are the prognostic classes. The last
    class (``Others`` in the supplied configuration) participates in the slide
    representation and confidence entropy but not in prior alignment.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        prototype_dim: int = 512,
        projection_dim: int = 256,
        num_classes: int = 5,
        num_prototypes: int = 8,
        num_gcn_layers: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        if prototype_dim % num_attention_heads != 0:
            raise ValueError("prototype_dim must be divisible by num_attention_heads")
        if num_classes < 2:
            raise ValueError("num_classes must include at least one prognostic and one residual class")
        self.input_dim = input_dim
        self.prototype_dim = prototype_dim
        self.projection_dim = projection_dim
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.epsilon = epsilon

        self.feature_to_proto = nn.Sequential(
            nn.Linear(input_dim, prototype_dim),
            nn.LayerNorm(prototype_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.feature_to_graph = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        graph_layers: list[nn.Module] = []
        for _ in range(num_gcn_layers):
            graph_layers.append(GraphConv(hidden_dim, hidden_dim, dropout))
        self.graph_layers = nn.ModuleList(graph_layers)
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_dim * (num_gcn_layers + 1), prototype_dim),
            nn.LayerNorm(prototype_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.prototype_offsets = nn.Parameter(
            torch.randn(num_classes, num_prototypes, prototype_dim) * 0.02
        )
        self.prototype_refinement = nn.MultiheadAttention(
            prototype_dim, num_attention_heads, dropout=dropout, batch_first=True
        )
        self.prototype_norm = nn.LayerNorm(prototype_dim)
        self.prototype_feedforward = nn.Sequential(
            nn.Linear(prototype_dim, prototype_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prototype_dim * 2, prototype_dim),
        )
        self.fusion = nn.MultiheadAttention(
            prototype_dim, num_attention_heads, dropout=dropout, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(prototype_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.prototype_to_projection = nn.Linear(prototype_dim, projection_dim, bias=False)
        self.risk_head = nn.Linear(prototype_dim, 1, bias=False)

    def _anchors(self, features: Tensor, priors: Tensor) -> Tensor:
        weights = priors.transpose(0, 1).unsqueeze(-1)
        denominator = weights.sum(dim=1) + self.epsilon
        anchors = (weights * features.unsqueeze(0)).sum(dim=1) / denominator
        return self.feature_to_proto(anchors)

    def _refine_prototypes(self, features: Tensor, priors: Tensor) -> Tensor:
        anchors = self._anchors(features, priors)
        prototypes = anchors[:, None, :] + self.prototype_offsets
        prototypes = prototypes.reshape(1, self.num_classes * self.num_prototypes, -1)
        patch_tokens = self.feature_to_proto(features).unsqueeze(0)
        refined, _ = self.prototype_refinement(prototypes, patch_tokens, patch_tokens, need_weights=False)
        refined = self.prototype_norm(prototypes + refined)
        refined = self.prototype_norm(refined + self.prototype_feedforward(refined))
        return refined.squeeze(0)

    def _graph_context(self, features: Tensor, edge_index: Tensor) -> Tensor:
        h = self.feature_to_graph(features)
        levels = [h]
        for layer in self.graph_layers:
            h = layer(h, edge_index)
            levels.append(h)
        return self.context_projection(torch.cat(levels, dim=-1))

    def _regularizers(self, features: Tensor, priors: Tensor, prototypes: Tensor):
        patch_projection = F.normalize(self.projection_head(features), dim=-1)
        grouped = prototypes.reshape(self.num_classes, self.num_prototypes, -1)
        class_projection = F.normalize(self.prototype_to_projection(grouped), dim=-1)
        scores = torch.einsum("nd,ckd->nck", patch_projection, class_projection).mean(dim=-1)
        learned = scores[:, : self.num_classes - 1].softmax(dim=-1)
        confidence_entropy = -(priors * torch.log(priors + self.epsilon)).sum(dim=-1)
        confidence = 1.0 - confidence_entropy / torch.log(
            torch.tensor(float(self.num_classes), device=priors.device)
        )
        target = priors[:, : self.num_classes - 1]
        alignment = -(confidence[:, None] * target * torch.log(learned + self.epsilon)).sum(dim=-1).mean()

        orthogonality_terms = []
        for class_prototypes in grouped[: self.num_classes - 1]:
            gram = class_prototypes @ class_prototypes.transpose(0, 1)
            gram = gram / gram.norm(p="fro").clamp_min(self.epsilon)
            identity = torch.eye(self.num_prototypes, device=gram.device, dtype=gram.dtype)
            orthogonality_terms.append((gram - identity).norm(p="fro"))
        orthogonality = torch.stack(orthogonality_terms).mean()
        return alignment, orthogonality

    def forward(
        self,
        features: Tensor,
        priors: Tensor,
        *,
        coords: Tensor | None = None,
        edge_index: Tensor | None = None,
        return_attention: bool = True,
    ) -> ModelOutput:
        if features.ndim != 2 or priors.ndim != 2:
            raise ValueError("features and priors must have shape [N, D] and [N, C]")
        if features.shape[0] != priors.shape[0] or features.shape[1] != self.input_dim:
            raise ValueError("features and priors have incompatible shapes")
        if edge_index is None:
            if coords is None:
                raise ValueError("coords are required when edge_index is not supplied")
            if features.shape[0] < 2:
                edges_np = torch.empty((2, 0), dtype=torch.long).numpy()
            else:
                edges_np = coordinate_knn_edges(
                    coords.detach().cpu().numpy(), min(8, features.shape[0] - 1)
                )
            edge_index = torch.as_tensor(edges_np, device=features.device, dtype=torch.long)
        prototypes = self._refine_prototypes(features, priors)
        context = self._graph_context(features, edge_index)
        fused, attention = self.fusion(
            prototypes.unsqueeze(0), context.unsqueeze(0), context.unsqueeze(0), need_weights=return_attention,
            average_attn_weights=False,
        )
        fused = self.fusion_norm(prototypes.unsqueeze(0) + fused).squeeze(0)
        slide_embedding = fused.mean(dim=0)
        risk = self.risk_head(slide_embedding).reshape(())
        # The regularizers are defined on the refined tissue prototypes P in the
        # manuscript.  Graph-context fusion is a downstream readout operation
        # and must not silently change the object being regularized.
        alignment, orthogonality = self._regularizers(features, priors, prototypes)
        if return_attention:
            attention = attention.mean(dim=1).squeeze(0)
        else:
            attention = torch.empty(0, device=features.device)
        return ModelOutput(risk, slide_embedding, fused, attention, alignment, orthogonality)
