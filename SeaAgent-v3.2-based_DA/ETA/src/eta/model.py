"""ETA/DPD model and paper-aligned structural ablations."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PostLNCrossBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        residual_ffn: bool,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.residual_ffn = residual_ffn
        self.norm_attention = nn.LayerNorm(hidden_dim)
        if residual_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, hidden_dim),
            )
            self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            query,
            keys,
            keys,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        if self.residual_ffn:
            query = self.norm_attention(query + self.dropout(attended))
            query = self.norm_ffn(query + self.dropout(self.ffn(query)))
        else:
            query = self.norm_attention(attended)
        return query


class DecisionHeads(nn.Module):
    def __init__(self, hidden_dim: int, heads_hidden: int, dropout: float, dual_head: bool = True):
        super().__init__()
        self.admission = nn.Sequential(
            nn.Linear(hidden_dim, heads_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(heads_hidden, 2),
        )
        self.completeness = (
            nn.Sequential(
                nn.Linear(hidden_dim, heads_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(heads_hidden, 2),
            )
            if dual_head
            else None
        )

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        admission = self.admission(feature)
        completeness = self.completeness(feature) if self.completeness is not None else None
        return admission, completeness


class ETACore(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        tokens_per_frame: int,
        stages: int,
        layers_per_stage: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        residual_ffn: bool = True,
    ):
        super().__init__()
        self.tokens_per_frame = max(1, int(tokens_per_frame))
        self.stages = max(1, int(stages))
        self.shared_layers = nn.ModuleList(
            [
                PostLNCrossBlock(
                    hidden_dim,
                    heads,
                    ffn_dim,
                    dropout,
                    residual_ffn=residual_ffn,
                )
                for _ in range(max(1, int(layers_per_stage)))
            ]
        )

    def forward(
        self,
        candidate: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = candidate.unsqueeze(1)
        for _ in range(self.stages):
            for slot in range(history.shape[1]):
                valid = history_mask[:, slot]
                if not bool(valid.any()):
                    continue
                updated = query[valid]
                keys = history[valid, slot]
                for block in self.shared_layers:
                    updated = block(updated, keys)
                next_query = query.clone()
                next_query[valid] = updated
                query = next_query
        return query[:, 0]


class ETADPD(nn.Module):
    """Paper-aligned serial history-conditioned ETA with dual decision heads."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        settings = config["model"]
        self.variant = str(settings.get("variant", "eta"))
        if self.variant not in {"eta", "mlp_cur", "mlp_hist", "sa_dh", "ca_only", "ca_sh"}:
            raise ValueError(f"Unknown ETA variant: {self.variant}")
        self.hidden_dim = int(settings["hidden_dim"])
        self.tokens_per_frame = max(1, int(settings.get("tokens_per_frame", 8)))
        input_dim = int(settings["input_dim"])
        if input_dim % self.tokens_per_frame:
            raise ValueError("input_dim must be divisible by tokens_per_frame")
        self.token_dim = input_dim // self.tokens_per_frame
        self.candidate_projection = nn.Sequential(
            nn.Linear(int(settings["input_dim"]), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(settings["dropout"])),
        )
        self.token_projection = nn.Sequential(
            nn.Linear(self.token_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(settings["dropout"])),
        )
        self.core = ETACore(
            hidden_dim=self.hidden_dim,
            tokens_per_frame=self.tokens_per_frame,
            stages=int(settings["stages"]),
            layers_per_stage=int(settings["layers_per_stage"]),
            heads=int(settings["attention_heads"]),
            ffn_dim=int(settings["ffn_dim"]),
            dropout=float(settings["dropout"]),
            residual_ffn=self.variant != "ca_only",
        )
        self.self_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(settings["attention_heads"]),
                dim_feedforward=int(settings["ffn_dim"]),
                dropout=float(settings["dropout"]),
                batch_first=True,
                norm_first=False,
            ),
            num_layers=1,
        )
        self.history_norm = nn.LayerNorm(self.hidden_dim)
        self.heads = DecisionHeads(
            self.hidden_dim,
            int(settings["heads_hidden"]),
            float(settings["dropout"]),
            dual_head=self.variant != "ca_sh",
        )
        self.apply(self._initialize_parameters)

    @staticmethod
    def _initialize_parameters(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _project_history(self, history: torch.Tensor) -> torch.Tensor:
        batch, slots, dimension = history.shape
        if dimension != self.tokens_per_frame * self.token_dim:
            raise ValueError("Unexpected history feature dimension")
        tokens = history.reshape(
            batch,
            slots,
            self.tokens_per_frame,
            self.token_dim,
        )
        projected = self.token_projection(tokens.reshape(batch * slots * self.tokens_per_frame, -1))
        return projected.reshape(batch, slots, self.tokens_per_frame, self.hidden_dim)

    def _history_summary(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        count = mask.sum(dim=1).clamp_min(1).to(history.dtype).unsqueeze(1)
        return (history * mask.unsqueeze(-1)).sum(dim=1) / count

    def forward(
        self,
        candidate: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        projected_candidate = self.candidate_projection(candidate)
        if self.variant == "mlp_cur":
            feature = projected_candidate
        else:
            projected_history = self._project_history(history)
        if self.variant == "mlp_hist":
            history_summary = self._history_summary(projected_history.mean(dim=2), history_mask)
            feature = projected_candidate + history_summary
        elif self.variant == "sa_dh":
            tokens = projected_history.reshape(
                candidate.shape[0],
                history.shape[1] * self.tokens_per_frame,
                -1,
            )
            token_mask = history_mask.repeat_interleave(self.tokens_per_frame, dim=1)
            sequence = torch.cat([projected_candidate.unsqueeze(1), tokens], dim=1)
            padding = torch.cat(
                [
                    torch.zeros((candidate.shape[0], 1), dtype=torch.bool, device=candidate.device),
                    ~token_mask,
                ],
                dim=1,
            )
            encoded = self.self_attention(sequence, src_key_padding_mask=padding)
            feature = encoded[:, 0]
        elif self.variant not in {"mlp_cur", "mlp_hist", "sa_dh"}:
            feature = self.core(projected_candidate, projected_history, history_mask)
        feature = self.history_norm(feature)
        return self.heads(feature)


def build_model(config: dict[str, Any]) -> ETADPD:
    return ETADPD(config)
