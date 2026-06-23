# -*- coding: utf-8 -*-
"""
Trainable feature-level fusion modules for multimodal AutoRG-Brain features.

Expected input shape:
    image_hidden_states: [B, M, P, D]

where:
    B = batch size
    M = number of MRI modalities, usually 4: T1, T1ce, T2, FLAIR
    P = number of visual/region tokens, usually 128
    D = decoder hidden dimension, usually 1024

Output shape:
    fused_image_hidden_states: [B, P, D]

This file is intentionally independent from language_model_patchwise_fused.py.
You can first import one of these modules into the fused language model and
replace the current mean(dim=1) baseline in _prepare_fused_image_hidden_states.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_multimodal_feature_shape(
        image_hidden_states: torch.Tensor,
        expected_modalities: Optional[int],
        expected_patch_tokens: Optional[int],
        expected_hidden_dim: Optional[int],
) -> Tuple[int, int, int, int]:
    if image_hidden_states is None:
        raise ValueError("image_hidden_states cannot be None.")

    if image_hidden_states.dim() != 4:
        raise ValueError(
            "Trainable fusion expects image_hidden_states with shape [B, M, P, D]. "
            f"Got shape {tuple(image_hidden_states.shape)}."
        )

    batch_size, num_modalities, num_patch_tokens, hidden_dim = image_hidden_states.shape

    if expected_modalities is not None and num_modalities != expected_modalities:
        raise ValueError(
            f"Expected {expected_modalities} modalities, but got {num_modalities}."
        )

    if expected_patch_tokens is not None and num_patch_tokens != expected_patch_tokens:
        raise ValueError(
            f"Expected {expected_patch_tokens} patch tokens, but got {num_patch_tokens}."
        )

    if expected_hidden_dim is not None and hidden_dim != expected_hidden_dim:
        raise ValueError(
            f"Expected hidden dim {expected_hidden_dim}, but got {hidden_dim}."
        )

    return batch_size, num_modalities, num_patch_tokens, hidden_dim


class LearnableWeightedMeanFusion(nn.Module):
    """
    Minimal trainable upgrade over mean fusion.

    It learns one global scalar weight per modality:
        [B, M, P, D] -> softmax([M]) -> [B, P, D]

    This is a strong first baseline because it starts close to mean fusion when
    modality_logits are initialized to zero.
    """

    def __init__(
            self,
            num_modalities: int = 4,
            num_patch_tokens: int = 128,
            hidden_dim: int = 1024,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_patch_tokens = num_patch_tokens
        self.hidden_dim = hidden_dim
        self.modality_logits = nn.Parameter(torch.zeros(num_modalities))

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        _check_multimodal_feature_shape(
            image_hidden_states,
            self.num_modalities,
            self.num_patch_tokens,
            self.hidden_dim,
        )
        weights = F.softmax(self.modality_logits, dim=0)
        return (image_hidden_states * weights.view(1, -1, 1, 1)).sum(dim=1)

    def get_last_fusion_weights(self) -> torch.Tensor:
        return F.softmax(self.modality_logits.detach(), dim=0)


class TokenwiseGatedFusion(nn.Module):
    """
    Recommended simple trainable fusion module.

    It predicts a modality gate for every visual token:
        gate:  [B, M, P, 1]
        fused: [B, P, D]

    This lets different region tokens rely on different modalities. For example,
    enhancement-related tokens may put more weight on T1ce, while edema-related
    tokens may put more weight on FLAIR/T2.
    """

    def __init__(
            self,
            num_modalities: int = 4,
            num_patch_tokens: int = 128,
            hidden_dim: int = 1024,
            gate_hidden_dim: int = 256,
            dropout: float = 0.1,
            use_residual: bool = True,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_patch_tokens = num_patch_tokens
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.last_gate_weights = None

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        _check_multimodal_feature_shape(
            image_hidden_states,
            self.num_modalities,
            self.num_patch_tokens,
            self.hidden_dim,
        )

        gate_logits = self.gate_mlp(image_hidden_states)  # [B, M, P, 1]
        gate_weights = F.softmax(gate_logits, dim=1)
        fused = (image_hidden_states * gate_weights).sum(dim=1)

        if self.use_residual:
            fused = fused + image_hidden_states.mean(dim=1)

        fused = self.output_norm(fused)
        self.last_gate_weights = gate_weights.detach()
        return fused

    def get_last_fusion_weights(self) -> Optional[torch.Tensor]:
        return self.last_gate_weights


class ConcatProjectionFusion(nn.Module):
    """
    More expressive but heavier fusion baseline.

    It concatenates modalities along the hidden dimension for each token:
        [B, M, P, D] -> [B, P, M*D] -> Linear(M*D, D)
    """

    def __init__(
            self,
            num_modalities: int = 4,
            num_patch_tokens: int = 128,
            hidden_dim: int = 1024,
            dropout: float = 0.1,
            use_residual: bool = True,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_patch_tokens = num_patch_tokens
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        self.proj = nn.Sequential(
            nn.LayerNorm(num_modalities * hidden_dim),
            nn.Linear(num_modalities * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_modalities, num_patch_tokens, hidden_dim = _check_multimodal_feature_shape(
            image_hidden_states,
            self.num_modalities,
            self.num_patch_tokens,
            self.hidden_dim,
        )

        x = image_hidden_states.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, num_patch_tokens, num_modalities * hidden_dim)
        fused = self.proj(x)

        if self.use_residual:
            fused = fused + image_hidden_states.mean(dim=1)

        return self.output_norm(fused)


def build_multimodal_fusion_module(
        fusion_type: str = "token_gated",
        num_modalities: int = 4,
        num_patch_tokens: int = 128,
        hidden_dim: int = 1024,
        gate_hidden_dim: int = 256,
        dropout: float = 0.1,
) -> nn.Module:
    """
    Factory used by the fused language model or training script.

    Supported fusion_type:
        - "weighted_mean": global trainable modality weights
        - "token_gated": per-token modality gating, recommended first choice
        - "concat_projection": concatenate modalities then project to hidden_dim
    """
    fusion_type = fusion_type.lower()

    if fusion_type == "weighted_mean":
        return LearnableWeightedMeanFusion(
            num_modalities=num_modalities,
            num_patch_tokens=num_patch_tokens,
            hidden_dim=hidden_dim,
        )

    if fusion_type == "token_gated":
        return TokenwiseGatedFusion(
            num_modalities=num_modalities,
            num_patch_tokens=num_patch_tokens,
            hidden_dim=hidden_dim,
            gate_hidden_dim=gate_hidden_dim,
            dropout=dropout,
        )

    if fusion_type == "concat_projection":
        return ConcatProjectionFusion(
            num_modalities=num_modalities,
            num_patch_tokens=num_patch_tokens,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    raise ValueError(
        f"Unsupported fusion_type: {fusion_type}. "
        "Choose from: weighted_mean, token_gated, concat_projection."
    )


if __name__ == "__main__":
    features = torch.randn(2, 4, 128, 1024)
    for name in ["weighted_mean", "token_gated", "concat_projection"]:
        module = build_multimodal_fusion_module(name)
        fused_features = module(features)
        print(name, "input:", tuple(features.shape), "output:", tuple(fused_features.shape))
