from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext

import torch
from torch import nn

from .actor_variants import (
    actor_variant,
    actor_variant_ids,
    entity_tensor_prefix,
)

CRITIC_MODES = frozenset({"local", "centralized_team"})


def mlp_body(input_size, hidden_size, activation=nn.SiLU):
    return nn.Sequential(nn.Linear(input_size, hidden_size), activation(),
                         nn.Linear(hidden_size, hidden_size), activation())


def _ownship_attention(attention, query, tokens, padding_mask):
    # Measured on the small CUDA/FP32 ownship query; other attention stays default.
    from torch.nn.attention import sdpa_kernel, SDPBackend
    context = (sdpa_kernel(SDPBackend.MATH)
               if query.device.type == 'cuda' and query.dtype == torch.float32
               and query.shape[1] == 1 and 0 < tokens.shape[1] <= 20 else nullcontext())
    with context:
        return attention(query, tokens, tokens, key_padding_mask=padding_mask, need_weights=False)


class TanhNormal:
    """Small reparameterized tanh-normal distribution used by the PPO baseline.

    Keeping this local avoids importing TorchRL's optional C++ extension on
    Windows while preserving the exact action and log-probability contract the
    trainer needs.
    """

    def __init__(self, location: torch.Tensor, scale: torch.Tensor):
        self.loc = location
        self.scale = scale
        self._normal = torch.distributions.Normal(location, scale)

    def rsample(self) -> torch.Tensor:
        return self._normal.rsample().tanh()

    def from_standard_noise(self, noise: torch.Tensor) -> torch.Tensor:
        return (self.loc + self.scale * noise).tanh()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        epsilon = torch.finfo(value.dtype).eps
        bounded = value.clamp(-1.0 + epsilon, 1.0 - epsilon)
        pre_tanh = torch.atanh(bounded)
        log_base = self._normal.log_prob(pre_tanh)
        log_jacobian = 2.0 * (
            torch.log(torch.tensor(2.0, dtype=value.dtype, device=value.device))
            - pre_tanh
            - torch.nn.functional.softplus(-2.0 * pre_tanh)
        )
        return (log_base - log_jacobian).sum(dim=-1)

    @property
    def deterministic_sample(self) -> torch.Tensor:
        return self.loc.tanh()


class TruncatedNormal:
    """Reparameterized Gaussian conditioned on every action lying in [-1, 1]."""

    def __init__(self, location: torch.Tensor, scale: torch.Tensor):
        self.loc = location
        self.scale = scale
        self._normal = torch.distributions.Normal(location, scale)

    def _standardized_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        location = self.loc.to(torch.float64)
        scale = self.scale.to(torch.float64)
        return (-1.0 - location) / scale, (1.0 - location) / scale

    def _probability_mass(self) -> torch.Tensor:
        lower, upper = self._standardized_bounds()
        direct = torch.special.ndtr(upper) - torch.special.ndtr(lower)
        survival = torch.special.ndtr(-lower) - torch.special.ndtr(-upper)
        mass = torch.where(lower >= 0.0, survival, direct)
        return mass.clamp_min(torch.finfo(torch.float64).tiny)

    def rsample(self) -> torch.Tensor:
        return self._from_uniform(torch.rand_like(self.loc, dtype=torch.float64))

    def from_standard_noise(self, noise: torch.Tensor) -> torch.Tensor:
        return self._from_uniform(torch.special.ndtr(noise.to(torch.float64)))

    def _from_uniform(self, uniform: torch.Tensor) -> torch.Tensor:
        lower, upper = self._standardized_bounds()
        lower_cdf = torch.special.ndtr(lower)
        upper_cdf = torch.special.ndtr(upper)
        direct_probability = lower_cdf + uniform * (upper_cdf - lower_cdf)
        lower_survival = torch.special.ndtr(-lower)
        upper_survival = torch.special.ndtr(-upper)
        survival_probability = lower_survival - uniform * (
            lower_survival - upper_survival
        )
        tiny = torch.finfo(torch.float64).tiny
        epsilon = torch.finfo(torch.float64).eps
        direct_standard = torch.special.ndtri(
            direct_probability.clamp(tiny, 1.0 - epsilon)
        )
        survival_standard = -torch.special.ndtri(
            survival_probability.clamp(tiny, 1.0 - epsilon)
        )
        standard = torch.where(lower >= 0.0, survival_standard, direct_standard)
        sample = self.loc.to(torch.float64) + self.scale.to(torch.float64) * standard
        return sample.to(self.loc.dtype).clamp(-1.0, 1.0)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        bounded = value.clamp(-1.0, 1.0)
        log_base = self._normal.log_prob(bounded)
        log_mass = self._probability_mass().log().to(log_base.dtype)
        result = (log_base - log_mass).sum(dim=-1)
        inside = ((value >= -1.0) & (value <= 1.0)).all(dim=-1)
        return torch.where(inside, result, torch.full_like(result, -torch.inf))

    @property
    def deterministic_sample(self) -> torch.Tensor:
        return self.loc.clamp(-1.0, 1.0)


class ClippedNormal:
    """Gaussian passed through hard clipping, including both boundary atoms."""

    def __init__(self, location: torch.Tensor, scale: torch.Tensor):
        self.loc = location
        self.scale = scale
        self._normal = torch.distributions.Normal(location, scale)

    def _standardized_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        location = self.loc.to(torch.float64)
        scale = self.scale.to(torch.float64)
        return (-1.0 - location) / scale, (1.0 - location) / scale

    def _boundary_log_probabilities(self) -> tuple[torch.Tensor, torch.Tensor]:
        lower, upper = self._standardized_bounds()
        return torch.special.log_ndtr(lower), torch.special.log_ndtr(-upper)

    def rsample(self) -> torch.Tensor:
        return self._normal.rsample().clamp(-1.0, 1.0)

    def from_standard_noise(self, noise: torch.Tensor) -> torch.Tensor:
        return (self.loc + self.scale * noise).clamp(-1.0, 1.0)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        log_left, log_right = self._boundary_log_probabilities()
        interior = self._normal.log_prob(value)
        component = torch.where(
            value == -1.0,
            log_left.to(interior.dtype),
            torch.where(value == 1.0, log_right.to(interior.dtype), interior),
        )
        valid = ((value >= -1.0) & (value <= 1.0)).all(dim=-1)
        result = component.sum(dim=-1)
        return torch.where(valid, result, torch.full_like(result, -torch.inf))

    def entropy(self) -> torch.Tensor:
        lower, upper = self._standardized_bounds()
        log_left, log_right = self._boundary_log_probabilities()
        left = log_left.exp()
        right = log_right.exp()
        interior = (1.0 - left - right).clamp(0.0, 1.0)

        safe_log_left = torch.where(torch.isfinite(log_left), log_left, 0.0)
        safe_log_right = torch.where(torch.isfinite(log_right), log_right, 0.0)
        atom_entropy = -left * safe_log_left - right * safe_log_right

        inverse_sqrt_two_pi = 1.0 / math.sqrt(2.0 * math.pi)
        lower_density = inverse_sqrt_two_pi * (-0.5 * lower.square()).exp()
        upper_density = inverse_sqrt_two_pi * (-0.5 * upper.square()).exp()
        continuous_entropy = interior * (
            self.scale.to(torch.float64).log()
            + 0.5 * math.log(2.0 * math.pi * math.e)
        ) + 0.5 * (lower * lower_density - upper * upper_density)
        return (atom_entropy + continuous_entropy).sum(dim=-1).to(self.loc.dtype)

    @property
    def deterministic_sample(self) -> torch.Tensor:
        return self.loc.clamp(-1.0, 1.0)


BoundedActionDistribution = TanhNormal | TruncatedNormal | ClippedNormal


class VisualEncoder(nn.Module):
    def __init__(self, channels: int, input_size: int, output_size: int = 128):
        super().__init__()
        final_size = (input_size + 7) // 8
        self.network = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * final_size * final_size, output_size),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        leading = value.shape[:-3]
        value = value.reshape(-1, *value.shape[-3:]).to(torch.float32)
        encoded = self.network(value)
        return encoded.reshape(*leading, -1)


class EntitySetAttention(nn.Module):
    """Permutation-invariant learned or uniform pooling over one entity set."""

    def __init__(
        self,
        feature_size: int,
        embedding_size: int = 64,
        heads: int = 4,
        pooling: str = "learned_attention",
        mode: str = "independent_cross",
        self_attention_layers: int = 0,
    ):
        super().__init__()
        self.feature_size = int(feature_size)
        self.embedding_size = int(embedding_size)
        self.pooling = pooling
        self.mode = mode
        self.self_attention_blocks = nn.ModuleList(
            CrossTypeSelfAttentionBlock(self.embedding_size, int(heads))
            for _ in range(int(self_attention_layers))
        )
        if self.pooling not in {"learned_attention", "masked_mean"}:
            raise ValueError(f"Unsupported entity pooling: {self.pooling}")
        supported_modes = {
            "independent_cross",
            "mean_attention_residual",
            "direct_qkv_cross",
            "same_type_self_then_cross",
            "token_mlp_then_cross",
        }
        if self.mode not in supported_modes:
            raise ValueError(f"Unsupported entity attention mode: {self.mode}")
        if self.mode != "direct_qkv_cross":
            self.encoder = nn.Sequential(
                nn.Linear(self.feature_size, self.embedding_size),
                nn.SiLU(),
                nn.Linear(self.embedding_size, self.embedding_size),
                nn.SiLU(),
            )
        if self.pooling == "learned_attention":
            attention_arguments = (
                {"kdim": self.feature_size, "vdim": self.feature_size}
                if self.mode == "direct_qkv_cross"
                else {}
            )
            self.attention = nn.MultiheadAttention(
                self.embedding_size,
                int(heads),
                batch_first=True,
                **attention_arguments,
            )
        if self.mode == "mean_attention_residual":
            self.alpha = nn.Parameter(torch.ones(()))
        elif self.mode == "same_type_self_then_cross":
            self.pre_norm = nn.LayerNorm(self.embedding_size)
            self.pre_attention = nn.MultiheadAttention(
                self.embedding_size, int(heads), batch_first=True
            )
        elif self.mode == "token_mlp_then_cross":
            self.pre_norm = nn.LayerNorm(self.embedding_size)
            self.pre_mlp = nn.Sequential(
                nn.Linear(self.embedding_size, 2 * self.embedding_size),
                nn.SiLU(),
                nn.Linear(2 * self.embedding_size, self.embedding_size),
            )
        self.output = nn.Sequential(
            nn.LayerNorm(self.embedding_size),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.SiLU(),
        )

    def forward(
        self,
        query: torch.Tensor,
        entities: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 2 or query.shape[-1] != self.embedding_size:
            raise ValueError("Entity attention query must be [batch,embedding]")
        if entities.ndim != 3 or entities.shape[-1] != self.feature_size:
            raise ValueError("Entity attention values must be [batch,items,features]")
        if active.shape != entities.shape[:2]:
            raise ValueError("Entity attention mask must match [batch,items]")
        if entities.shape[0] != query.shape[0]:
            raise ValueError("Entity attention batch dimensions do not match")
        if entities.shape[1] == 0:
            return torch.zeros_like(query)

        active = active.to(torch.bool)
        active_entities = torch.where(
            active.unsqueeze(-1),
            entities.to(torch.float32),
            torch.zeros((), dtype=torch.float32, device=entities.device),
        )
        tokens = (
            active_entities
            if self.mode == "direct_qkv_cross"
            else self.encoder(active_entities)
        )
        safe_active = active.clone()
        empty = ~safe_active.any(dim=-1)
        safe_active[empty, 0] = True
        for block in self.self_attention_blocks:
            tokens = block(tokens, active, safe_active)
        if self.mode == "same_type_self_then_cross":
            normalized = self.pre_norm(tokens)
            related, _ = self.pre_attention(
                normalized,
                normalized,
                normalized,
                key_padding_mask=~safe_active,
                need_weights=False,
            )
            tokens = tokens + related
            tokens = torch.where(active.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        elif self.mode == "token_mlp_then_cross":
            tokens = tokens + self.pre_mlp(self.pre_norm(tokens))
            tokens = torch.where(active.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        if self.pooling == "learned_attention":
            attended, _ = _ownship_attention(self.attention, query.unsqueeze(1), tokens, ~safe_active)
            pooled = attended.squeeze(1)
            if self.mode == "mean_attention_residual":
                projection_start = 2 * self.embedding_size
                value_tokens = torch.nn.functional.linear(
                    tokens,
                    self.attention.in_proj_weight[projection_start:],
                    self.attention.in_proj_bias[projection_start:],
                )
                weights = active.to(value_tokens.dtype).unsqueeze(-1)
                mean_value = (value_tokens * weights).sum(dim=1) / weights.sum(
                    dim=1
                ).clamp_min(1.0)
                mean_context = self.attention.out_proj(mean_value)
                pooled = pooled + (1.0 - self.alpha) * (mean_context - pooled)
        else:
            weights = active.to(tokens.dtype).unsqueeze(-1)
            pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        context = self.output(pooled)
        return torch.where(empty.unsqueeze(-1), torch.zeros_like(context), context)


class CrossTypeEntityAttention(nn.Module):
    """One shared cross-type self-attention layer, then per-type ownship queries."""

    def __init__(
        self,
        feature_sizes: dict[str, int],
        embedding_size: int = 64,
        heads: int = 4,
    ):
        super().__init__()
        self.set_names = tuple(feature_sizes)
        self.feature_sizes = {name: int(size) for name, size in feature_sizes.items()}
        self.embedding_size = int(embedding_size)
        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(size, self.embedding_size),
                    nn.SiLU(),
                    nn.Linear(self.embedding_size, self.embedding_size),
                    nn.SiLU(),
                )
                for name, size in self.feature_sizes.items()
            }
        )
        self.type_embedding = nn.Embedding(len(self.set_names), self.embedding_size)
        self.self_attention = nn.MultiheadAttention(
            self.embedding_size, int(heads), batch_first=True
        )
        self.self_attention_norm = nn.LayerNorm(self.embedding_size)
        self.pool_attentions = nn.ModuleDict(
            {
                name: nn.MultiheadAttention(
                    self.embedding_size, int(heads), batch_first=True
                )
                for name in self.set_names
            }
        )
        self.outputs = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(self.embedding_size),
                    nn.Linear(self.embedding_size, self.embedding_size),
                    nn.SiLU(),
                )
                for name in self.set_names
            }
        )

    def _pool(
        self,
        name: str,
        query: torch.Tensor,
        tokens: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return torch.zeros_like(query)
        safe_active = active.clone()
        empty = ~safe_active.any(dim=-1)
        safe_active[empty, 0] = True
        attended, _ = self.pool_attentions[name](
            query.unsqueeze(1),
            tokens,
            tokens,
            key_padding_mask=~safe_active,
            need_weights=False,
        )
        context = self.outputs[name](attended.squeeze(1))
        return torch.where(empty.unsqueeze(-1), torch.zeros_like(context), context)

    def forward(
        self,
        query: torch.Tensor,
        entities_by_set: dict[str, torch.Tensor],
        active_by_set: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        if query.ndim != 2 or query.shape[-1] != self.embedding_size:
            raise ValueError("Cross-type entity query must be [batch,embedding]")
        encoded_sets: list[torch.Tensor] = []
        active_sets: list[torch.Tensor] = []
        counts: list[int] = []
        for type_index, name in enumerate(self.set_names):
            entities = entities_by_set[name]
            active = active_by_set[name].to(torch.bool)
            if entities.ndim != 3 or entities.shape[-1] != self.feature_sizes[name]:
                raise ValueError(
                    f"Cross-type {name} entities must be [batch,items,features]"
                )
            if active.shape != entities.shape[:2] or entities.shape[0] != query.shape[0]:
                raise ValueError(f"Cross-type {name} entity mask is incompatible")
            values = torch.where(
                active.unsqueeze(-1),
                entities.to(torch.float32),
                torch.zeros((), dtype=torch.float32, device=entities.device),
            )
            tokens = self.encoders[name](values) + self.type_embedding.weight[type_index]
            tokens = torch.where(active.unsqueeze(-1), tokens, torch.zeros_like(tokens))
            encoded_sets.append(tokens)
            active_sets.append(active)
            counts.append(tokens.shape[1])

        tokens = torch.cat(encoded_sets, dim=1)
        active = torch.cat(active_sets, dim=1)
        if tokens.shape[1] == 0:
            return tuple(torch.zeros_like(query) for _ in self.set_names)
        safe_active = active.clone()
        empty = ~safe_active.any(dim=-1)
        safe_active[empty, 0] = True
        related, _ = self.self_attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=~safe_active,
            need_weights=False,
        )
        tokens = self.self_attention_norm(tokens + related)
        tokens = torch.where(active.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        split_tokens = tokens.split(counts, dim=1)
        return tuple(
            self._pool(name, query, set_tokens, set_active)
            for name, set_tokens, set_active in zip(
                self.set_names, split_tokens, active_sets, strict=True
            )
        )


class CrossTypeSelfAttentionBlock(nn.Module):
    """Pre-normalized cross-type self-attention without an extra FFN."""

    def __init__(self, embedding_size: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(embedding_size))
        self.attention = nn.MultiheadAttention(
            int(embedding_size), int(heads), batch_first=True
        )

    def forward(
        self,
        tokens: torch.Tensor,
        active: torch.Tensor,
        safe_active: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm(tokens)
        related, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~safe_active,
            need_weights=False,
        )
        tokens = tokens + related
        return torch.where(active.unsqueeze(-1), tokens, torch.zeros_like(tokens))


class TypedUnifiedEntityAttention(nn.Module):
    """Encode entity types explicitly, then query one shared entity sequence."""

    def __init__(
        self,
        feature_sizes: dict[str, int],
        embedding_size: int = 64,
        heads: int = 4,
        self_attention_layers: int = 0,
    ):
        super().__init__()
        self.set_names = tuple(feature_sizes)
        self.feature_sizes = {name: int(size) for name, size in feature_sizes.items()}
        self.embedding_size = int(embedding_size)
        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(size, self.embedding_size),
                    nn.SiLU(),
                )
                for name, size in self.feature_sizes.items()
            }
        )
        self.type_embedding = nn.Embedding(len(self.set_names), self.embedding_size)
        self.self_attention_blocks = nn.ModuleList(
            CrossTypeSelfAttentionBlock(self.embedding_size, int(heads))
            for _ in range(int(self_attention_layers))
        )
        self.cross_attention = nn.MultiheadAttention(
            self.embedding_size, int(heads), batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(self.embedding_size),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.SiLU(),
        )

    def forward(
        self,
        query: torch.Tensor,
        entities_by_set: dict[str, torch.Tensor],
        active_by_set: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if query.ndim != 2 or query.shape[-1] != self.embedding_size:
            raise ValueError("Typed entity query must be [batch,embedding]")
        encoded_sets: list[torch.Tensor] = []
        active_sets: list[torch.Tensor] = []
        for type_index, name in enumerate(self.set_names):
            entities = entities_by_set[name]
            active = active_by_set[name].to(torch.bool)
            if (
                entities.ndim != 3
                or entities.shape[-1] != self.feature_sizes[name]
            ):
                raise ValueError(f"Typed {name} entities must be [batch,items,features]")
            if (
                active.shape != entities.shape[:2]
                or entities.shape[0] != query.shape[0]
            ):
                raise ValueError(f"Typed {name} entity mask is incompatible")
            values = torch.where(
                active.unsqueeze(-1),
                entities.to(torch.float32),
                torch.zeros((), dtype=torch.float32, device=entities.device),
            )
            tokens = (
                self.encoders[name](values) + self.type_embedding.weight[type_index]
            )
            encoded_sets.append(
                torch.where(active.unsqueeze(-1), tokens, torch.zeros_like(tokens))
            )
            active_sets.append(active)

        tokens = torch.cat(encoded_sets, dim=1)
        active = torch.cat(active_sets, dim=1)
        if tokens.shape[1] == 0:
            return torch.zeros_like(query)
        safe_active = active.clone()
        empty = ~safe_active.any(dim=-1)
        safe_active[empty, 0] = True
        for block in self.self_attention_blocks:
            tokens = block(tokens, active, safe_active)
        attended, _ = self.cross_attention(
            query.unsqueeze(1),
            tokens,
            tokens,
            key_padding_mask=~safe_active,
            need_weights=False,
        )
        context = self.output(attended.squeeze(1))
        return torch.where(empty.unsqueeze(-1), torch.zeros_like(context), context)


def _fixed_entity_slots(
    entities: torch.Tensor,
    active: torch.Tensor,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if entities.ndim != 3 or active.shape != entities.shape[:2]:
        raise ValueError("Entity slots must be [batch,items,features] with a matching mask")
    capacity = int(capacity)
    if capacity < 1 or entities.shape[1] > capacity:
        raise ValueError(f"Entity slots exceed fixed capacity {capacity}")
    active = active.to(torch.bool)
    padding = capacity - entities.shape[1]
    if padding:
        entities = torch.nn.functional.pad(entities, (0, 0, 0, padding))
        active = torch.nn.functional.pad(active, (0, padding), value=False)
    return entities, active


class DeepSetEncoder(nn.Module):
    """Lightweight masked sum over one variable-length entity set."""

    def __init__(self, feature_size: int, embedding_size: int = 32):
        super().__init__()
        self.feature_size = int(feature_size)
        self.embedding_size = int(embedding_size)
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_size, self.embedding_size),
            nn.SiLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.SiLU(),
        )

    def forward(self, entities: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        if entities.ndim != 3 or entities.shape[-1] != self.feature_size:
            raise ValueError("DeepSets entities must be [batch,items,features]")
        if active.shape != entities.shape[:2]:
            raise ValueError("DeepSets mask must match [batch,items]")
        encoded = self.encoder(entities.to(torch.float32))
        return (encoded * active.to(encoded.dtype).unsqueeze(-1)).sum(dim=1)


class RelationalEntityAttention(nn.Module):
    """Joint entity self-attention followed by ownship cross-attention."""

    def __init__(self, feature_size: int, embedding_size: int = 64, heads: int = 4):
        super().__init__()
        self.feature_size = int(feature_size)
        self.embedding_size = int(embedding_size)
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_size, self.embedding_size),
            nn.SiLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.SiLU(),
        )
        self.type_embedding = nn.Embedding(2, self.embedding_size)
        self.self_attention = nn.MultiheadAttention(
            self.embedding_size,
            int(heads),
            batch_first=True,
        )
        self.self_attention_norm = nn.LayerNorm(self.embedding_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embedding_size, 2 * self.embedding_size),
            nn.SiLU(),
            nn.Linear(2 * self.embedding_size, self.embedding_size),
        )
        self.feed_forward_norm = nn.LayerNorm(self.embedding_size)
        self.goal_attention = nn.MultiheadAttention(
            self.embedding_size,
            int(heads),
            batch_first=True,
        )
        self.obstacle_attention = nn.MultiheadAttention(
            self.embedding_size,
            int(heads),
            batch_first=True,
        )
        self.goal_output = nn.Sequential(
            nn.LayerNorm(self.embedding_size),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.SiLU(),
        )
        self.obstacle_output = nn.Sequential(
            nn.LayerNorm(self.embedding_size),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.SiLU(),
        )

    def _validate_set(
        self,
        name: str,
        entities: torch.Tensor,
        active: torch.Tensor,
        batch_size: int,
    ) -> None:
        if entities.ndim != 3 or entities.shape[-1] != self.feature_size:
            raise ValueError(
                f"Relational {name} entities must be [batch,items,features]"
            )
        if active.shape != entities.shape[:2]:
            raise ValueError(f"Relational {name} mask must match [batch,items]")
        if entities.shape[0] != batch_size:
            raise ValueError(f"Relational {name} batch dimension does not match")

    def _pool(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        active: torch.Tensor,
        attention: nn.MultiheadAttention,
        output: nn.Module,
    ) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return torch.zeros_like(query)
        safe_active = active.clone()
        empty = ~safe_active.any(dim=-1)
        safe_active[empty, 0] = True
        attended, _ = attention(
            query.unsqueeze(1),
            tokens,
            tokens,
            key_padding_mask=~safe_active,
            need_weights=False,
        )
        context = output(attended.squeeze(1))
        return torch.where(empty.unsqueeze(-1), torch.zeros_like(context), context)

    def forward(
        self,
        query: torch.Tensor,
        goal_entities: torch.Tensor,
        goal_active: torch.Tensor,
        obstacle_entities: torch.Tensor,
        obstacle_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 2 or query.shape[-1] != self.embedding_size:
            raise ValueError("Relational entity query must be [batch,embedding]")
        self._validate_set("goal", goal_entities, goal_active, query.shape[0])
        self._validate_set(
            "obstacle", obstacle_entities, obstacle_active, query.shape[0]
        )
        goal_active = goal_active.to(torch.bool)
        obstacle_active = obstacle_active.to(torch.bool)
        goal_values = torch.where(
            goal_active.unsqueeze(-1),
            goal_entities.to(torch.float32),
            torch.zeros((), dtype=torch.float32, device=goal_entities.device),
        )
        obstacle_values = torch.where(
            obstacle_active.unsqueeze(-1),
            obstacle_entities.to(torch.float32),
            torch.zeros((), dtype=torch.float32, device=obstacle_entities.device),
        )
        goal_tokens = self.encoder(goal_values) + self.type_embedding.weight[0]
        obstacle_tokens = self.encoder(obstacle_values) + self.type_embedding.weight[1]
        goal_count = goal_tokens.shape[1]
        tokens = torch.cat((goal_tokens, obstacle_tokens), dim=1)
        active = torch.cat((goal_active, obstacle_active), dim=1)
        if tokens.shape[1] == 0:
            return torch.zeros_like(query), torch.zeros_like(query)

        safe_active = active.clone()
        empty = ~safe_active.any(dim=-1)
        safe_active[empty, 0] = True
        related, _ = self.self_attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=~safe_active,
            need_weights=False,
        )
        tokens = self.self_attention_norm(tokens + related)
        tokens = self.feed_forward_norm(tokens + self.feed_forward(tokens))
        tokens = torch.where(active.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        goal_tokens = tokens[:, :goal_count]
        obstacle_tokens = tokens[:, goal_count:]
        return (
            self._pool(
                query,
                goal_tokens,
                goal_active,
                self.goal_attention,
                self.goal_output,
            ),
            self._pool(
                query,
                obstacle_tokens,
                obstacle_active,
                self.obstacle_attention,
                self.obstacle_output,
            ),
        )


def _distribution(
    location: torch.Tensor,
    log_scale: torch.Tensor,
    distribution_kind: str,
) -> BoundedActionDistribution:
    scale = log_scale.clamp(-5.0, 1.5).exp()
    if distribution_kind == "tanh_normal":
        return TanhNormal(location, scale)
    if distribution_kind == "truncated_normal":
        return TruncatedNormal(location, scale)
    if distribution_kind == "clipped_normal":
        return ClippedNormal(location, scale)
    raise ValueError(f"Unknown action distribution: {distribution_kind}")


class ActorCritic(nn.Module):
    """One parameterized Actor-Critic shared by every active input variant."""

    def __init__(self, architecture_id: str, critic_mode: str = "local", *, build_critic: bool = True):
        super().__init__()
        self.architecture_id = architecture_id
        self.variant = actor_variant(architecture_id)
        if critic_mode not in CRITIC_MODES:
            raise ValueError(f"Unsupported Critic mode: {critic_mode}")
        if critic_mode == "centralized_team" and not (
            self.variant.actor_kind == "defender"
            and "friendlies" in self.variant.entity_sets
        ):
            raise ValueError(
                "Centralized team Critic requires the multi-patrol observation"
            )
        self.critic_mode = critic_mode
        self.build_critic = build_critic
        actor_vector_size = self.variant.vector_input_size
        critic_vector_size = self.variant.critic_vector_input_size
        if self.variant.input_kind == "semantic_map":
            semantic_channels = self.variant.semantic_input_shape[0]
            self.actor_semantic = VisualEncoder(semantic_channels, 96, 64)
            if build_critic:
                self.critic_semantic = VisualEncoder(semantic_channels, 96, 64)
            self.actor_vector = nn.Sequential(
                nn.Linear(actor_vector_size, 64), nn.SiLU()
            )
            if build_critic:
                self.critic_vector = nn.Sequential(
                    nn.Linear(critic_vector_size, 64), nn.SiLU()
                )
            fusion_size = 2 * 64
        elif self.variant.input_kind == "entity_attention":
            embedding_size = self.variant.attention_embedding_size
            attention_heads = self.variant.attention_heads
            self.actor_vector = nn.Sequential(
                nn.Linear(actor_vector_size, embedding_size), nn.SiLU()
            )
            if build_critic:
                self.critic_vector = nn.Sequential(
                    nn.Linear(critic_vector_size, embedding_size), nn.SiLU()
                )
            if (
                self.variant.entity_attention_kind
                == "independent_ownship_cross_attention"
            ):
                if self.variant.entity_streams == "unified":
                    actor_feature_sizes = {
                        buffer_name: self.variant.entity_feature_size_for(buffer_name)
                        for buffer_name in self.variant.entity_sets
                    }
                    critic_feature_sizes = {
                        buffer_name: self.variant.critic_entity_feature_size_for(
                            buffer_name
                        )
                        for buffer_name in self.variant.entity_sets
                    }
                    self.actor_typed_entities = TypedUnifiedEntityAttention(
                        actor_feature_sizes,
                        embedding_size,
                        attention_heads,
                        self.variant.entity_self_attention_layers,
                    )
                    if build_critic:
                        self.critic_typed_entities = TypedUnifiedEntityAttention(
                            critic_feature_sizes,
                            embedding_size,
                            attention_heads,
                            self.variant.entity_self_attention_layers,
                        )
                    fusion_size = 2 * embedding_size
                elif (
                    self.variant.entity_attention_mode
                    == "cross_type_softmax_self_then_cross"
                ):
                    actor_feature_sizes = {
                        buffer_name: self.variant.entity_feature_size_for(buffer_name)
                        for buffer_name in self.variant.entity_sets
                    }
                    critic_feature_sizes = {
                        buffer_name: self.variant.critic_entity_feature_size_for(
                            buffer_name
                        )
                        for buffer_name in self.variant.entity_sets
                    }
                    self.actor_cross_type_entities = CrossTypeEntityAttention(
                        actor_feature_sizes, embedding_size, attention_heads
                    )
                    if build_critic:
                        self.critic_cross_type_entities = CrossTypeEntityAttention(
                            critic_feature_sizes, embedding_size, attention_heads
                        )
                    fusion_size = (
                        1 + len(self.variant.entity_sets)
                    ) * embedding_size
                else:
                    attention_arguments = {
                        "embedding_size": embedding_size,
                        "heads": attention_heads,
                        "pooling": self.variant.entity_pooling,
                        "mode": self.variant.entity_attention_mode,
                        "self_attention_layers": (
                            self.variant.entity_self_attention_layers
                        ),
                    }
                    for buffer_name in self.variant.entity_sets:
                        prefix = entity_tensor_prefix(buffer_name)
                        setattr(
                            self,
                            f"actor_{prefix}_entities",
                            EntitySetAttention(
                                self.variant.entity_feature_size_for(buffer_name),
                                **attention_arguments,
                            ),
                        )
                        if build_critic:
                            setattr(
                                self,
                                f"critic_{prefix}_entities",
                                EntitySetAttention(
                                    self.variant.critic_entity_feature_size_for(
                                        buffer_name
                                    ),
                                    **attention_arguments,
                                ),
                            )
                    fusion_size = (
                        1 + len(self.variant.entity_sets)
                    ) * embedding_size
            elif (
                self.variant.entity_attention_kind
                == "joint_self_then_ownship_cross_attention"
            ):
                entity_size = self.variant.entity_feature_size
                self.actor_relational_entities = RelationalEntityAttention(
                    entity_size, embedding_size, attention_heads
                )
                if build_critic:
                    self.critic_relational_entities = RelationalEntityAttention(
                        entity_size, embedding_size, attention_heads
                    )
            else:
                raise ValueError("Entity attention architecture kind is missing")
            if (
                self.variant.entity_attention_kind
                != "independent_ownship_cross_attention"
            ):
                fusion_size = (1 + len(self.variant.entity_sets)) * embedding_size
        elif self.variant.input_kind == "entity_mlp":
            self.actor_vector = nn.Sequential(
                nn.Linear(actor_vector_size, 64), nn.SiLU()
            )
            if build_critic:
                self.critic_vector = nn.Sequential(
                    nn.Linear(critic_vector_size, 64), nn.SiLU()
                )
            entity_size = self.variant.entity_feature_size
            goal_capacity = int(self.variant.goal_entity_capacity or 0)
            obstacle_capacity = int(self.variant.obstacle_entity_capacity or 0)
            if self.variant.entity_mlp_kind == "flattened":
                flattened_size = (goal_capacity + obstacle_capacity) * (entity_size + 1)
                self.actor_flat_entities = nn.Sequential(
                    nn.Linear(flattened_size, 64), nn.SiLU()
                )
                if build_critic:
                    self.critic_flat_entities = nn.Sequential(
                        nn.Linear(flattened_size, 64), nn.SiLU()
                    )
            elif self.variant.entity_mlp_kind == "tied_linear_sum":
                tied_input_size = 2 * (entity_size + 1)
                self.actor_tied_entities = nn.Sequential(
                    nn.Linear(tied_input_size, 64), nn.SiLU()
                )
                if build_critic:
                    self.critic_tied_entities = nn.Sequential(
                        nn.Linear(tied_input_size, 64), nn.SiLU()
                    )
            elif self.variant.entity_mlp_kind == "deep_sets":
                self.actor_goal_entities = DeepSetEncoder(entity_size)
                self.actor_obstacle_entities = DeepSetEncoder(entity_size)
                if build_critic:
                    self.critic_goal_entities = DeepSetEncoder(entity_size)
                if build_critic:
                    self.critic_obstacle_entities = DeepSetEncoder(entity_size)
            else:
                raise ValueError("Entity MLP architecture kind is missing")
            fusion_size = 2 * 64
        else:
            raise ValueError(f"Unsupported Actor input kind: {self.variant.input_kind}")
        body_size = self.variant.body_hidden_size
        self.actor_body = mlp_body(fusion_size, body_size)
        if build_critic:
            self.critic_body = mlp_body(fusion_size, body_size)
        self.actor_distribution_head = nn.Linear(body_size, 4)
        with torch.no_grad():
            self.actor_distribution_head.weight[2:].zero_()
            self.actor_distribution_head.bias[2:].fill_(self.variant.actor_initial_log_scale)
        if build_critic:
            self.value_head = nn.Linear(body_size, 1)
        if build_critic and self.critic_mode == "centralized_team":
            self.central_value_head = nn.Linear(body_size, 1, bias=False)
            nn.init.zeros_(self.central_value_head.weight)

    @property
    def uses_centralized_critic(self) -> bool:
        return self.critic_mode == "centralized_team"

    @classmethod
    def architecture_contract(cls, architecture_id: str) -> dict[str, object]:
        del cls
        return actor_variant(architecture_id).architecture_contract()

    @classmethod
    def supported_architecture_ids(cls) -> tuple[str, ...]:
        del cls
        return actor_variant_ids()

    def parameter_counts(self) -> dict[str, int]:
        named = tuple(self.named_parameters())
        actor = sum(parameter.numel() for name, parameter in named if name.startswith("actor"))
        critic = sum(
            parameter.numel() for name, parameter in named if not name.startswith("actor")
        )
        return {"actor": actor, "critic": critic, "total": actor + critic}

    def actor_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.named_parameters()
            if name.startswith("actor")
        ]

    def _require_input_kind(self, expected: str) -> None:
        if self.variant.input_kind != expected:
            raise ValueError(
                f"Actor input kind is {self.variant.input_kind}, expected {expected}"
            )

    def _validate_semantic_input(self, semantic_map: torch.Tensor) -> None:
        expected = self.variant.semantic_input_shape
        if semantic_map.shape[-3:] != expected:
            raise ValueError(
                "Actor semantic input must end in "
                f"{list(expected)}, got {tuple(semantic_map.shape)}"
            )

    def actor_features(self, semantic_map: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        self._require_input_kind("semantic_map")
        self._validate_semantic_input(semantic_map)
        features = (
            self.actor_semantic(semantic_map),
            self.actor_vector(vector.to(torch.float32)),
        )
        return self.actor_body(torch.cat(features, dim=-1))

    def critic_features(self, semantic_map: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        self._require_input_kind("semantic_map")
        self._validate_semantic_input(semantic_map)
        features = (
            self.critic_semantic(semantic_map),
            self.critic_vector(vector.to(torch.float32)),
        )
        return self.critic_body(torch.cat(features, dim=-1))

    def _entity_features(
        self,
        observation: dict[str, torch.Tensor],
        *,
        actor: bool,
    ) -> torch.Tensor:
        if not self.variant.uses_entity_buffers:
            raise ValueError("Actor architecture does not use entity buffers")
        vector_encoder = self.actor_vector if actor else self.critic_vector
        vector_key = (
            "vector"
            if actor or "critic_vector" not in observation
            else "critic_vector"
        )
        vector = observation[vector_key].to(torch.float32)
        vector_feature = vector_encoder(vector)

        def entity_values(buffer_name: str) -> torch.Tensor:
            prefix = entity_tensor_prefix(buffer_name)
            critic_key = f"{prefix}_critic_entities"
            if not actor and critic_key in self.variant.critic_observation_keys:
                return observation[critic_key]
            return observation[f"{prefix}_entities"]

        def entity_mask(buffer_name: str) -> torch.Tensor:
            prefix = entity_tensor_prefix(buffer_name)
            critic_key = f"{prefix}_critic_mask"
            if not actor and critic_key in self.variant.critic_observation_keys:
                return observation[critic_key]
            return observation[f"{prefix}_mask"]

        if (
            self.variant.entity_attention_kind
            == "independent_ownship_cross_attention"
        ):
            if self.variant.entity_streams == "unified":
                encoder = (
                    self.actor_typed_entities
                    if actor
                    else self.critic_typed_entities
                )
                unified_entity_feature = encoder(
                    vector_feature,
                    {
                        buffer_name: entity_values(buffer_name)
                        for buffer_name in self.variant.entity_sets
                    },
                    {
                        buffer_name: entity_mask(buffer_name)
                        for buffer_name in self.variant.entity_sets
                    },
                )
            elif (
                self.variant.entity_attention_mode
                == "cross_type_softmax_self_then_cross"
            ):
                encoder = (
                    self.actor_cross_type_entities
                    if actor
                    else self.critic_cross_type_entities
                )
                entity_features = encoder(
                    vector_feature,
                    {
                        buffer_name: entity_values(buffer_name)
                        for buffer_name in self.variant.entity_sets
                    },
                    {
                        buffer_name: entity_mask(buffer_name)
                        for buffer_name in self.variant.entity_sets
                    },
                )
            else:
                feature_by_set = {}
                for buffer_name in self.variant.entity_sets:
                    prefix = entity_tensor_prefix(buffer_name)
                    encoder = getattr(
                        self,
                        f"{'actor' if actor else 'critic'}_{prefix}_entities",
                    )
                    feature_by_set[buffer_name] = encoder(
                        vector_feature,
                        entity_values(buffer_name),
                        entity_mask(buffer_name),
                    )
                entity_features = tuple(
                    feature_by_set[buffer_name]
                    for buffer_name in self.variant.entity_sets
                )
        elif (
            self.variant.entity_attention_kind
            == "joint_self_then_ownship_cross_attention"
        ):
            relational_encoder = (
                self.actor_relational_entities
                if actor
                else self.critic_relational_entities
            )
            goal_feature, obstacle_feature = relational_encoder(
                vector_feature,
                observation["goal_entities"],
                observation["goal_mask"],
                observation["obstacle_entities"],
                observation["obstacle_mask"],
            )
        elif self.variant.entity_mlp_kind == "flattened":
            goal_entities, goal_active = _fixed_entity_slots(
                observation["goal_entities"],
                observation["goal_mask"],
                int(self.variant.goal_entity_capacity or 0),
            )
            obstacle_entities, obstacle_active = _fixed_entity_slots(
                observation["obstacle_entities"],
                observation["obstacle_mask"],
                int(self.variant.obstacle_entity_capacity or 0),
            )
            entities = torch.cat((goal_entities, obstacle_entities), dim=1)
            active = torch.cat((goal_active, obstacle_active), dim=1)
            flattened = torch.cat(
                (entities.to(torch.float32), active.to(torch.float32).unsqueeze(-1)),
                dim=-1,
            ).flatten(start_dim=1)
            entity_encoder = self.actor_flat_entities if actor else self.critic_flat_entities
            entity_feature = entity_encoder(flattened)
            body = self.actor_body if actor else self.critic_body
            return body(torch.cat((vector_feature, entity_feature), dim=-1))
        elif self.variant.entity_mlp_kind == "tied_linear_sum":
            goal_entities, goal_active = _fixed_entity_slots(
                observation["goal_entities"],
                observation["goal_mask"],
                int(self.variant.goal_entity_capacity or 0),
            )
            obstacle_entities, obstacle_active = _fixed_entity_slots(
                observation["obstacle_entities"],
                observation["obstacle_mask"],
                int(self.variant.obstacle_entity_capacity or 0),
            )

            def masked_sum(
                entities: torch.Tensor, active: torch.Tensor
            ) -> torch.Tensor:
                active_float = active.to(torch.float32).unsqueeze(-1)
                slots = torch.cat(
                    (entities.to(torch.float32) * active_float, active_float), dim=-1
                )
                return slots.sum(dim=1)

            tied_input = torch.cat(
                (
                    masked_sum(goal_entities, goal_active),
                    masked_sum(obstacle_entities, obstacle_active),
                ),
                dim=-1,
            )
            entity_encoder = (
                self.actor_tied_entities if actor else self.critic_tied_entities
            )
            entity_feature = entity_encoder(tied_input)
            body = self.actor_body if actor else self.critic_body
            return body(torch.cat((vector_feature, entity_feature), dim=-1))
        elif self.variant.entity_mlp_kind == "deep_sets":
            goal_entities, goal_active = _fixed_entity_slots(
                observation["goal_entities"],
                observation["goal_mask"],
                int(self.variant.goal_entity_capacity or 0),
            )
            obstacle_entities, obstacle_active = _fixed_entity_slots(
                observation["obstacle_entities"],
                observation["obstacle_mask"],
                int(self.variant.obstacle_entity_capacity or 0),
            )
            goal_encoder = self.actor_goal_entities if actor else self.critic_goal_entities
            obstacle_encoder = (
                self.actor_obstacle_entities if actor else self.critic_obstacle_entities
            )
            goal_feature = goal_encoder(goal_entities, goal_active)
            obstacle_feature = obstacle_encoder(obstacle_entities, obstacle_active)
        else:
            raise ValueError("Entity-buffer architecture kind is missing")
        body = self.actor_body if actor else self.critic_body
        if (
            self.variant.entity_attention_kind
            == "independent_ownship_cross_attention"
        ):
            if self.variant.entity_streams == "unified":
                return body(torch.cat((vector_feature, unified_entity_feature), dim=-1))
            fused = torch.cat(
                (
                    vector_feature,
                    *entity_features,
                ),
                dim=-1,
            )
            return body(fused)
        return body(torch.cat((vector_feature, goal_feature, obstacle_feature), dim=-1))

    def actor_features_from_observation(
        self, observation: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.variant.input_kind == "semantic_map":
            return self.actor_features(observation["semantic_map"], observation["vector"])
        return self._entity_features(observation, actor=True)

    def critic_features_from_observation(
        self, observation: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.variant.input_kind == "semantic_map":
            return self.critic_features(observation["semantic_map"], observation["vector"])
        return self._entity_features(observation, actor=False)

    def _actor_distribution_parameters(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        location, raw_log_scale = self.actor_distribution_head(feature).split(2, dim=-1)
        return location, raw_log_scale.clamp(-5.0, 1.5)

    def forward(
        self, semantic_map: torch.Tensor, vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_feature = self.actor_features(semantic_map, vector)
        critic_feature = self.critic_features(semantic_map, vector)
        location, log_scale = self._actor_distribution_parameters(actor_feature)
        value = self.value_head(critic_feature)
        return location, log_scale, value

    def distribution_and_value(
        self, semantic_map: torch.Tensor, vector: torch.Tensor
    ) -> tuple[BoundedActionDistribution, torch.Tensor]:
        location, log_scale, value = self(semantic_map, vector)
        return _distribution(location, log_scale, self.variant.action_distribution), value

    def distribution_and_value_from_observation(
        self, observation: dict[str, torch.Tensor]
    ) -> tuple[BoundedActionDistribution, torch.Tensor]:
        actor_feature = self.actor_features_from_observation(observation)
        critic_feature = self.critic_features_from_observation(observation)
        location, log_scale = self._actor_distribution_parameters(actor_feature)
        return (
            _distribution(
                location,
                log_scale,
                self.variant.action_distribution,
            ),
            self.value_head(critic_feature),
        )

    def encode_semantic(self, semantic_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_semantic_input(semantic_map)
        return self.actor_semantic(semantic_map), self.critic_semantic(semantic_map)

    def distribution_and_value_from_encoded(
        self,
        actor_semantic: torch.Tensor,
        critic_semantic: torch.Tensor,
        vector: torch.Tensor,
    ) -> tuple[BoundedActionDistribution, torch.Tensor]:
        vector = vector.to(torch.float32)
        actor_feature = self.actor_body(
            torch.cat((actor_semantic, self.actor_vector(vector)), dim=-1)
        )
        critic_feature = self.critic_body(
            torch.cat((critic_semantic, self.critic_vector(vector)), dim=-1)
        )
        location, log_scale = self._actor_distribution_parameters(actor_feature)
        return (
            _distribution(
                location,
                log_scale,
                self.variant.action_distribution,
            ),
            self.value_head(critic_feature),
        )

    def value_from_encoded(
        self, critic_semantic: torch.Tensor, vector: torch.Tensor
    ) -> torch.Tensor:
        feature = self.critic_body(
            torch.cat((critic_semantic, self.critic_vector(vector.to(torch.float32))), dim=-1)
        )
        return self.value_head(feature)

    def actor_distribution(
        self, semantic_map: torch.Tensor, vector: torch.Tensor
    ) -> BoundedActionDistribution:
        feature = self.actor_features(semantic_map, vector)
        location, log_scale = self._actor_distribution_parameters(feature)
        return _distribution(
            location,
            log_scale,
            self.variant.action_distribution,
        )

    def actor_distribution_from_observation(
        self, observation: dict[str, torch.Tensor]
    ) -> BoundedActionDistribution:
        feature = self.actor_features_from_observation(observation)
        location, log_scale = self._actor_distribution_parameters(feature)
        return _distribution(
            location,
            log_scale,
            self.variant.action_distribution,
        )

    def value(self, semantic_map: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        feature = self.critic_features(semantic_map, vector)
        return self.value_head(feature)

    def value_from_observation(
        self, observation: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.value_head(self.critic_features_from_observation(observation))

    def value_from_grouped_observation(
        self,
        observation: dict[str, torch.Tensor],
        actor_active: torch.Tensor,
    ) -> torch.Tensor:
        """Centralized values for complete per-scene patrol groups.

        The final active dimension is the patrol slot axis. Local Critic features
        are pooled with a mask, so the result is invariant to patrol ordering and
        supports a variable active fleet size without padded slots contributing.
        """
        if not self.uses_centralized_critic:
            raise RuntimeError("Grouped values require the centralized team Critic")
        if actor_active.ndim < 2:
            raise ValueError("Centralized Critic input must include a patrol slot axis")
        active = actor_active.to(torch.bool)
        leading_dimensions = active.ndim
        actor_count = active.numel()
        flat_active = active.reshape(-1)
        flat_observation = {
            key: observation[key].reshape(
                actor_count, *observation[key].shape[leading_dimensions:]
            )
            for key in self.variant.observation_keys
        }
        active_features = self.critic_features_from_observation(
            {key: value[flat_active] for key, value in flat_observation.items()}
        )
        feature_size = self.variant.body_hidden_size
        flat_features = torch.zeros(
            actor_count,
            feature_size,
            dtype=active_features.dtype,
            device=actor_active.device,
        )
        flat_features[flat_active] = active_features
        grouped_features = flat_features.reshape(*active.shape, feature_size)
        team_feature = grouped_features.sum(dim=-2) / active.sum(
            dim=-1, keepdim=True
        ).clamp_min(1).to(grouped_features.dtype)
        local_value = self.value_head(grouped_features)
        team_delta = self.central_value_head(team_feature).unsqueeze(-2)
        return torch.where(
            active.unsqueeze(-1),
            local_value + team_delta,
            torch.zeros_like(local_value),
        )


def actor_deployment_state_dict(
    model: ActorCritic, state: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Return the shared decentralized Actor state from PPO or MAPPO weights."""
    target = model.state_dict()
    missing = set(target) - set(state)
    if missing:
        raise ValueError(f"Deployment checkpoint model state is incomplete: {sorted(missing)}")
    for name, value in target.items():
        if value.shape != state[name].shape:
            raise ValueError(f"Deployment checkpoint shape mismatch: {name}")
    return {name: state[name] for name in target}
