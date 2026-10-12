"""External-conditioned direct WIS latent fields with genuine global carry.

The original learned feature/attention/geometry blocks are reused, not the
original liquid heads or 4180-based initialization. This file neither imports
nor calls a physical time-step integrator. This is a new, untrained component.
"""

from __future__ import annotations

import math
from copy import copy
from dataclasses import asdict, dataclass

import torch
from torch import nn

from ._features.model import GeometryBlock
from .clock_features import (
    AVAILABILITY_DELAY_SCALE_S,
    HISTORY_POSITION_SCALE_S,
    LEGACY_CLOCK_CONTRACT_ID,
    NATIVE_DURATION_SCALE_S,
    token_clock_features,
    validate_clock_config,
)
from .utc_features import UTC_FEATURE_CONTRACT_ID, utc_cycle_features

NODE_NAMES = ("liquid_enthalpy_coordinate", "bed_temperature_C")
FEATURE_NORM_CONTRACT_ID = "MIXED_HIDDEN_HEAD_INPUT_LEARNED_LAYERNORM_V1"

GLOBAL_NAMES = (
    "ice_mass_coordinate",
    "snow_mass_coordinate",
    "ice_cold_coordinate",
    "snow_cold_coordinate",
    "ice_area_coordinate",
    "snow_area_coordinate",
    "snow_density_coordinate",
)


@dataclass
class DirectPhaseModelConfig:
    node_features: int
    token_features: int
    token_kinds: int = 2
    width: int = 32
    blocks: int = 2
    heads: int = 4
    spectral_modes: int = 4
    time_scale_s: float = 86400.0
    head_initial_scale: float = 0.001
    utc_feature_contract_id: str = UTC_FEATURE_CONTRACT_ID
    feature_norm_contract_id: str = FEATURE_NORM_CONTRACT_ID
    mixed_hidden_feature_normalization: str = "learned_layernorm"
    feature_norm_eps: float = 1e-5
    # Old configurations missing these fields retain the exact old clock map.
    # The NEW execution_common factory explicitly selects the new contract.
    clock_feature_contract_id: str = LEGACY_CLOCK_CONTRACT_ID
    token_clock_feature_mode: str = "legacy_relative_days"
    clock_history_position_scale_s: float = HISTORY_POSITION_SCALE_S
    clock_duration_scale_s: float = NATIVE_DURATION_SCALE_S
    clock_availability_delay_scale_s: float = AVAILABILITY_DELAY_SCALE_S
    # Explicit candidate coordinate mean, not an observed initial lake profile.
    liquid_enthalpy_coordinate_reference: float = 0.5
    feature_dtype: str = "float64"
    liquid_coordinate_contract_id: str = "NATIVE_REFERENCE_WATER_LIQUID0_40_RATIONAL_ENTHALPY_V5"
    native_liquid_temperature_upper_C: float = 40.0
    liquid_latent_ceiling_C: float = 40.0 * (1.0 - 8.0 * torch.finfo(torch.float64).eps)
    solid_coordinate_contract_id: str = "NATIVE_SOLID100K_0C_RATIONAL_COLD_ENTHALPY_V6"
    solid_cold_ceiling_K: float = (273.15 - 100.0) * (1.0 - 4096.0 * torch.finfo(torch.float64).eps)


@dataclass
class DirectPhaseCarry:
    node_raw: torch.Tensor  # [B,N,2], fixed external liquid mass is not learned
    global_raw: torch.Tensor  # [B,7], one reservoir head per lake
    memory: torch.Tensor  # [B,N,W]


@dataclass
class DirectPhaseRaw:
    node_raw: torch.Tensor  # [B,Q,N,2]
    global_raw: torch.Tensor  # [B,Q,7]
    memory: torch.Tensor  # [B,Q,N,W]
    process_control: torch.Tensor

    def physical_coordinates(self):
        result = {name: self.node_raw[..., i] for i, name in enumerate(NODE_NAMES)}
        result.update({name: self.global_raw[..., i] for i, name in enumerate(GLOBAL_NAMES)})
        return result


class DirectPhaseInverseOperator(nn.Module):
    def __init__(self, cfg: DirectPhaseModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.utc_feature_contract_id != UTC_FEATURE_CONTRACT_ID:
            raise ValueError("Explicit V7 Gregorian UTC feature contract required")
        if (
            cfg.solid_coordinate_contract_id != "NATIVE_SOLID100K_0C_RATIONAL_COLD_ENTHALPY_V6"
            or cfg.solid_cold_ceiling_K
            != (273.15 - 100.0) * (1.0 - 4096.0 * torch.finfo(torch.float64).eps)
        ):
            raise ValueError("Declared native atmosphere solid latent contract required")
        if (
            cfg.liquid_coordinate_contract_id
            != "NATIVE_REFERENCE_WATER_LIQUID0_40_RATIONAL_ENTHALPY_V5"
            or cfg.native_liquid_temperature_upper_C != 40.0
            or cfg.liquid_latent_ceiling_C != 40.0 * (1.0 - 8.0 * torch.finfo(torch.float64).eps)
        ):
            raise ValueError("Declared native-domain liquid latent contract required")
        if cfg.feature_norm_contract_id != FEATURE_NORM_CONTRACT_ID:
            raise ValueError("Explicit mixed hidden feature normalization contract required")
        if cfg.mixed_hidden_feature_normalization not in ("learned_layernorm", "identity_control"):
            raise ValueError("Explicit learned_layernorm or identity_control required")
        if not math.isfinite(cfg.feature_norm_eps) or cfg.feature_norm_eps <= 0:
            raise ValueError("Finite positive hidden LayerNorm epsilon required")
        validate_clock_config(
            cfg.clock_feature_contract_id,
            cfg.token_clock_feature_mode,
            cfg.clock_history_position_scale_s,
            cfg.clock_duration_scale_s,
            cfg.clock_availability_delay_scale_s,
        )
        w = cfg.width
        if w % cfg.heads or cfg.time_scale_s <= 0 or not 0 < cfg.head_initial_scale <= 1:
            raise ValueError("Invalid model feature/time/initialization configuration")
        for name, count, value in (
            ("node_mean", cfg.node_features, 0),
            ("node_std", cfg.node_features, 1),
            ("token_mean", cfg.token_features, 0),
            ("token_std", cfg.token_features, 1),
        ):
            self.register_buffer(name, torch.full((count,), float(value)))
        self.node_encoder = nn.Sequential(
            nn.Linear(cfg.node_features, w), nn.GELU(), nn.Linear(w, w)
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(cfg.token_features + 8, w), nn.GELU(), nn.Linear(w, w)
        )
        self.kind_embedding = nn.Embedding(cfg.token_kinds, w)
        self.null_token = nn.Parameter(torch.zeros(1, 1, w))
        self.time_encoder = nn.Sequential(nn.Linear(9, w), nn.GELU(), nn.Linear(w, w))
        self.attention = nn.MultiheadAttention(w, cfg.heads, batch_first=True, dropout=0.0)
        self.blocks = nn.ModuleList(
            [GeometryBlock(w, cfg.spectral_modes) for _ in range(cfg.blocks)]
        )
        # New architecture hypothesis: normalize mixed latent feature vectors,
        # never physical states, seconds, weather, target values, or temperatures.
        self.feature_norm = nn.LayerNorm(w, eps=cfg.feature_norm_eps, elementwise_affine=True)
        self.node_carry_encoder = nn.Linear(len(NODE_NAMES) + w, w)
        self.global_carry_encoder = nn.Linear(len(GLOBAL_NAMES), w)
        self.initial_node = nn.Linear(w, len(NODE_NAMES))
        self.initial_global = nn.Linear(w, len(GLOBAL_NAMES))
        self.delta_node = nn.Linear(w, len(NODE_NAMES))
        self.delta_global = nn.Linear(w, len(GLOBAL_NAMES))
        self.delta_memory = nn.Linear(w, w)
        self.process = nn.Linear(w, 1)
        if cfg.feature_dtype not in ("float32", "float64"):
            raise ValueError("Explicit float32/float64 feature arithmetic required")
        self.to(dtype=torch.float64 if cfg.feature_dtype == "float64" else torch.float32)
        with torch.no_grad():
            for head in (
                self.initial_node,
                self.initial_global,
                self.delta_node,
                self.delta_global,
                self.delta_memory,
            ):
                head.weight.mul_(cfg.head_initial_scale)
                head.bias.mul_(cfg.head_initial_scale)
            self.initial_node.bias.copy_(
                torch.tensor(
                    [cfg.liquid_enthalpy_coordinate_reference, 0.0], dtype=self.node_mean.dtype
                )
            )
            self.initial_global.bias.copy_(
                torch.tensor([0.02, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0], dtype=self.node_mean.dtype)
            )

    def set_normalization(self, node_mean, node_std, token_mean, token_std):
        # Candidates must be feature-contract-specific before this API is used.
        for name, value in (
            ("node_mean", node_mean),
            ("node_std", node_std),
            ("token_mean", token_mean),
            ("token_std", token_std),
        ):
            target = getattr(self, name)
            v = torch.as_tensor(value, dtype=target.dtype, device=target.device)
            if (
                v.shape != target.shape
                or not bool(torch.isfinite(v).all())
                or ("std" in name and not bool((v > 0).all()))
            ):
                raise ValueError("Finite positive feature normalization required")
            target.copy_(v)

    def token_clock_features(self, batch):
        """Exact four encoder clock inputs for read-only diagnostics/binding."""
        return token_clock_features(
            batch.token_start,
            batch.token_end,
            batch.token_available,
            contract_id=self.cfg.clock_feature_contract_id,
            mode=self.cfg.token_clock_feature_mode,
            position_scale_s=self.cfg.clock_history_position_scale_s,
            duration_scale_s=self.cfg.clock_duration_scale_s,
            delay_scale_s=self.cfg.clock_availability_delay_scale_s,
            legacy_time_scale_s=self.cfg.time_scale_s,
        ).to(self.node_mean.dtype)

    def _tokens(self, batch):
        extra = self.token_clock_features(batch)
        cached = getattr(batch, "utc_token_start_features", None)
        if cached is None:
            utc = self._utc(batch, batch.token_start)
        else:
            self._require_utc(batch)
            if not hasattr(batch, "utc_token_origin_ns") or not hasattr(
                batch, "utc_token_start_seconds"
            ):
                raise ValueError("Cached token phases require exact external clock witnesses")
            if not torch.equal(batch.utc_origin_ns, batch.utc_token_origin_ns) or not torch.equal(
                batch.token_start, batch.utc_token_start_seconds
            ):
                raise ValueError("Cached UTC token clock changed; rebuild external batch")
            if (
                cached.shape != (*batch.token_start.shape, 4)
                or cached.dtype != torch.float64
                or not bool(torch.isfinite(cached).all())
            ):
                raise ValueError("Finite float64 native token UTC phases required")
            utc = cached
        utc = utc.to(self.node_mean.dtype)
        z = torch.cat(((batch.tokens - self.token_mean) / self.token_std, extra, utc), -1)
        y = self.token_encoder(z) + self.kind_embedding(batch.token_kind)
        return torch.cat((self.null_token.expand(batch.size, -1, -1), y), 1)

    def _require_utc(self, batch):
        if getattr(batch, "utc_feature_contract_id", None) != self.cfg.utc_feature_contract_id:
            raise ValueError("Batch UTC feature contract absent or incompatible")
        if not hasattr(batch, "utc_origin_ns"):
            raise ValueError("External window UTC origin required; no relative-time fallback")

    def _utc(self, batch, times):
        self._require_utc(batch)
        return utc_cycle_features(times, batch.utc_origin_ns)

    def _context(self, batch, times, cutoff_s, mode):
        if mode not in ("historical", "causal", "forcing_only"):
            raise ValueError("Unknown information mode")
        if mode != "historical" and any(
            e.meta.get("public_support", {}).get("historical_chemistry_priors")
            for e in batch.external
        ):
            raise ValueError("Undated chemistry is historical only")
        node = self.node_encoder((batch.node_features - self.node_mean) / self.node_std)
        q = len(times)
        t = times / self.cfg.time_scale_s
        features = torch.stack(
            (
                t,
                torch.sin(t / 365.25 * 2 * math.pi),
                torch.cos(t / 365.25 * 2 * math.pi),
                torch.zeros_like(t),
                torch.ones_like(t),
            ),
            -1,
        ).to(self.node_mean.dtype)
        features = torch.cat(
            (
                features[None].expand(batch.size, -1, -1),
                self._utc(batch, times).to(self.node_mean.dtype),
            ),
            -1,
        )
        query = node[:, None] + self.time_encoder(features)[:, :, None]
        keys = self._tokens(batch)
        cap = times if mode == "causal" else torch.full_like(times, cutoff_s)
        cap = torch.minimum(cap, torch.full_like(cap, cutoff_s))
        valid = (
            batch.token_valid[:, None]
            & (batch.token_available[:, None] <= cap[None, :, None])
            & (batch.token_end[:, None] <= cap[None, :, None])
        )
        if mode == "forcing_only":
            valid = valid & (batch.token_kind[:, None] == 0)
        valid = torch.cat(
            (torch.ones(batch.size, q, 1, dtype=torch.bool, device=batch.device), valid), -1
        )
        mask = (
            (~valid)[:, :, None]
            .expand(-1, -1, batch.nodes, -1)
            .reshape(batch.size, q * batch.nodes, -1)
        )
        mask = mask.repeat_interleave(self.cfg.heads, 0)
        y, _ = self.attention(
            query.reshape(batch.size, q * batch.nodes, -1),
            keys,
            keys,
            attn_mask=mask,
            need_weights=False,
        )
        return query + y.reshape(batch.size, q, batch.nodes, -1)

    def _mix(self, value, batch):
        if batch.adjacency.dtype != value.dtype:
            batch = copy(batch)
            batch.adjacency = batch.adjacency.to(value.dtype)
        for block in self.blocks:
            value = block(value, batch)
        if self.cfg.mixed_hidden_feature_normalization == "learned_layernorm":
            value = self.feature_norm(value)
        return value

    def _pool(self, value, batch):
        weight = batch.mass.to(value.dtype)
        if not bool(torch.isfinite(weight).all()) or not bool((weight > 0).all()):
            raise ValueError("Positive external reference masses required for feature pooling")
        weight = weight / weight.sum(-1, keepdim=True)
        return torch.einsum("bn,bqnw->bqw", weight, value)

    def initialize(self, batch, cutoff_s, mode="historical"):
        times = torch.zeros(1, dtype=torch.float64, device=batch.device)
        value = self._mix(self._context(batch, times, cutoff_s, mode), batch)
        return DirectPhaseCarry(
            self.initial_node(value)[:, 0],
            self.initial_global(self._pool(value, batch))[:, 0],
            value[:, 0],
        )

    def decode(self, batch, state, times, start_s, end_s, cutoff_s, mode="historical"):
        if end_s <= start_s or bool(((times < start_s) | (times > end_s)).any()):
            raise ValueError("Query outside positive direct time block")
        fraction = ((times - start_s) / (end_s - start_s)).to(self.node_mean.dtype)
        dt = (times - start_s) / self.cfg.time_scale_s
        features = torch.stack(
            (
                dt,
                torch.sin(dt),
                torch.cos(dt),
                torch.full_like(dt, math.log1p((end_s - start_s) / self.cfg.time_scale_s)),
                fraction.double(),
            ),
            -1,
        ).to(self.node_mean.dtype)
        # UTC is added exactly once in _context using window-relative query time.
        # The unchanged block-relative five features have zero UTC slots here.
        features = torch.cat((features, torch.zeros_like(features[:, :4])), -1)
        carry = torch.cat((state.node_raw, state.memory), -1)
        value = self._context(batch, times, cutoff_s, mode)
        value = value + self.node_carry_encoder(carry)[:, None]
        value = value + self.global_carry_encoder(state.global_raw)[:, None, None]
        value = self._mix(value + self.time_encoder(features)[None, :, None], batch)
        node = state.node_raw[:, None] + fraction[None, :, None, None] * self.delta_node(value)
        global_raw = state.global_raw[:, None] + fraction[None, :, None] * self.delta_global(
            self._pool(value, batch)
        )
        memory = state.memory[:, None] + fraction[None, :, None, None] * self.delta_memory(value)
        return DirectPhaseRaw(node, global_raw, memory, torch.tanh(self.process(value).squeeze(-1)))

    @staticmethod
    def carry(raw, query_index=-1):
        return DirectPhaseCarry(
            raw.node_raw[:, query_index], raw.global_raw[:, query_index], raw.memory[:, query_index]
        )

    def config_dict(self):
        return {
            **asdict(self.cfg),
            "state_layout_id": "DIRECT_PHASE_REFERENCE_LATENT_NODE2_GLOBAL7_NATIVE_DOMAIN_V6",
            "enthalpy_datum": "native_4186_liquid0C",
            "trained": False,
            "phase_initialization_certified": False,
        }
