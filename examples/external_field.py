"""Decode an exterior-only, manufactured 64-cell field on the CPU.

The small weather/geometry arrays and normalization below are interface
fixtures, not certified lake data or fitted TRAIN statistics. No temperature
labels, optimizer, accuracy metric, or lake-state time integrator is used.
The two 24-hour blocks pass learned latent carry to the next direct decoder.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

from pure_pino import CoupledExternal, create_model, native_adapter
from pure_pino.utc_features import UTC_FEATURE_CONTRACT_ID

HOUR_S = 3600.0
THERMAL_SCALE_K = 7.0
WEATHER = (10.0, 2.0, 3.0, 1.0, 101325.0, 200.0, 300.0, 0.001, 0.0005)
TOKEN_STD = (10.0, 10.0, 5.0, 5.0, 10000.0, 200.0, 200.0, 0.001, 0.001)


def exterior_batch(nodes: int = 64) -> SimpleNamespace:
    """Build a manufactured batch using raw seconds and Gregorian UTC."""
    if nodes < 2:
        raise ValueError("At least two column cells are required")
    dtype = torch.float64
    adjacency = torch.eye(nodes, dtype=dtype)
    indices = torch.arange(nodes - 1)
    adjacency[indices, indices + 1] = 1.0
    adjacency[indices + 1, indices] = 1.0
    adjacency /= adjacency.sum(dim=-1, keepdim=True)
    starts = torch.tensor(
        [[-HOUR_S, 0.0, 6 * HOUR_S, 12 * HOUR_S, 24 * HOUR_S, 36 * HOUR_S]],
        dtype=dtype,
    )
    tokens = torch.tensor(WEATHER, dtype=dtype).expand(1, 6, 9).clone()
    tokens[0, :, 0] += torch.linspace(-1.0, 1.0, 6, dtype=dtype)
    origin = int(datetime(2020, 6, 1, tzinfo=timezone.utc).timestamp()) * 10**9
    return SimpleNamespace(
        size=1,
        nodes=nodes,
        columns=1,
        levels=nodes,
        device=torch.device("cpu"),
        node_features=torch.arange(nodes * 9, dtype=dtype).reshape(1, nodes, 9) / (nodes * 9),
        mass=torch.full((1, nodes), 1000.0, dtype=dtype),
        adjacency=adjacency[None],
        tokens=tokens,
        token_start=starts,
        token_end=starts + HOUR_S,
        token_available=starts + HOUR_S,
        token_valid=torch.ones((1, 6), dtype=torch.bool),
        token_kind=torch.zeros((1, 6), dtype=torch.long),
        utc_origin_ns=torch.tensor([origin], dtype=torch.int64),
        utc_feature_contract_id=UTC_FEATURE_CONTRACT_ID,
        external=[SimpleNamespace(meta={"fixture": "manufactured exterior"})],
    )


def configure_fixture_normalization(model) -> None:
    """Set explicit illustrative scales, without consulting any target values."""
    model.set_normalization(
        torch.zeros(9, dtype=torch.float64),
        torch.ones(9, dtype=torch.float64),
        torch.tensor(WEATHER, dtype=torch.float64),
        torch.tensor(TOKEN_STD, dtype=torch.float64),
    )


def coupled_exterior(
    nodes: int, queries: int, thermal_scale_K: float = THERMAL_SCALE_K
) -> CoupledExternal:
    """Manufactured physical support for actual packaged native laws."""
    if nodes < 2 or queries < 1:
        raise ValueError("Positive query count and at least two cells required")
    dtype = torch.float64
    edges = torch.linspace(0.0, 10.0, nodes + 1, dtype=dtype)[None]
    area = torch.linspace(1000.0, 500.0, nodes + 1, dtype=dtype)[None]
    volume = 0.5 * (area[:, :-1] + area[:, 1:]) * edges.diff(dim=-1)
    return CoupledExternal(
        edges_m=edges,
        face_area_m2=area,
        native_volume_m3=volume,
        core9=torch.tensor(WEATHER, dtype=dtype).expand(1, queries, 9).clone(),
        native_supply_interval_s=torch.full((1, queries), HOUR_S, dtype=dtype),
        kd_visible_m_inv=torch.full((1, queries), 0.25, dtype=dtype),
        kd_nir_m_inv=torch.full((1, queries), 1.5, dtype=dtype),
        visible_fraction=torch.full((1, queries), 0.45, dtype=dtype),
        thermal_scale_K=thermal_scale_K,
        source_bindings={"fixture": "manufactured geometry and weather"},
    )


def make_fixture_model(width: int = 16):
    model = create_model(
        node_features=9,
        token_features=9,
        width=width,
        blocks=1,
        heads=2,
        spectral_modes=2,
        reference_temperature_C=7.0,
        thermal_scale_K=THERMAL_SCALE_K,
    )
    configure_fixture_normalization(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "example.json",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["schema"] != "PURE_PINO_MANUFACTURED_INTERFACE_EXAMPLE/1":
        parser.error("This example accepts only its manufactured fixture schema")
    if config["optimizer_updates"] != 0 or config["target_values_used"]:
        parser.error("The interface example cannot train or use target values")
    if config["direct_block_hours"] != 24 or config["geometry"]["nodes"] != 64:
        parser.error("The example demonstrates exactly two 24-hour, 64-cell blocks")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(config["seed"] if args.seed is None else args.seed)
    batch = exterior_batch()
    model = create_model(**config["model"]).eval()
    configure_fixture_normalization(model)
    if model.cfg.clock_feature_contract_id != config["factory_clock_contract"]:
        parser.error("Factory clock contract differs from the example configuration")
    if model.cfg.token_clock_feature_mode != config["factory_token_clock_feature_mode"]:
        parser.error("Factory clock mode differs from the example configuration")
    adapter = native_adapter()
    cutoff = 48 * HOUR_S
    blocks = []
    with torch.inference_mode():
        state = model.initialize(batch, cutoff, "historical")
        for start in (0.0, 24 * HOUR_S):
            end = start + 24 * HOUR_S
            times = torch.tensor([start, start + 12 * HOUR_S, end])
            times = times.to(dtype=torch.float64)
            raw = model.decode(batch, state, times, start, end, cutoff)
            decoded = adapter.decode(
                raw, coupled_exterior(64, len(times), config["model"]["thermal_scale_K"])
            )
            temperature = decoded.observables["water_temperature_C"]
            blocks.append(
                {
                    "start_s": start,
                    "end_s": end,
                    "raw_node_shape": list(raw.node_raw.shape),
                    "temperature_shape": list(temperature.shape),
                    "predicted_temperature_min_C": float(temperature.min()),
                    "predicted_temperature_max_C": float(temperature.max()),
                    "max_latent_boundary_difference": float(
                        (raw.node_raw[:, 0] - state.node_raw).abs().max()
                    ),
                }
            )
            state = model.carry(raw)
    print(
        json.dumps(
            {
                "scope": "manufactured_external_interface_example",
                "optimizer_updates": 0,
                "target_values_read": 0,
                "lake_accuracy_evaluated": False,
                "real_data_or_phase_population_qualified": False,
                "clock_contract": model.cfg.clock_feature_contract_id,
                "blocks": blocks,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
