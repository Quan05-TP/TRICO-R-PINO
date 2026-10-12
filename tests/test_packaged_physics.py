"""Real packaged constitutive laws on manufactured inputs, not lake accuracy."""

import pytest
import torch

from pure_pino import CoupledExternal, native_adapter
from pure_pino.model import DirectPhaseRaw
from pure_pino.phase import (
    inverse_native_liquid_reference,
    inverse_native_solid_reference,
)


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


@pytest.fixture(scope="module")
def adapter():
    return native_adapter()


def external(queries=3):
    return CoupledExternal(
        edges_m=tensor([[0.0, 1.0, 2.0, 3.0]]),
        face_area_m2=tensor([[100.0, 90.0, 70.0, 40.0]]),
        native_volume_m3=tensor([[100.0, 80.0, 50.0]]),
        core9=tensor([10, 2, 3, 1, 101325, 200, 300, 0.001, 0.0005]).expand(1, queries, 9).clone(),
        native_supply_interval_s=torch.full((1, queries), 3600.0, dtype=torch.float64),
        kd_visible_m_inv=torch.full((1, queries), 0.25, dtype=torch.float64),
        kd_nir_m_inv=torch.full((1, queries), 1.5, dtype=torch.float64),
        visible_fraction=torch.full((1, queries), 0.45, dtype=torch.float64),
        thermal_scale_K=7.0,
        source_bindings={"fixture": "manufactured, actual native code"},
    )


def raw_control(x):
    batch, queries, _ = x.core9.shape
    nodes = x.native_volume_m3.shape[-1]
    reference_global_mass = float((1000 * x.native_volume_m3).sum()) * 0.001
    node = torch.zeros((batch, queries, nodes, 2), dtype=torch.float64)
    node[..., 0] = inverse_native_liquid_reference(4.0, x.thermal_scale_K)
    node[..., 1] = 6.0
    global_raw = torch.zeros((batch, queries, 7), dtype=torch.float64)
    global_raw[..., 0] = (10000.0 / reference_global_mass) ** 0.5
    global_raw[..., 1] = (2.0 / reference_global_mass) ** 0.5
    global_raw[..., 2] = inverse_native_solid_reference(-5.0, x.thermal_scale_K)
    global_raw[..., 3] = inverse_native_solid_reference(-2.0, x.thermal_scale_K)
    return DirectPhaseRaw(
        node,
        global_raw,
        torch.zeros((batch, queries, nodes, 4), dtype=torch.float64),
        torch.zeros((batch, queries, nodes), dtype=torch.float64),
    )


def test_actual_native_calorimetry_eos_and_global_inventory_once(adapter):
    x = external()
    decoded = adapter.decode(raw_control(x), x)
    temperature = decoded.observables["water_temperature_C"]
    torch.testing.assert_close(temperature, torch.full_like(temperature, 4.0))
    physical = decoded.state.physical
    recovered = physical.liquid_energy_J / (physical.liquid_mass_kg * adapter.bindings.cp_water)
    torch.testing.assert_close(recovered, temperature, rtol=1e-14, atol=1e-14)
    density = adapter.bindings.daily.eos80_density_atm(temperature, torch.zeros_like(temperature))
    assert torch.isfinite(density).all()
    assert ((density > 990) & (density < 1010)).all()
    assert decoded.inventory_energy_J.shape == (1, 3, 8)
    expected_mass = torch.full((1, 3), 230000.0 + 10000.0 + 2.0, dtype=torch.float64)
    torch.testing.assert_close(decoded.inventory_mass_kg.sum(-1), expected_mass)
    assert adapter.synthetic_only is False
    assert decoded.qualification["salinity_measured"] is False


def test_native_solar_water_bed_route_and_internal_pairs_cancel(adapter):
    x = external()
    raw = raw_control(x)
    powers = adapter.powers(raw, adapter.decode(raw, x), x)
    torch.testing.assert_close(
        powers.internal_energy_W.sum(-1),
        tensor([[0.0, 0.0, 0.0]]),
        rtol=0,
        atol=1e-9,
    )
    torch.testing.assert_close(
        powers.internal_mass_kg_s.sum(-1),
        tensor([[0.0, 0.0, 0.0]]),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        sum(powers.external_components_W.values()),
        powers.external_energy_W,
        rtol=1e-14,
        atol=1e-9,
    )
    assert powers.external_components_W["solar_bed"][..., :5].abs().sum() == 0
    assert powers.external_components_W["solar_bed"][..., 5:].sum() > 0
    assert powers.diagnostics["bed_to_water_W"].sum() > 0
    torch.testing.assert_close(
        powers.diagnostics["shortwave_budget_residual_W"],
        tensor([[0.0, 0.0, 0.0]]),
        rtol=0,
        atol=1e-9,
    )
    # Local flux cancellation is not a certificate that a learned trajectory
    # obeys its interval energy/mass balances or predicts a real lake accurately.


@pytest.mark.parametrize("field", [5, 7, 8])
def test_negative_native_forcing_rejected_without_mutating_input(adapter, field):
    x = external()
    x.core9[0, 1, field] = -1e-7
    original = x.core9.clone()
    with pytest.raises(ValueError):
        adapter.decode(raw_control(x), x)
    torch.testing.assert_close(x.core9, original, rtol=0, atol=0)
