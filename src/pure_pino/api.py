"""Factories selecting explicit current contracts and packaged native laws."""

from pathlib import Path

import torch

from .clock_features import CLOCK_FEATURE_CONTRACT_ID
from .coupled import DirectPhaseCoupledAdapter
from .model import DirectPhaseInverseOperator, DirectPhaseModelConfig
from .native_bindings import NativeBindings
from .phase import DirectPhaseComponents, inverse_native_liquid_reference
from .surface_melting import SurfaceMeltingBoundary

NATIVE_HASHES = {
    "snow_ice_components": "4710685e7a57f3466d463775cdd143a25de788529a24ed7fbb0f98bcf85f1cc0",
    "weather": "1a6a6633e8602f7c1c951f56ea66fd1de51eedd0b078fa12380f9f883d1522dd",
    "daily_physics": "c1f447c991ed75de484902d32c40460c3ebbb803d625991d653f6eb1d2e13fc7",
    "bed_exchange": "77c5bdfde8106d3f2ac843f45237063ca71b75cd62b820043943e51148b3192f",
}


def create_model(
    node_features: int = 9,
    token_features: int = 9,
    *,
    width: int = 32,
    blocks: int = 2,
    heads: int = 4,
    spectral_modes: int = 4,
    reference_temperature_C: float = 7.0,
    thermal_scale_K: float = 7.0,
    device: str | torch.device = "cpu",
) -> DirectPhaseInverseOperator:
    """Create an untrained model using the new split SI clock.

    The two thermal arguments are explicit external/manufactured candidates,
    never target profile statistics. Real runs must use their frozen SOURCE
    external prior and normalization. This factory does not set the random seed,
    read data, load weights or start optimization.
    """
    config = DirectPhaseModelConfig(
        node_features=node_features,
        token_features=token_features,
        width=width,
        blocks=blocks,
        heads=heads,
        spectral_modes=spectral_modes,
        clock_feature_contract_id=CLOCK_FEATURE_CONTRACT_ID,
        token_clock_feature_mode="split_native_support_affine",
        liquid_enthalpy_coordinate_reference=inverse_native_liquid_reference(
            reference_temperature_C, thermal_scale_K
        ),
        feature_dtype="float64",
    )
    return DirectPhaseInverseOperator(config).to(device)


def native_adapter() -> DirectPhaseCoupledAdapter:
    """Bind the four original pure physics modules; no old time integrator.

    SHA checks reject edited native dependencies. Changing a native law requires
    a reviewed manifest update and new physical qualification. The adapter's
    admission flags remain false until a separate real-data qualification.
    """
    root = Path(__file__).resolve().parent / "_native"
    core = root / "core"
    runtime = root / "runtime"
    bindings = NativeBindings(core, runtime, expected_hashes=NATIVE_HASHES)
    phase = DirectPhaseComponents(
        core,
        expected_hashes={name: NATIVE_HASHES[name] for name in ("snow_ice_components", "weather")},
    )
    return DirectPhaseCoupledAdapter(bindings, phase, SurfaceMeltingBoundary(core))
