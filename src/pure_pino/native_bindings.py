"""Hash-bound native constitutive, partial-cover radiation and bed coefficients.

Only standalone snow/weather/daily/bed utilities are loaded. No state time
integrator, old model, label reader or source-file writer is used. Native cell
volume is the storage measure; face area is a separate radiation/bed measure.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import torch

GEOMETRY_CONTRACT = "NATIVE_STORAGE_FACE_AREA_RADIATION_FV_V1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("Nonfinite native physical input/output; no replacement")


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _tensor(value, reference):
    return torch.as_tensor(value, dtype=torch.float64, device=reference.device)


def _load_standalone(path, expected_sha256):
    """Protect original trees even when caller omits Python's -B flag."""
    path = Path(path).resolve()
    if file_sha256(path) != expected_sha256:
        raise ValueError("Native dependency changed before import: " + str(path))
    path_key = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    name = "_direct_wis_v2_native_" + path.stem + "_" + expected_sha256[:16] + "_" + path_key
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Standalone native module cannot be bound: " + str(path))
    module = importlib.util.module_from_spec(spec)
    # dataclasses inspect sys.modules during execution of the actual module.
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous
    if file_sha256(path) != expected_sha256:
        sys.modules.pop(name, None)
        raise ValueError("Native dependency changed during import: " + str(path))
    return module


class NativeBindings:
    """Native pure functions plus explicitly separate storage/ray metrics."""

    version = "TACO_NATIVE_WIS_PARTIAL_SW_BED_BINDING_V1"
    model_admitted = False
    full_phase_adapter_qualified = False

    def __init__(self, core_root, runtime_root, *, expected_hashes=None):
        engine = Path(core_root).resolve() / "inputs/PINO281_2001_inputs/engine"
        runtime = Path(runtime_root).resolve()
        self.paths = {
            "snow_ice_components": engine
            / "v35_7_enthalpy_window_operator_r1/snow_ice_components.py",
            "weather": engine / "v35_7_g0_coupled_r1/weather.py",
            "daily_physics": runtime / "daily_physics.py",
            "bed_exchange": runtime / "bed_exchange.py",
        }
        for path in self.paths.values():
            if not path.is_file():
                raise FileNotFoundError("Actual native pure dependency missing: " + str(path))
        self.hashes = {name: file_sha256(path) for name, path in self.paths.items()}
        if expected_hashes is not None and self.hashes != dict(expected_hashes):
            raise ValueError("Native dependency hash set changed or incomplete")
        self.modules = {
            name: _load_standalone(path, self.hashes[name]) for name, path in self.paths.items()
        }
        self.snow = self.modules["snow_ice_components"]
        self.weather = self.modules["weather"]
        self.daily = self.modules["daily_physics"]
        self.bed = self.modules["bed_exchange"]
        self.cp_water = float(self.snow.CP_WATER_J_KG_K)
        self.cp_ice = float(self.snow.CP_ICE_J_KG_K)
        self.latent = float(self.snow.LATENT_J_KG)
        if (self.cp_water, self.cp_ice, self.latent) != (4186.0, 2100.0, 334000.0):
            raise ValueError("Native finite-capacity freshwater datum/constants changed")
        self.dependency_code_hashes = {
            str(path): self.hashes[name] for name, path in self.paths.items()
        }
        self.dependency_code_hashes[str(Path(__file__).resolve())] = file_sha256(__file__)

    def reverify_sources(self):
        current = {name: file_sha256(path) for name, path in self.paths.items()}
        if current != self.hashes:
            raise ValueError("Native dependency changed after binding")
        return dict(current)

    @staticmethod
    def validate_geometry(depth_edges_m, face_area_m2, native_volume_m3):
        """Validate inputs without recomputing or projecting native volumes.

        Leading dimensions must broadcast explicitly. For [B,Q] trajectories,
        use [B,1,N+1] geometry and [B,1,N] native volume. No axis inference.
        A trapezoid/native-volume mismatch is diagnostic, never a rejection.
        """
        z = torch.as_tensor(depth_edges_m, dtype=torch.float64)
        a = _tensor(face_area_m2, z)
        v = _tensor(native_volume_m3, z)
        _finite(z, a, v)
        _require(
            z.ndim >= 1 and z.shape[-1] >= 2 and a.shape == z.shape,
            "Depth/face-area support mismatch",
        )
        _require(
            v.ndim >= 1 and v.shape[-1] + 1 == z.shape[-1], "Native storage-volume support mismatch"
        )
        leading = torch.broadcast_shapes(z.shape[:-1], v.shape[:-1])
        z = torch.broadcast_to(z, leading + (z.shape[-1],))
        a = torch.broadcast_to(a, z.shape)
        v = torch.broadcast_to(v, leading + (v.shape[-1],))
        _require(
            (z[..., 0] == 0).all() and (z.diff(dim=-1) > 0).all(),
            "Surface-origin increasing depth required",
        )
        _require(
            (a[..., 0] > 0).all() and (a >= 0).all() and (a.diff(dim=-1) <= 0).all(),
            "Nonnegative nonincreasing radiation face area required",
        )
        _require((v > 0).all(), "Positive original native storage volumes required")
        dz = z.diff(dim=-1)
        trapezoid = 0.5 * (a[..., :-1] + a[..., 1:]) * dz
        mismatch = ((trapezoid - v).abs() / v).amax()
        return (
            z,
            a,
            v,
            {
                "geometry_contract": GEOMETRY_CONTRACT,
                "native_volume_input_values_not_modified": True,
                "native_volume_used_as_storage_not_optical_area_integral": True,
                "linear_area_volume_max_relative_mismatch": float(mismatch.detach()),
                "same_exact_continuous_area_volume_measure": False,
                "moving_bathymetry_certified": False,
                "geometry_source_independence_certified_here": False,
            },
        )

    def bed_parameters(self, face_area_m2, config=None):
        """Unmodified native projected side plus final-bottom bed coefficients."""
        area = torch.as_tensor(face_area_m2, dtype=torch.float64)
        _finite(area)
        _require(
            area.ndim >= 1
            and area.shape[-1] >= 2
            and (area[..., 0] > 0).all()
            and (area >= 0).all()
            and (area.diff(dim=-1) <= 0).all(),
            "Invalid native bed face area",
        )
        config = config or self.bed.BedConfig()
        for name in (
            "volumetric_heat_capacity_j_m3_k",
            "thickness_m",
            "conductivity_w_m_k",
            "mean_temperature_distance_fraction",
        ):
            value = float(getattr(config, name))
            _require(
                math.isfinite(value) and value > 0,
                "Finite positive shared bed material/distance candidate required",
            )
        result = self.bed.bed_parameters_from_hypsography(area, config)
        _finite(
            *(
                result[name]
                for name in ("projected_area_m2", "heat_capacity_j_k", "conductance_w_k")
            )
        )
        return result

    def partial_shortwave(
        self,
        shortwave,
        depth_edges,
        face_area,
        native_volume,
        *,
        ice_fraction,
        snow_fraction_on_ice,
        ice_thickness_m,
        snow_thickness_m,
        water_albedo,
        ice_albedo,
        snow_albedo,
        snow_extinction_m_inv,
        ice_extinction_m_inv,
        kd_visible_m_inv,
        kd_nir_m_inv,
        visible_fraction,
    ):
        """Route SW once: reflection/solids, then transmitted water/side/bottom.

        Native shortwave_partition outputs per TOTAL lake area. Its open-water
        and under-ice terms already paid surface albedo. Their sum enters the
        original two_band_shortwave_partition with albedo exactly zero; no
        second reflection is introduced. Bed radiation is stored in bed only.
        Native V is preserved and checked, not replaced by integral(face A).
        """
        z, a, v, metric = self.validate_geometry(depth_edges, face_area, native_volume)
        q = _tensor(shortwave, z)
        leading = torch.broadcast_shapes(q.shape, z.shape[:-1])
        q = torch.broadcast_to(q, leading)
        z = torch.broadcast_to(z, leading + (z.shape[-1],))
        a = torch.broadcast_to(a, z.shape)
        # Keep the storage input in the declared support; never derive it from A.
        v = torch.broadcast_to(v, leading + (v.shape[-1],))
        values = {
            name: torch.broadcast_to(_tensor(value, z), leading)
            for name, value in {
                "ice_fraction": ice_fraction,
                "snow_fraction_on_ice": snow_fraction_on_ice,
                "ice_thickness_m": ice_thickness_m,
                "snow_thickness_m": snow_thickness_m,
                "water_albedo": water_albedo,
                "ice_albedo": ice_albedo,
                "snow_albedo": snow_albedo,
                "snow_extinction_m_inv": snow_extinction_m_inv,
                "ice_extinction_m_inv": ice_extinction_m_inv,
                "visible_fraction": visible_fraction,
            }.items()
        }
        kv, kn = _tensor(kd_visible_m_inv, z), _tensor(kd_nir_m_inv, z)
        _finite(q, kv, kn, *values.values())
        _require(
            (q >= 0).all() and (kv >= 0).all() and (kn >= 0).all(),
            "Negative native irradiance/optical coefficient",
        )
        for name in (
            "ice_fraction",
            "snow_fraction_on_ice",
            "water_albedo",
            "ice_albedo",
            "snow_albedo",
            "visible_fraction",
        ):
            _require(
                ((values[name] >= 0) & (values[name] <= 1)).all(),
                "Fraction/albedo outside [0,1]: " + name,
            )
        for name in (
            "ice_thickness_m",
            "snow_thickness_m",
            "snow_extinction_m_inv",
            "ice_extinction_m_inv",
        ):
            _require((values[name] >= 0).all(), "Negative covered-layer optical input: " + name)
        surface = self.snow.shortwave_partition(
            q,
            values["ice_fraction"],
            values["snow_fraction_on_ice"],
            values["snow_thickness_m"],
            values["ice_thickness_m"],
            values["water_albedo"],
            values["ice_albedo"],
            values["snow_albedo"],
            values["snow_extinction_m_inv"],
            values["ice_extinction_m_inv"],
        )
        entering = surface["open_water_w_m2"] + surface["under_ice_water_w_m2"]
        liquid = self.daily.two_band_shortwave_partition(
            entering,
            z,
            a,
            kv,
            kd_nir_m_inv=kn,
            visible_fraction=values["visible_fraction"],
            albedo=torch.zeros_like(entering),
        )
        combined = self.bed.combine_bed_radiation(liquid["bed_w"], liquid["bottom_w"])
        area0 = a[..., 0]
        reflected = surface["reflected_w_m2"] * area0
        ice = surface["ice_absorbed_w_m2"] * area0
        snow = surface["snow_absorbed_w_m2"] * area0
        incoming = q * area0
        absorbed = liquid["water_w"].sum(-1) + combined.sum(-1) + ice + snow
        residual = incoming - reflected - absorbed
        _finite(
            liquid["water_w"],
            liquid["bed_w"],
            liquid["bottom_w"],
            combined,
            reflected,
            ice,
            snow,
            residual,
        )
        return {
            "water_w": liquid["water_w"],
            "combined_bed_w": combined,
            "side_bed_w": liquid["bed_w"],
            "bottom_w": liquid["bottom_w"],
            "ice_w": ice,
            "snow_w": snow,
            "reflected_w": reflected,
            "incoming_w": incoming,
            "absorbed_total_w": absorbed,
            "entering_water_w_m2": entering,
            "balance_residual_w": residual,
            "native_surface_balance_residual_w": surface["budget_residual_w_m2"] * area0,
            "native_water_bed_balance_residual_w": liquid["balance_residual_w"],
            "surface_partition_per_lake_area": surface,
            "water_partition": liquid,
            "geometry_diagnostics": metric,
            "surface_albedo_applied_once": True,
            "water_partition_albedo": 0.0,
            "native_cell_count": v.shape[-1],
            "bed_radiation_enters_water_directly": False,
            "model_admitted": False,
            "full_phase_adapter_qualified": False,
        }


# Explicit aliases for callers naming the source API/receipt rather than class.
NativeConstitutiveBindings = NativeBindings
