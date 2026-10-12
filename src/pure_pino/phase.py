"""Direct freshwater water/ice/snow inventories and continuous-time ledgers.

This is an independently versioned component, not the full model adapter.
It never advances temperature, adjusts a supplied state to satisfy a budget,
reads internal lake observations, or imports CoupledLake/enthalpy_graph_step.
Native constitutive files are hash-bound. Liquid reference enthalpy is 0 at 0C.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

LAYOUT = "TACO_DIRECT_WATER_ICE_SNOW_BED_EXTENSIVE_COMPONENT_V1"
RESERVOIR_NAMES = ("liquid", "ice", "snow", "bed")
LIQUID_COORDINATE_CONTRACT = "NATIVE_REFERENCE_WATER_LIQUID0_40_RATIONAL_ENTHALPY_V5"
LIQUID_NATIVE_LATENT_CEILING_C = 40.0 * (1.0 - 8.0 * torch.finfo(torch.float64).eps)
SOLID_COORDINATE_CONTRACT = "NATIVE_SOLID100K_0C_RATIONAL_COLD_ENTHALPY_V6"
# The unchanged weather.atmosphere rejects Ts+273.15 <=100 K.  This is a
# definition of the learned extensive state, not a post-decoding projection.
# 4096 binary64 eps guard the m*h -> E/m -> native phase round trip.
SOLID_NATIVE_COLD_CEILING_K = (273.15 - 100.0) * (1.0 - 4096.0 * torch.finfo(torch.float64).eps)


def native_domain_solid_temperature(raw, thermal_scale_K):
    """Solid T=-B*u/(B+u), u=scale*r^2; exact melt point at r=0.

    Mass/area definitions and all energy carriers use this same decoded state.
    Nonfinite coordinates/overflow fail rather than being replaced or clipped.
    """
    finite(raw)
    scale = scalar_like(thermal_scale_K, raw)
    require(
        torch.isfinite(scale).all() and (scale > 0).all(),
        "Finite positive exterior solid thermal scale required",
    )
    u = raw.square() * scale
    finite(u)
    temperature = -SOLID_NATIVE_COLD_CEILING_K * (u / (SOLID_NATIVE_COLD_CEILING_K + u))
    finite(temperature)
    return temperature


def inverse_native_solid_reference(reference_C, thermal_scale_K):
    """Inverse of a declared exterior/manufactured solid candidate, never truth."""
    require(
        math.isfinite(reference_C) and -SOLID_NATIVE_COLD_CEILING_K < reference_C <= 0,
        "Solid initialization candidate outside native atmosphere domain",
    )
    require(
        math.isfinite(thermal_scale_K) and thermal_scale_K > 0,
        "Finite positive exterior solid thermal scale required",
    )
    cold = -reference_C
    return math.sqrt(
        SOLID_NATIVE_COLD_CEILING_K
        * cold
        / ((SOLID_NATIVE_COLD_CEILING_K - cold) * thermal_scale_K)
    )


def native_domain_liquid_temperature(raw, thermal_scale_K, upper_C=40.0):
    """Declared smooth latent definition; no decoded-temperature projection.

    T=L*u/(L+u), u=s*r^2. Exact r=0 retains the native 0C branch and its
    true zero derivative. Very large raw/scale overflow remains a failure.
    """
    finite(raw)
    require(
        math.isfinite(float(upper_C)) and upper_C == 40.0,
        "Only the documented native reference-water 0..40C domain is bound",
    )
    scale = scalar_like(thermal_scale_K, raw)
    require(
        torch.isfinite(scale).all() and (scale > 0).all(),
        "Finite positive exterior thermal scale required",
    )
    u = raw.square() * scale
    finite(u)
    # The predefined latent ceiling leaves 8 binary64 eps of the native 40C
    # ceiling for m*CP*T -> E/m/CP roundoff. The observed/predicted T is never
    # clipped after decoding, and the native EOS <=40 assertion stays exact.
    ceiling = LIQUID_NATIVE_LATENT_CEILING_C
    temperature = ceiling * (u / (ceiling + u))
    finite(temperature)
    return temperature


def inverse_native_liquid_reference(reference_C, thermal_scale_K, upper_C=40.0):
    """Inverse mapping of the frozen ALL-TRAIN external-air candidate only."""
    require(
        math.isfinite(reference_C)
        and 0 <= reference_C < LIQUID_NATIVE_LATENT_CEILING_C
        and upper_C == 40.0,
        "External liquid initialization candidate outside native model domain",
    )
    require(
        math.isfinite(thermal_scale_K) and thermal_scale_K > 0,
        "Finite positive exterior thermal scale required",
    )
    ceiling = LIQUID_NATIVE_LATENT_CEILING_C
    return math.sqrt(ceiling * reference_C / ((ceiling - reference_C) * thermal_scale_K))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def finite(*values):
    if any(not bool(torch.isfinite(v).all()) for v in values):
        raise ValueError("Nonfinite state or flux; no missing-value replacement")


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def scalar_like(value, reference):
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


@dataclass
class PhysicalState:
    # Leading axes can be batch/query; the final liquid/bed axis is spatial.
    liquid_mass_kg: torch.Tensor
    liquid_energy_J: torch.Tensor
    ice_mass_kg: torch.Tensor
    ice_energy_J: torch.Tensor
    snow_mass_kg: torch.Tensor
    snow_energy_J: torch.Tensor
    ice_area_fraction: torch.Tensor
    snow_fraction_on_ice: torch.Tensor
    snow_density_kg_m3: torch.Tensor
    bed_energy_J: torch.Tensor
    bed_capacity_J_K: torch.Tensor
    liquid_salt_mass_kg: torch.Tensor
    ice_salt_mass_kg: torch.Tensor
    snow_salt_mass_kg: torch.Tensor


@dataclass
class DecodedState:
    physical: PhysicalState
    liquid_temperature_C: torch.Tensor
    liquid_valid_mask: torch.Tensor
    ice_temperature_C: torch.Tensor
    snow_temperature_C: torch.Tensor
    ice_present: torch.Tensor
    snow_present: torch.Tensor
    bed_temperature_C: torch.Tensor
    reservoir_mass_kg: torch.Tensor
    reservoir_energy_J: torch.Tensor
    reservoir_salt_kg: torch.Tensor
    total_mass_kg: torch.Tensor
    total_energy_J: torch.Tensor
    total_salt_kg: torch.Tensor
    qualification: dict


class DirectPhaseComponents:
    """Use only native standalone pure phase, optical and weather functions."""

    def __init__(self, taco_core_evidence_root, *, expected_hashes=None):
        root = Path(taco_core_evidence_root)
        engine = root / "inputs/PINO281_2001_inputs/engine"
        self.paths = {
            "snow_ice_components": engine
            / "v35_7_enthalpy_window_operator_r1/snow_ice_components.py",
            "weather": engine / "v35_7_g0_coupled_r1/weather.py",
        }
        self.hashes = {name: sha256(path) for name, path in self.paths.items()}
        if expected_hashes is not None and self.hashes != expected_hashes:
            raise ValueError("Constitutive source hash changed")
        self.modules = {}
        previous_bytecode = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            for name, path in self.paths.items():
                spec = importlib.util.spec_from_file_location("direct_phase_native_" + name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.modules[name] = module
        finally:
            sys.dont_write_bytecode = previous_bytecode
        native = self.modules["snow_ice_components"]
        self.cp_water = float(native.CP_WATER_J_KG_K)
        self.cp_ice = float(native.CP_ICE_J_KG_K)
        self.latent = float(native.LATENT_J_KG)
        if (self.cp_water, self.cp_ice, self.latent) != (4186.0, 2100.0, 334000.0):
            raise ValueError("Native enthalpy datum/constant version changed")

    def decode(self, state: PhysicalState, *, chemistry_status="unknown_reference_water"):
        """Validate supplied extensive state without projecting/clipping it.

        All P0 masses describe LIQUID, never a second representation of floating
        ice. Dry cells have M=E=salt=0 and an explicit invalid observation mask.
        Global ice/snow reservoirs are counted once. A salt bookkeeping test is
        possible separately, but this native phase decoder is freshwater only.
        """
        s = state
        finite(*vars(s).values())
        leading = s.liquid_mass_kg.shape[:-1]
        require(s.liquid_mass_kg.ndim >= 1, "Liquid spatial axis required")
        for name in ("liquid_energy_J", "liquid_salt_mass_kg"):
            require(getattr(s, name).shape == s.liquid_mass_kg.shape, "Liquid support mismatch")
        for name in (
            "ice_mass_kg",
            "ice_energy_J",
            "snow_mass_kg",
            "snow_energy_J",
            "ice_area_fraction",
            "snow_fraction_on_ice",
            "snow_density_kg_m3",
            "ice_salt_mass_kg",
            "snow_salt_mass_kg",
        ):
            require(
                getattr(s, name).shape == leading,
                "Global inventory must have one value per lake/query",
            )
        require(
            s.bed_energy_J.shape[:-1] == leading
            and s.bed_energy_J.shape == s.bed_capacity_J_K.shape,
            "Bed support/capacity mismatch",
        )
        require((s.bed_capacity_J_K >= 0).all(), "Frozen external bed capacity must be nonnegative")
        require(
            ((s.bed_capacity_J_K > 0) | (s.bed_energy_J == 0)).all(),
            "Absent bed cannot retain energy",
        )
        for mass, energy in (
            (s.liquid_mass_kg, s.liquid_energy_J),
            (s.ice_mass_kg, s.ice_energy_J),
            (s.snow_mass_kg, s.snow_energy_J),
        ):
            require((mass >= 0).all(), "Negative inventory mass")
            require(((mass != 0) | (energy == 0)).all(), "Zero mass cannot retain energy")
        require(
            (s.liquid_energy_J >= 0).all(),
            "Liquid compartment has negative enthalpy; model phase transfer explicitly",
        )
        for mass, energy in ((s.ice_mass_kg, s.ice_energy_J), (s.snow_mass_kg, s.snow_energy_J)):
            require((energy <= -self.latent * mass).all(), "Frozen reservoir is not entirely solid")
        for frac in (s.ice_area_fraction, s.snow_fraction_on_ice):
            require(((frac >= 0) & (frac <= 1)).all(), "Area fraction outside [0,1]")
        ip, sp = s.ice_mass_kg > 0, s.snow_mass_kg > 0
        require((ip == (s.ice_area_fraction > 0)).all(), "Ice mass/area support disagree")
        require((sp == (s.snow_fraction_on_ice > 0)).all(), "Snow mass/area support disagree")
        require((~sp | ip).all(), "Unsupported snow after ice disappearance")
        require(
            ((s.snow_density_kg_m3 > 0) & (s.snow_density_kg_m3 <= 917)).all(),
            "Frozen porous snow density must be in (0,917]",
        )
        salt = torch.cat(
            (s.liquid_salt_mass_kg, s.ice_salt_mass_kg[..., None], s.snow_salt_mass_kg[..., None]),
            -1,
        )
        require((salt >= 0).all(), "Negative salt inventory")
        require(
            (salt == 0).all(), "Native freshwater constitutive law cannot certify saline freezing"
        )
        require(
            chemistry_status in ("unknown_reference_water", "externally_verified_freshwater"),
            "Explicit external chemistry status required",
        )
        wet = s.liquid_mass_kg > 0
        liquid_h = s.liquid_energy_J / torch.where(
            wet, s.liquid_mass_kg, torch.ones_like(s.liquid_mass_kg)
        )
        ice_h = s.ice_energy_J / torch.where(ip, s.ice_mass_kg, torch.ones_like(s.ice_mass_kg))
        snow_h = s.snow_energy_J / torch.where(sp, s.snow_mass_kg, torch.ones_like(s.snow_mass_kg))
        phase = self.modules["snow_ice_components"].freshwater_phase
        lt = phase(liquid_h)["phase_temperature_c"]
        it = phase(ice_h)["phase_temperature_c"]
        st = phase(snow_h)["phase_temperature_c"]
        require(
            (lt + 273.15 > 0).all() and (it + 273.15 > 0).all() and (st + 273.15 > 0).all(),
            "Nonphysical absolute temperature",
        )
        zero = torch.zeros_like(s.ice_mass_kg)
        masses = torch.stack((s.liquid_mass_kg.sum(-1), s.ice_mass_kg, s.snow_mass_kg, zero), -1)
        energies = torch.stack(
            (s.liquid_energy_J.sum(-1), s.ice_energy_J, s.snow_energy_J, s.bed_energy_J.sum(-1)), -1
        )
        salts = torch.stack(
            (s.liquid_salt_mass_kg.sum(-1), s.ice_salt_mass_kg, s.snow_salt_mass_kg, zero), -1
        )
        bt = s.bed_energy_J / torch.where(
            s.bed_capacity_J_K > 0, s.bed_capacity_J_K, torch.ones_like(s.bed_capacity_J_K)
        )
        require((bt + 273.15 > 0).all(), "Nonphysical bed absolute temperature")
        return DecodedState(
            s,
            lt,
            wet,
            it,
            st,
            ip,
            sp,
            bt,
            masses,
            energies,
            salts,
            masses.sum(-1),
            energies.sum(-1),
            salts.sum(-1),
            {
                "state_layout_id": LAYOUT,
                "enthalpy_reference": "liquid_water_0C_zero_J_kg",
                "cp_water_J_kg_K": self.cp_water,
                "cp_ice_J_kg_K": self.cp_ice,
                "latent_J_kg": self.latent,
                "chemistry_status": chemistry_status,
                "chemistry_is_measured": chemistry_status == "externally_verified_freshwater",
                "liquid_EOS_density_certified": False,
                "temperature_upper_40C_clip": False,
                "liquid_mass_volume_constraint_certified": False,
                "moving_mass_coordinate_certified": False,
                "mass_and_energy_support": "variable_liquid_P0_plus_single_global_ice_snow_plus_bed",
                "model_admitted": False,
                "full_phase_adapter_qualified": False,
            },
        )

    def raw_to_physical(
        self,
        raw: dict,
        *,
        liquid_mass_scale,
        global_mass_scale,
        liquid_h_scale,
        cold_temperature_scale,
        bed_capacity_J_K,
        liquid_temperature_upper_C=None,
    ):
        """Declared new latent-coordinate layout, never old water_h reinterpretation.

        Squared coordinates allow exact zero inventories/phase-boundary states.
        Their zero derivative there is retained and tested, not replaced by a
        straight-through estimator. Mass budgets remain residuals, not a hidden
        projection. Smooth mass-supported coverage vanishes at disappearance.
        """
        finite(*raw.values())
        ref = raw["liquid_mass_coordinate"]
        scales = [
            scalar_like(x, ref)
            for x in (liquid_mass_scale, global_mass_scale, liquid_h_scale, cold_temperature_scale)
        ]
        require(
            all(bool((s > 0).all()) for s in scales), "Positive frozen external scales required"
        )
        ml = raw["liquid_mass_coordinate"].square() * scales[0]
        mi = raw["ice_mass_coordinate"].square() * scales[1]
        ms = raw["snow_mass_coordinate"].square() * scales[1]
        if liquid_temperature_upper_C is None:
            hl = raw["liquid_enthalpy_coordinate"].square() * scales[2]
        else:
            hl = self.cp_water * native_domain_liquid_temperature(
                raw["liquid_enthalpy_coordinate"],
                scales[2] / self.cp_water,
                liquid_temperature_upper_C,
            )
        hi = -self.latent + self.cp_ice * native_domain_solid_temperature(
            raw["ice_cold_coordinate"], scales[3]
        )
        hs = -self.latent + self.cp_ice * native_domain_solid_temperature(
            raw["snow_cold_coordinate"], scales[3]
        )
        fi = mi / (mi + scales[1]) * torch.sigmoid(raw["ice_area_coordinate"])
        fs = ms / (ms + scales[1]) * torch.sigmoid(raw["snow_area_coordinate"])
        rho = 50.0 + 867.0 * torch.sigmoid(raw["snow_density_coordinate"])
        cb = scalar_like(bed_capacity_J_K, ref)
        eb = raw["bed_temperature_C"] * cb
        cb = torch.broadcast_to(cb, eb.shape)
        return PhysicalState(
            ml,
            ml * hl,
            mi,
            mi * hi,
            ms,
            ms * hs,
            fi,
            fs,
            rho,
            eb,
            cb,
            torch.zeros_like(ml),
            torch.zeros_like(mi),
            torch.zeros_like(ms),
        )

    def atmosphere_carrier(
        self,
        core9,
        surface_C,
        bulk_coefficient,
        area_m2,
        condensed_h_J_kg,
        *,
        frozen,
        max_vapor=None,
    ):
        """Native latent power plus separate advected condensed-phase enthalpy.

        max_vapor, if used, must come from declared continuous interval supply;
        this function does not choose a numerical integration step or cap.
        """
        finite(core9, surface_C, bulk_coefficient, area_m2, condensed_h_J_kg)
        if max_vapor is not None:
            finite(max_vapor)
            require(
                (max_vapor >= 0).all(), "Finite nonnegative explicitly supported vapor cap required"
            )
        require(
            (area_m2 >= 0).all() and (bulk_coefficient >= 0).all(), "Invalid atmospheric support"
        )
        native = self.modules["weather"]
        weather = native.physical_weather(core9, partition_policy=native.PARTITION_POLICY)
        q, dq, vapor = native.atmosphere(
            surface_C, weather, bulk_coefficient, frozen=frozen, max_vapor=max_vapor
        )
        finite(q, dq, vapor)
        mass = -vapor * area_m2
        carrier = mass * condensed_h_J_kg
        return {
            "mass_kg_s": mass,
            "atmospheric_power_W": q * area_m2,
            "condensed_carrier_power_W": carrier,
            "total_external_power_W": q * area_m2 + carrier,
            "vapor_kg_m2_s": vapor,
            "surface_derivative_W_m2_K": dq,
            "latent_already_in_atmospheric_power": True,
            "precipitation_partition_policy": native.PARTITION_POLICY,
            "snowfall_adjustment_m_hour": weather["sf_adjustment_m_hour"],
        }

    def precipitation_carrier(self, core9, area_m2, ice_fraction):
        """Native precipitation split and explicitly MODELLED carrier T=air.

        No observed precipitation-temperature claim; original CORE9 untouched.
        Source policy reconciles snowfall to immutable total precipitation and
        reports the adjustment. Negative TP/SW still fail native validation.
        """
        finite(core9, area_m2, ice_fraction)
        require(
            (area_m2 >= 0).all() and ((ice_fraction >= 0) & (ice_fraction <= 1)).all(),
            "Bad precipitation support",
        )
        native = self.modules["weather"]
        w = native.physical_weather(core9, partition_policy=native.PARTITION_POLICY)
        rain = 1000.0 * w["rain_m_hour"] * area_m2 / 3600.0
        snow = 1000.0 * w["snow_m_hour"] * area_m2 / 3600.0
        rain_h = self.cp_water * w["air"].clamp_min(0)
        snow_h = -self.latent + self.cp_ice * w["air"].clamp_max(0)
        z = torch.zeros_like(rain)
        m = torch.stack((rain + (1 - ice_fraction) * snow, z, ice_fraction * snow, z), -1)
        e = torch.stack(
            (
                rain * rain_h + (1 - ice_fraction) * snow * snow_h,
                z,
                ice_fraction * snow * snow_h,
                z,
            ),
            -1,
        )
        return {
            "reservoir_mass_kg_s": m,
            "reservoir_energy_W": e,
            "total_mass_kg_s": m.sum(-1),
            "total_energy_W": e.sum(-1),
            "precipitation_carrier_temperature_policy": "native_model_air_temperature_rain_above0_snow_below0_not_observed",
            "snowfall_adjustment_m_hour": w["sf_adjustment_m_hour"],
        }

    def phase_transfer(self, rate_kg_s, *, donor, receiver, heat_reservoir):
        """At-Tf transfer with matching latent heat into/from a named reservoir.

        This is a POWER RHS, not a mass/temperature time step. Cold bulk donor
        warming must be in its separate energy residual. Salt is excluded from
        pure frozen phase and remains liquid; no fabricated saline Tf.
        """
        finite(rate_kg_s)
        require((rate_kg_s >= 0).all(), "Use positive directed transfer rates")
        require(
            donor in ("liquid", "ice", "snow")
            and receiver in ("liquid", "ice", "snow")
            and donor != receiver,
            "Invalid phase reservoir",
        )
        require(
            heat_reservoir in RESERVOIR_NAMES, "Name the reservoir paying/receiving latent heat"
        )
        hs = {"liquid": 0.0, "ice": -self.latent, "snow": -self.latent}
        m = torch.zeros((*rate_kg_s.shape, 4), dtype=rate_kg_s.dtype, device=rate_kg_s.device)
        e = torch.zeros_like(m)
        i, j, k = (RESERVOIR_NAMES.index(x) for x in (donor, receiver, heat_reservoir))
        m[..., i] = -rate_kg_s
        m[..., j] = rate_kg_s
        e[..., i] = -rate_kg_s * hs[donor]
        e[..., j] = rate_kg_s * hs[receiver]
        e[..., k] = e[..., k] + rate_kg_s * (hs[donor] - hs[receiver])
        return {
            "reservoir_mass_kg_s": m,
            "reservoir_energy_W": e,
            "reservoir_salt_kg_s": torch.zeros_like(m),
            "internal_mass_residual_kg_s": m.sum(-1),
            "internal_energy_residual_W": e.sum(-1),
        }


def directed_carrier(rate_kg_s, carrier_h_J_kg, carrier_salt_mass_fraction, *, donor, receiver):
    """General non-phase internal carrier (including salt), paired exactly."""
    finite(rate_kg_s, carrier_h_J_kg, carrier_salt_mass_fraction)
    require((rate_kg_s >= 0).all(), "Positive directed rate required")
    require(
        ((carrier_salt_mass_fraction >= 0) & (carrier_salt_mass_fraction <= 1)).all(),
        "Invalid salt fraction",
    )
    require(
        donor != receiver and donor in RESERVOIR_NAMES and receiver in RESERVOIR_NAMES,
        "Invalid internal edge",
    )
    m = torch.zeros((*rate_kg_s.shape, 4), dtype=rate_kg_s.dtype, device=rate_kg_s.device)
    i, j = RESERVOIR_NAMES.index(donor), RESERVOIR_NAMES.index(receiver)
    m[..., i], m[..., j] = -rate_kg_s, rate_kg_s
    return {
        "reservoir_mass_kg_s": m,
        "reservoir_energy_W": m * carrier_h_J_kg[..., None],
        "reservoir_salt_kg_s": m * carrier_salt_mass_fraction[..., None],
    }


def interval_balance(time_s, reservoir_inventory, reservoir_rate):
    """Autograd-safe exact endpoints minus declared quadrature, never rollout.

    Time is penultimate axis; reservoir is final axis. Provides individual and
    whole-lake residual. Checking only the sum could conceal compensating errors.
    """
    finite(time_s, reservoir_inventory, reservoir_rate)
    require(time_s.ndim == 1 and time_s.numel() >= 2, "At least two declared time nodes required")
    require((time_s[1:] > time_s[:-1]).all(), "Nonmonotone or duplicate quadrature nodes")
    require(
        reservoir_inventory.shape == reservoir_rate.shape
        and reservoir_inventory.shape[-2] == time_s.numel(),
        "Time/inventory/rate support mismatch",
    )
    dt = time_s[1:] - time_s[:-1]
    integral = (
        (reservoir_rate[..., 1:, :] + reservoir_rate[..., :-1, :]) * 0.5 * dt[..., None]
    ).sum(-2)
    residual = reservoir_inventory[..., -1, :] - reservoir_inventory[..., 0, :] - integral
    return {
        "per_reservoir_residual": residual,
        "total_residual": residual.sum(-1),
        "integrated_rate": integral,
        "quadrature": "declared_trapezoid_nodes_no_integrator",
    }


def floating_geometry(
    decoded: DecodedState, area_m2, liquid_surface_elevation_m, *, rho_water=1000.0, rho_ice=917.0
):
    """Conditional uniform-floating, undrowned-snow hydrostatics.

    Water level must be supplied by a declared external/model water balance.
    Native dual V/A alone does not certify a moving bathymetry inverse. Snow
    loading is averaged over ice; flooding, snow-ice and subgrid flexure absent.
    """
    s = decoded.physical
    finite(area_m2, liquid_surface_elevation_m)
    require(
        area_m2.shape == s.ice_mass_kg.shape and liquid_surface_elevation_m.shape == area_m2.shape,
        "Global area/elevation support mismatch",
    )
    require(
        (area_m2 > 0).all() and rho_water > 0 and rho_ice > 0,
        "Positive geometry densities required",
    )
    ai, ass = area_m2 * s.ice_area_fraction, area_m2 * s.ice_area_fraction * s.snow_fraction_on_ice
    hi = s.ice_mass_kg / (rho_ice * torch.where(ai > 0, ai, torch.ones_like(ai)))
    hs = s.snow_mass_kg / (s.snow_density_kg_m3 * torch.where(ass > 0, ass, torch.ones_like(ass)))
    draft = (s.ice_mass_kg + s.snow_mass_kg) / (
        rho_water * torch.where(ai > 0, ai, torch.ones_like(ai))
    )
    freeboard = hi - draft
    require(
        (freeboard >= 0).all(),
        "Negative freeboard requires flooding/snow-ice physics, never clamp to zero",
    )
    level = liquid_surface_elevation_m
    return {
        "ice_thickness_m": hi,
        "snow_thickness_m": hs,
        "draft_m": draft,
        "freeboard_m": freeboard,
        "ice_base_elevation_m": level - draft,
        "ice_top_elevation_m": level + freeboard,
        "snow_top_elevation_m": level + freeboard + hs,
        "assumption": "uniform_floating_ice_mean_snow_load_no_flooding_external_water_level",
        "moving_bathymetry_certified": False,
        "model_admitted": False,
    }


def observation_elevation(depth_m, *, datum, liquid_surface_elevation_m, geometry=None):
    """Explicit reference only; unknown datum is not silently below ice."""
    finite(depth_m, liquid_surface_elevation_m)
    require((depth_m >= 0).all(), "Depth is nonnegative below the declared reference")
    if datum == "below_liquid_surface":
        return liquid_surface_elevation_m - depth_m
    if datum in ("below_ice_base", "below_ice_top", "below_snow_top"):
        require(geometry is not None, "Ice/snow datum requires conditional draft geometry")
        key = {
            "below_ice_base": "ice_base_elevation_m",
            "below_ice_top": "ice_top_elevation_m",
            "below_snow_top": "snow_top_elevation_m",
        }[datum]
        return geometry[key] - depth_m
    raise ValueError("Unknown or unsupported depth datum; do not infer from internal temperature")


def load_full_train_prior(path, *, expected_sha256):
    """Bind actual complete external TRAIN candidate; do not certify phase init."""
    require(sha256(path) == expected_sha256, "TRAIN external prior hash changed")
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    require(
        p.get("schema") == "full-TRAIN-external-statistics-candidate/1",
        "Wrong external prior schema",
    )
    require(
        p.get("full_original_population_role_audit_completed") is True
        and p.get("full_original_TRAIN_covered") is True,
        "Complete original-role audit and TRAIN coverage required",
    )
    require(
        p.get("role_entity_counts") == {"train": 169, "validation": 58, "test": 65}
        and p.get("original_TRAIN_entity_count") == 169,
        "Frozen population differs",
    )
    require(
        p.get("target_reader_calls") == 0
        and p.get("temperature_or_internal_depth_values_read") is False
        and p.get("raw_negative_values_clipped") is False
        and p.get("values_filled_or_imputed") is False,
        "Unsafe candidate provenance",
    )
    require(
        p.get("phase_initialization_certified") is False
        and p.get("model_normalization_certified") is False,
        "Candidate must not impersonate model qualification",
    )
    air = p["fields"]["air_temperature_c"]
    require(
        air["nonfinite_count"] == 0 and air["finite_count"] > 0,
        "Air reference requires all finite raw TRAIN hours",
    )
    return {
        "candidate_sha256": expected_sha256,
        "air_mean_C": air["mean"],
        "air_std_C": air["std_population_ddof0"],
        "native_hours": air["finite_count"],
        "unique_TRAIN_forcing_objects": p["unique_TRAIN_object_count"],
        "full_TRAIN_entities": 169,
        "dependency_hashes": p["dependency_hashes"],
        "phase_initialization_certified": False,
        "model_normalization_certified": False,
        "usage": "external_conditioning_and_scale_candidate_only_not_water_or_phase_initial_state",
    }
