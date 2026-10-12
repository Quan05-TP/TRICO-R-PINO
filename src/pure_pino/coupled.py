"""Direct conditional freshwater WIS/bed RHS with explicit compensation.

No state integration or budget projection. Predicted fields are continuous
learned functions. Fixed native liquid storage is a declared OPEN model;
compensation is computed from independently physical mass rates, never from an
energy error. It is not observed runoff or moving-water/draft reconstruction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from .phase import (
    LIQUID_COORDINATE_CONTRACT,
    LIQUID_NATIVE_LATENT_CEILING_C,
    SOLID_COORDINATE_CONTRACT,
    SOLID_NATIVE_COLD_CEILING_K,
    DirectPhaseComponents,
    finite,
    interval_balance,
    require,
    sha256,
)

LAYOUT = "TACO_DIRECT_REFERENCE_WATER_ICE_SNOW_BED_P0_V2"
GEOMETRY_POLICY = "fixed_reference_liquid_volume_compensated_v2"


@dataclass
class CoupledExternal:
    edges_m: torch.Tensor  # [B,N+1], actual original external geometry
    face_area_m2: torch.Tensor  # [B,N+1]
    native_volume_m3: torch.Tensor  # [B,N], no trap(A) replacement
    core9: torch.Tensor  # [B,Q,9], already audited native reconstruction
    native_supply_interval_s: torch.Tensor  # [B,Q], original external native-hour support
    kd_visible_m_inv: torch.Tensor  # [B,Q] or explicit broadcast support
    kd_nir_m_inv: torch.Tensor
    visible_fraction: torch.Tensor
    thermal_scale_K: float  # frozen external TRAIN scale, not learned variance
    source_bindings: dict
    depth_reference_policy: str = "fixed_reference_depth_P0"
    chemistry_status: str = "unknown_reference_water"
    forcing_policy: str = "raw_signed_preserved_native_domain_per_actual_window_no_clipping"


@dataclass
class CoupledDecoded:
    state: object
    inventory_energy_J: torch.Tensor  # [B,Q,2N+2]: waterN,ice,snow,bedN
    inventory_mass_kg: torch.Tensor
    inventory_salt_kg: torch.Tensor
    observables: dict
    observable_valid_masks: dict
    energy_scale_J: torch.Tensor  # [B,2N+2], frozen native storage/material scales
    mass_scale_kg: torch.Tensor
    qualification: dict


@dataclass
class CoupledPowers:
    energy_rhs_W: torch.Tensor
    mass_rhs_kg_s: torch.Tensor
    salt_rhs_kg_s: torch.Tensor
    internal_energy_W: torch.Tensor
    internal_mass_kg_s: torch.Tensor
    external_energy_W: torch.Tensor
    external_mass_kg_s: torch.Tensor
    external_components_W: dict
    diagnostics: dict


class DirectPhaseCoupledAdapter:
    """New interface; supported conditional domains obtain actual fresh evidence."""

    name = "taco_direct_conditional_reference_wis_bed_native_liquid_solid_domain_v6"
    synthetic_only = False
    state_layout_id = LAYOUT
    latent_coordinate_contract_id = LIQUID_COORDINATE_CONTRACT
    solid_coordinate_contract_id = SOLID_COORDINATE_CONTRACT
    geometry_policy = GEOMETRY_POLICY
    model_admitted = False
    full_phase_adapter_qualified = False

    def __init__(
        self,
        bindings,
        phase: DirectPhaseComponents,
        surface_boundary,
        *,
        water_albedo=0.07,
        ice_albedo=0.4,
        snow_albedo=0.8,
        ice_extinction_m_inv=1.5,
        snow_extinction_m_inv=20.0,
        global_mass_reference_fraction=0.001,
    ):
        self.bindings, self.phase, self.surface = bindings, phase, surface_boundary
        self.water_albedo, self.ice_albedo, self.snow_albedo = water_albedo, ice_albedo, snow_albedo
        self.ice_extinction, self.snow_extinction = ice_extinction_m_inv, snow_extinction_m_inv
        self.global_mass_fraction = global_mass_reference_fraction
        if (bindings.cp_water, bindings.cp_ice, bindings.latent) != (
            phase.cp_water,
            phase.cp_ice,
            phase.latent,
        ):
            raise ValueError("Mixed phase constant/datum versions")
        require(0 < global_mass_reference_fraction < 1, "Frozen phase reference fraction required")
        self.dependency_code_hashes = {
            **bindings.dependency_code_hashes,
            **surface_boundary.dependency_code_hashes,
            str(__file__): sha256(__file__),
            str(Path(__file__).with_name("phase.py")): sha256(Path(__file__).with_name("phase.py")),
            str(Path(__file__).with_name("model.py")): sha256(Path(__file__).with_name("model.py")),
        }

    def validate(self, external: CoupledExternal, queries):
        x = external
        finite(
            x.edges_m,
            x.face_area_m2,
            x.native_volume_m3,
            x.core9,
            x.native_supply_interval_s,
            x.kd_visible_m_inv,
            x.kd_nir_m_inv,
            x.visible_fraction,
        )
        z, a, v, info = self.bindings.validate_geometry(
            x.edges_m, x.face_area_m2, x.native_volume_m3
        )
        require(
            z.ndim == 2 and x.core9.shape == (z.shape[0], queries, 9), "B/Q/native support mismatch"
        )
        require(
            x.native_supply_interval_s.shape == x.core9.shape[:-1]
            and (x.native_supply_interval_s > 0).all(),
            "Actual external native support intervals required, not query step length",
        )
        require(
            math.isfinite(x.thermal_scale_K) and x.thermal_scale_K > 0,
            "Frozen full-TRAIN external thermal scale required",
        )
        require(
            x.chemistry_status in ("unknown_reference_water", "externally_verified_freshwater"),
            "Known nonfreshwater needs another salt/phase law; retain UNSUPPORTED",
        )
        require(
            x.depth_reference_policy == "fixed_reference_depth_P0",
            "This conditional model does not certify moving water/draft datum",
        )
        require(
            x.forcing_policy == "raw_signed_preserved_native_domain_per_actual_window_no_clipping"
            and x.source_bindings,
            "Explicit actual forcing/geometry source bindings required",
        )
        # Root's frozen driver policy is stricter than native SF reconciliation:
        # raw negative SF remains a failed node/window, never quietly changed.
        require((x.core9[..., 8] >= 0).all(), "RAW_NEGATIVE_SF_WINDOW_DOMAIN_FAILURE")
        for row in x.core9.reshape(-1, 9):
            self.bindings.weather.physical_weather(
                row, partition_policy=self.bindings.weather.PARTITION_POLICY
            )
        return z, a, v, info

    def decode(self, raw, external: CoupledExternal):
        coords = {name: value.double() for name, value in raw.physical_coordinates().items()}
        b, q, n = coords["liquid_enthalpy_coordinate"].shape
        z, a, v, info = self.validate(external, q)
        require(v.shape == (b, n), "Native P0/raw field spatial support differs")
        bed = self.bindings.bed_parameters(a)
        cb = bed["heat_capacity_j_k"][:, None].expand(-1, q, -1)
        # Fixed liquid mass is an explicit model constraint. There is no unused
        # learned mass head and no adjustment of a predicted physical inventory.
        coords["liquid_mass_coordinate"] = torch.ones_like(coords["liquid_enthalpy_coordinate"])
        coords["bed_temperature_C"] = torch.where(
            cb > 0, coords["bed_temperature_C"], torch.zeros_like(coords["bed_temperature_C"])
        )
        m = 1000 * v
        mg = m.sum(-1)[:, None] * self.global_mass_fraction
        p = self.phase.raw_to_physical(
            coords,
            liquid_mass_scale=m[:, None],
            global_mass_scale=mg,
            liquid_h_scale=self.phase.cp_water * external.thermal_scale_K,
            cold_temperature_scale=external.thermal_scale_K,
            bed_capacity_J_K=cb,
            liquid_temperature_upper_C=40.0,
        )
        d = self.phase.decode(p, chemistry_status=external.chemistry_status)
        # Thermodynamic calorimetry remains broader; this particular density
        # closure applies the documented native EOS reference-water domain.
        require(
            ((d.liquid_temperature_C >= 0) & (d.liquid_temperature_C <= 40)).all(),
            "REFERENCE_WATER_EOS_DOCUMENTED_DOMAIN_FAILURE_NO_TEMPERATURE_CLIP",
        )
        energy = torch.cat(
            (
                p.liquid_energy_J,
                p.ice_energy_J[..., None],
                p.snow_energy_J[..., None],
                p.bed_energy_J,
            ),
            -1,
        )
        zero_bed = torch.zeros_like(p.bed_energy_J)
        mass = torch.cat(
            (p.liquid_mass_kg, p.ice_mass_kg[..., None], p.snow_mass_kg[..., None], zero_bed), -1
        )
        salt = torch.zeros_like(
            mass
        )  # explicit conditional reference water; not measured chemistry
        global_energy_scale = m.sum(-1) * self.global_mass_fraction * self.phase.latent
        energy_scale = torch.cat(
            (
                m * self.phase.cp_water * external.thermal_scale_K,
                global_energy_scale[:, None].expand(-1, 2),
                bed["heat_capacity_j_k"] * external.thermal_scale_K,
            ),
            -1,
        )
        mass_scale = torch.cat(
            (m, (m.sum(-1) * self.global_mass_fraction)[:, None].expand(-1, 2), torch.ones_like(m)),
            -1,
        )
        observables = {
            "water_temperature_C": d.liquid_temperature_C,
            "bed_temperature_C": d.bed_temperature_C,
            "ice_area_fraction": p.ice_area_fraction[..., None],
        }
        masks = {
            "water_temperature_C": d.liquid_valid_mask,
            "bed_temperature_C": cb > 0,
            "ice_area_fraction": torch.ones_like(p.ice_mass_kg[..., None], dtype=torch.bool),
        }
        return CoupledDecoded(
            d,
            energy,
            mass,
            salt,
            observables,
            masks,
            energy_scale,
            mass_scale,
            {
                **info,
                "state_layout_id": LAYOUT,
                "geometry_model": GEOMETRY_POLICY,
                "latent_coordinate_contract_id": LIQUID_COORDINATE_CONTRACT,
                "solid_coordinate_contract_id": SOLID_COORDINATE_CONTRACT,
                "solid_latent_mapping": "T=-B*u/(B+u);B=(273.15-100)*(1-4096*float64_eps);u=thermal_scale_K*r_squared;h=-Lf+CP_ICE*T",
                "predefined_solid_native_cold_ceiling_K": SOLID_NATIVE_COLD_CEILING_K,
                "solid_native_atmosphere_minimum_absolute_K": 100.0,
                "physical_inventory_and_observation_basis_unchanged_from_V2": True,
                "liquid_latent_mapping": "T=L*u/(L+u);L=40*(1-8*float64_eps);u=thermal_scale_K*r_squared;h=CP_WATER*T",
                "predefined_liquid_latent_ceiling_C": LIQUID_NATIVE_LATENT_CEILING_C,
                "latent_ceiling_roundoff_margin_float64_eps": 8,
                "chemistry_status": external.chemistry_status,
                "salinity_measured": external.chemistry_status == "externally_verified_freshwater",
                "modeled_compensation_not_observed_runoff": True,
                "ice_draft_freeboard_resolved": False,
                "native_density_domain_C": [0, 40],
                "predicted_temperature_clipped": False,
                "full_phase_adapter_qualified": False,
                "model_admitted": False,
            },
        )

    def powers(self, raw, decoded: CoupledDecoded, external: CoupledExternal):
        d, p = decoded.state, decoded.state.physical
        b, q, n = p.liquid_mass_kg.shape
        z, a, v, _ = self.validate(external, q)
        center = (z[:, 1:] + z[:, :-1]) * 0.5
        shape, total = (b, q), 2 * n + 2
        ice_index, snow_index, bed_start = n, n + 1, n + 2
        zero = torch.zeros(shape, dtype=torch.float64, device=z.device)

        def vec():
            return torch.zeros((*shape, total), dtype=torch.float64, device=z.device)

        def single(index, value):
            result = vec()
            result[..., index] = value
            return result

        def block(start, value):
            result = vec()
            result[..., start : start + value.shape[-1]] = value
            return result

        internal_e, internal_m, external_e, external_m = vec(), vec(), vec(), vec()
        mass_addition_absolute_scale = vec()
        components = {}
        native = self.bindings.weather
        rows = [
            native.physical_weather(row, partition_policy=native.PARTITION_POLICY)
            for row in external.core9.reshape(-1, 9)
        ]
        weather = {key: torch.stack([row[key] for row in rows]).reshape(shape) for key in rows[0]}
        area = a[:, None, 0].expand(shape)
        f, s = p.ice_area_fraction, p.snow_fraction_on_ice
        ai, ass, ab, ao = area * f, area * f * s, area * f * (1 - s), area * (1 - f)
        safe_ai = torch.where(ai > 0, ai, torch.ones_like(ai))
        safe_as = torch.where(ass > 0, ass, torch.ones_like(ass))
        hi = p.ice_mass_kg / (917 * safe_ai)
        hs = p.snow_mass_kg / (p.snow_density_kg_m3 * safe_as)
        optics = self.bindings.partial_shortwave(
            weather["sw"],
            z[:, None],
            a[:, None],
            v[:, None],
            ice_fraction=f,
            snow_fraction_on_ice=s,
            ice_thickness_m=hi,
            snow_thickness_m=hs,
            water_albedo=self.water_albedo,
            ice_albedo=self.ice_albedo,
            snow_albedo=self.snow_albedo,
            snow_extinction_m_inv=self.snow_extinction,
            ice_extinction_m_inv=self.ice_extinction,
            kd_visible_m_inv=external.kd_visible_m_inv,
            kd_nir_m_inv=external.kd_nir_m_inv,
            visible_fraction=external.visible_fraction,
        )
        for name, value in (
            ("solar_water", block(0, optics["water_w"])),
            ("solar_bed", block(bed_start, optics["combined_bed_w"])),
            ("solar_ice", single(ice_index, optics["ice_w"])),
            ("solar_snow", single(snow_index, optics["snow_w"])),
        ):
            components[name] = value
            external_e = external_e + value
        # Source formula: bounded learned diffusivity changes conductance only,
        # never adds a free W/m2 heat correction.
        tw = d.liquid_temperature_C
        density = self.bindings.daily.eos80_density_atm(tw, torch.zeros_like(tw))
        dz = center[:, 1:] - center[:, :-1]
        n2 = 9.81 / 1000 * density.diff(dim=-1) / dz[:, None]
        mid = (center[:, 1:] + center[:, :-1]) * 0.5
        wind_mix = (
            1e-5
            * (weather["wind"][..., None] / 3).square()
            * torch.exp(-mid[:, None] / (1 + 0.1 * z[:, None, -1:]))
            / (1 + 1000 * n2.clamp_min(0))
        )
        convect = 1e-3 * torch.tanh(1000 * (-n2).clamp_min(0))
        control = 0.5 * (
            raw.process_control[..., :-1].double() + raw.process_control[..., 1:].double()
        )
        kz = (1.4e-7 + (1 - f[..., None]) * wind_mix + convect) * torch.exp(2 * torch.tanh(control))
        conductance = 1000 * self.phase.cp_water * a[:, None, 1:-1] * kz / dz[:, None]
        edge = conductance * (tw[..., :-1] - tw[..., 1:])
        water_exchange = torch.zeros_like(tw)
        water_exchange[..., :-1] = water_exchange[..., :-1] - edge
        water_exchange[..., 1:] = water_exchange[..., 1:] + edge
        internal_e = internal_e + block(0, water_exchange)
        bed = self.bindings.bed_parameters(a)
        bed_to_water = bed["conductance_w_k"][:, None] * (d.bed_temperature_C - tw)
        internal_e = internal_e + block(0, bed_to_water) - block(bed_start, bed_to_water)
        coefficient = 0.0013 * torch.exp(0.5 * torch.tanh(raw.process_control[..., 0].double()))
        # Caps use immutable external native support, not the decoded query dt.
        tau = external.native_supply_interval_s

        def cap(mass, support):
            return mass / torch.where(support > 0, support * tau, torch.ones_like(support))

        open_cap = cap(p.liquid_mass_kg[..., 0], ao)
        skin_config = replace(self.bindings.daily.SkinBulkConfig(), neutral_exchange=coefficient)
        skin = self.bindings.daily.skin_bulk_exchange(
            tw[..., 0], external.core9, config=skin_config
        )
        finite(
            *(
                skin[key]
                for key in (
                    "skin_c",
                    "flux_components_w_m2",
                    "into_water_non_solar_w_m2",
                    "closure_residual_w_m2",
                    "skin_conductance_w_m2_k",
                    "stability_factor",
                )
            )
        )
        raw_skin = skin["skin_c"]
        freezing_surface = raw_skin < 0
        # Warm native Robin branch uses its ORIGINAL paired latent component.
        # If its potential supply exceeds immutable support, fail this domain;
        # never cap just mass while keeping unpaid original latent energy.
        potential_vapor = -skin["flux_components_w_m2"][..., 2] / skin_config.latent_heat_j_kg
        require(
            (freezing_surface | (potential_vapor <= open_cap)).all(),
            "NATIVE_WARM_SKIN_SUPPLY_DOMAIN_FAILURE_NO_UNPAID_LATENT_CAP",
        )
        # A subfreezing LIQUID skin is replaced by a PHYSICAL freezing interface
        # active set at Tf, not by clipping a learned bulk temperature. Native
        # bulk stability/G is frozen during this boundary solve as in source.
        cold_weather = {
            **weather,
            "wind": torch.sqrt(
                external.core9[..., 2].square()
                + external.core9[..., 3].square()
                + skin_config.gustiness_m_s**2
            )
            * skin["stability_factor"],
        }
        q_at_tf, _, vapor_at_tf = native.atmosphere(
            zero, cold_weather, coefficient, frozen=False, max_vapor=open_cap
        )
        qo = torch.where(freezing_surface, q_at_tf, skin["into_water_non_solar_w_m2"])
        vo = torch.where(freezing_surface, vapor_at_tf, potential_vapor)
        water_skin = torch.where(freezing_surface, zero, raw_skin)
        freeze_surplus = skin["skin_conductance_w_m2_k"] * (-tw[..., 0]) - q_at_tf
        require(
            (~freezing_surface | (freeze_surplus >= -1e-7)).all(),
            "Freezing skin complementarity failed",
        )
        skin_nucleation = torch.where(
            freezing_surface, freeze_surplus.clamp_min(0) * ao / self.phase.latent, zero
        )
        finite(qo, vo, open_cap, water_skin, skin_nucleation)
        components["atmosphere_open_native"] = single(0, qo * ao)
        external_e = external_e + components["atmosphere_open_native"]
        open_mass = -vo * ao
        open_carrier = open_mass * self.phase.cp_water * tw[..., 0]
        components["vapor_open_condensed_carrier"] = single(0, open_carrier)
        external_e = external_e + components["vapor_open_condensed_carrier"]
        external_m = external_m + single(0, open_mass)
        mass_addition_absolute_scale = mass_addition_absolute_scale + single(0, open_mass.abs())
        surface_diagnostics = {}
        surface_melt = {}
        skins = {}
        ice_sublimation_loss = zero
        for name, index, mass, support, resistance, temperature in (
            ("ice", ice_index, p.ice_mass_kg, ab, hi / 4.4, d.ice_temperature_C),
            ("snow", snow_index, p.snow_mass_kg, ass, hs / 0.6, d.snow_temperature_C),
        ):
            active = (mass > 0) & (support > 0)
            ids = active.reshape(-1).nonzero(as_tuple=False).flatten()

            def scatter(value):
                return zero.reshape(-1).index_copy(0, ids, value).reshape(shape)

            if len(ids):
                supply = cap(mass, support).reshape(-1)[ids]
                finite(supply)
                require((supply >= 0).all(), "Nonnegative finite vapor supply required")
                result = self.surface.evaluate(
                    temperature.reshape(-1)[ids],
                    resistance.reshape(-1)[ids],
                    external.core9.reshape(-1, 9)[ids],
                    coefficient.reshape(-1)[ids],
                    max_vapor_kg_m2_s=supply,
                )
                atmospheric = scatter(result["downward_atmosphere_W_m2"])
                vapor = scatter(result["outward_native_vapor_kg_m2_s"])
                melt_rate = scatter(result["melt_mass_rate_kg_m2_s"]) * support
                skins[name] = scatter(result["surface_temperature_C"])
                surface_diagnostics[name] = {
                    "energy_residual_W_m2": scatter(result["surface_energy_residual_W_m2"]),
                    "complementarity_K_W_m2": scatter(
                        result["surface_complementarity_residual_K_W_m2"]
                    ),
                }
            else:
                atmospheric, vapor, melt_rate, skins[name] = zero, zero, zero, zero
            components["atmosphere_" + name + "_native"] = single(index, atmospheric * support)
            external_e = external_e + components["atmosphere_" + name + "_native"]
            dm = -vapor * support
            if name == "ice":
                ice_sublimation_loss = (-dm).clamp_min(0)
            hcond = -self.phase.latent + self.phase.cp_ice * temperature
            components["vapor_" + name + "_condensed_carrier"] = single(index, dm * hcond)
            external_e = external_e + components["vapor_" + name + "_condensed_carrier"]
            external_m = external_m + single(index, dm)
            mass_addition_absolute_scale = mass_addition_absolute_scale + single(index, dm.abs())
            surface_melt[name] = melt_rate
            # At-surface melt energy Q is already in native atmospheric power.
            # Pure melt transfer has zero net E; latent removal/heat payment at
            # the same source reservoir cancel, leaving Q counted once.
            internal_m = internal_m - single(index, melt_rate) + single(0, melt_rate)
            mass_addition_absolute_scale = (
                mass_addition_absolute_scale
                + single(index, melt_rate.abs())
                + single(0, melt_rate.abs())
            )
        # Actual internal ice-water phase interface at Tf, separate heat paths.
        ice_active = (p.ice_mass_kg > 0) & (ai > 0)
        ri = torch.where(ice_active, hi / 4.4, torch.ones_like(hi))
        qi = torch.where(ice_active, ai * (-d.ice_temperature_C) / ri, zero)
        qw = torch.where(ice_active, ai * tw[..., 0] / (center[:, None, 0] / 0.6), zero)
        basal_freeze_signed = (qi - qw) / self.phase.latent
        # Carrier h_liquid(Tf)=0/h_ice(Tf)=-Lf; together with qi/qw they close.
        internal_m = (
            internal_m - single(0, basal_freeze_signed) + single(ice_index, basal_freeze_signed)
        )
        mass_addition_absolute_scale = (
            mass_addition_absolute_scale
            + single(0, basal_freeze_signed.abs())
            + single(ice_index, basal_freeze_signed.abs())
        )
        basal_e = single(0, -qw) + single(ice_index, qi - self.phase.latent * basal_freeze_signed)
        internal_e = internal_e + basal_e
        # Snow/ice conduction is paired. Snow unsupported after old-ice loss
        # carries its full bulk enthalpy into water, as the native source policy.
        snow_active = (p.snow_mass_kg > 0) & ice_active & (ass > 0)
        rs = torch.where(snow_active, hs / 0.6 + hi / 4.4, torch.ones_like(hi))
        snow_to_ice = torch.where(
            snow_active, ass * (d.snow_temperature_C - d.ice_temperature_C) / rs, zero
        )
        internal_e = internal_e + single(ice_index, snow_to_ice) - single(snow_index, snow_to_ice)
        ice_loss = surface_melt["ice"] + (-basal_freeze_signed).clamp_min(0) + ice_sublimation_loss
        ratio = p.snow_mass_kg / torch.where(
            p.ice_mass_kg > 0, p.ice_mass_kg, torch.ones_like(p.ice_mass_kg)
        )
        release = torch.where(p.ice_mass_kg > 0, ratio * ice_loss, zero)
        release_e = release * (-self.phase.latent + self.phase.cp_ice * d.snow_temperature_C)
        internal_m = internal_m - single(snow_index, release) + single(0, release)
        mass_addition_absolute_scale = (
            mass_addition_absolute_scale
            + single(snow_index, release.abs())
            + single(0, release.abs())
        )
        internal_e = internal_e - single(snow_index, release_e) + single(0, release_e)
        # Native air-temperature precip carrier policy, explicitly modelled.
        rain = 1000 * weather["rain_m_hour"] * area / 3600
        snow = 1000 * weather["snow_m_hour"] * area / 3600
        rain_h = self.phase.cp_water * weather["air"].clamp_min(0)
        snow_h = -self.phase.latent + self.phase.cp_ice * weather["air"].clamp_max(0)
        precip_m = single(0, rain + (1 - f) * snow) + single(snow_index, f * snow)
        precip_e = single(0, rain * rain_h + (1 - f) * snow * snow_h) + single(
            snow_index, f * snow * snow_h
        )
        external_m = external_m + precip_m
        mass_addition_absolute_scale = mass_addition_absolute_scale + precip_m.abs()
        external_e = external_e + precip_e
        components["precipitation_native_model_carrier"] = precip_e
        # New ice can form at a freezing atmospheric skin while liquid bulk is
        # warm, or at an exact bulk phase boundary. Neither clips bulk T.
        top_nonphase_heat = external_e[..., 0] + internal_e[..., 0]
        boundary_nucleation = torch.where(
            tw[..., 0] == 0, (-top_nonphase_heat).clamp_min(0) / self.phase.latent, zero
        )
        nucleation = torch.maximum(skin_nucleation, boundary_nucleation)
        internal_m = internal_m - single(0, nucleation) + single(ice_index, nucleation)
        mass_addition_absolute_scale = (
            mass_addition_absolute_scale
            + single(0, nucleation.abs())
            + single(ice_index, nucleation.abs())
        )
        internal_e = (
            internal_e
            + single(0, self.phase.latent * nucleation)
            - single(ice_index, self.phase.latent * nucleation)
        )
        # Fixed-reference compensation from MASS physics, with actual top-cell
        # model enthalpy as native inlet/outlet carrier. No energy residual used.
        liquid_net = (external_m + internal_m)[..., :n].sum(-1)
        incoming, outgoing = (-liquid_net).clamp_min(0), liquid_net.clamp_min(0)
        top_h = self.phase.cp_water * tw[..., 0]
        in_e, out_e = incoming * top_h, outgoing * top_h
        external_m = external_m + single(0, incoming - outgoing)
        mass_addition_absolute_scale = mass_addition_absolute_scale + single(0, incoming + outgoing)
        components["modeled_compensation_inflow_carrier"] = single(0, in_e)
        components["modeled_compensation_outflow_carrier"] = single(0, -out_e)
        external_e = (
            external_e
            + components["modeled_compensation_inflow_carrier"]
            + components["modeled_compensation_outflow_carrier"]
        )
        rhs_e, rhs_m = internal_e + external_e, internal_m + external_m
        finite(rhs_e, rhs_m, internal_e, internal_m, external_e, external_m)
        mass_roundoff_bound = 64 * torch.finfo(rhs_m.dtype).eps * (1 + mass_addition_absolute_scale)
        require(
            (rhs_m[..., :n].abs() <= mass_roundoff_bound[..., :n]).all(),
            "Fixed native liquid mass compensation does not close",
        )
        decoded.observables.update(
            {
                "ice_surface_temperature_C": skins["ice"][..., None],
                "snow_surface_temperature_C": skins["snow"][..., None],
                "water_skin_temperature_C": water_skin[..., None],
            }
        )
        decoded.observable_valid_masks.update(
            {
                "ice_surface_temperature_C": (ab > 0)[..., None],
                "snow_surface_temperature_C": (ass > 0)[..., None],
                "water_skin_temperature_C": ((ao > 0) & ~freezing_surface)[..., None],
            }
        )
        diagnostics = {
            "shortwave_budget_residual_W": optics["balance_residual_w"],
            "surface_complementarity": surface_diagnostics,
            "internal_energy_sum_W": internal_e.sum(-1),
            "internal_mass_sum_kg_s": internal_m.sum(-1),
            "actual_fixed_liquid_mass_roundoff_residual_kg_s": rhs_m[..., :n],
            "mass_addition_absolute_scale_kg_s": mass_addition_absolute_scale,
            "mass_floating_addition_64ulp_bound_kg_s": mass_roundoff_bound,
            "mass_residual_zeroed_or_corrected": False,
            "water_graph_edge_W": edge,
            "bed_to_water_W": bed_to_water,
            "basal_freezing_signed_kg_s": basal_freeze_signed,
            "open_nucleation_kg_s": nucleation,
            "surface_ice_melt_kg_s": surface_melt["ice"],
            "surface_snow_melt_kg_s": surface_melt["snow"],
            "snow_release_with_old_ice_loss_kg_s": release,
            "old_ice_sublimation_loss_kg_s": ice_sublimation_loss,
            "modeled_compensation_inflow_kg_s": incoming,
            "modeled_compensation_outflow_kg_s": outgoing,
            "modeled_compensation_inflow_W": in_e,
            "modeled_compensation_outflow_W": out_e,
            "native_snowfall_adjustment_m_hour": weather["sf_adjustment_m_hour"],
            "native_water_skin_closure_residual_W_m2": skin["closure_residual_w_m2"],
            "water_surface_freezing_active": freezing_surface,
            "water_surface_freezing_rate_kg_s": skin_nucleation,
            "water_skin_bulk_bridge_is_actual_native": True,
            "vapor_supply_support_seconds": tau,
            "supply_is_query_step_or_temperature_integrator": False,
            "precipitation_temperature_is_observed": False,
            "chemistry_status": external.chemistry_status,
            "salt_inventory_is_measured": False,
            "geometry_model": GEOMETRY_POLICY,
            "river_flow_is_observed": False,
            "ice_draft_freeboard_resolved": False,
            "full_phase_adapter_qualified": False,
        }
        return CoupledPowers(
            rhs_e,
            rhs_m,
            torch.zeros_like(rhs_m),
            internal_e,
            internal_m,
            external_e,
            external_m,
            components,
            diagnostics,
        )

    def balance(self, decoded: CoupledDecoded, powers: CoupledPowers, times):
        """Actual coupled per-reservoir and full external interval balances."""
        e = interval_balance(times, decoded.inventory_energy_J, powers.energy_rhs_W)
        m = interval_balance(times, decoded.inventory_mass_kg, powers.mass_rhs_kg_s)
        s = interval_balance(times, decoded.inventory_salt_kg, powers.salt_rhs_kg_s)
        whole_e = interval_balance(
            times,
            decoded.inventory_energy_J.sum(-1, keepdim=True),
            powers.external_energy_W.sum(-1, keepdim=True),
        )
        whole_m = interval_balance(
            times,
            decoded.inventory_mass_kg.sum(-1, keepdim=True),
            powers.external_mass_kg_s.sum(-1, keepdim=True),
        )
        scale_e = torch.where(
            decoded.energy_scale_J > 0,
            decoded.energy_scale_J,
            torch.ones_like(decoded.energy_scale_J),
        )
        scaled_e = e["per_reservoir_residual"] / scale_e
        scaled_m = m["per_reservoir_residual"] / decoded.mass_scale_kg
        energy_loss = scaled_e.square().mean()
        mass_loss = scaled_m.square().mean()
        total_scale = scale_e.sum(-1)
        total_energy_loss = (whole_e["total_residual"] / total_scale).square().mean()
        loss = energy_loss + mass_loss + total_energy_loss
        finite(loss)
        return {
            "loss": loss,
            "per_reservoir_energy_residual_J": e["per_reservoir_residual"],
            "per_reservoir_mass_residual_kg": m["per_reservoir_residual"],
            "per_reservoir_salt_residual_kg": s["per_reservoir_residual"],
            "total_external_energy_residual_J": whole_e["total_residual"],
            "total_external_mass_residual_kg": whole_m["total_residual"],
            "energy_loss": energy_loss,
            "mass_loss": mass_loss,
            "total_energy_loss": total_energy_loss,
            "salt_loss": torch.zeros_like(loss),
            "salt_is_conditional_reference_not_measured": True,
            "energy_components_disjoint": True,
            "model_admitted": False,
        }

    def balance_piecewise(
        self, decoded: CoupledDecoded, left: CoupledPowers, right: CoupledPowers, times
    ):
        """Each native-hour forcing is held unchanged at BOTH interval ends.

        The state remains a direct continuous prediction. Trapezoid applies
        only to state-dependent constitutive rates inside one forcing support,
        never averages two neighboring accumulated forcing rows.
        """
        finite(times)
        dt = times[1:] - times[:-1]
        require(
            times.ndim == 1 and len(times) >= 2 and (dt > 0).all(),
            "Ordered direct interval endpoints required",
        )
        require(
            decoded.inventory_energy_J.shape[1] == len(times),
            "Direct inventory endpoint count differs",
        )
        require(
            left.energy_rhs_W.shape
            == right.energy_rhs_W.shape
            == decoded.inventory_energy_J[:, :-1].shape,
            "Each native support needs left/right powers under SAME forcing row",
        )

        def residual(inventory, power_left, power_right):
            return (
                inventory[:, 1:]
                - inventory[:, :-1]
                - 0.5 * dt[None, :, None] * (power_left + power_right)
            )

        e = residual(decoded.inventory_energy_J, left.energy_rhs_W, right.energy_rhs_W)
        m = residual(decoded.inventory_mass_kg, left.mass_rhs_kg_s, right.mass_rhs_kg_s)
        s = residual(decoded.inventory_salt_kg, left.salt_rhs_kg_s, right.salt_rhs_kg_s)
        whole_e = residual(
            decoded.inventory_energy_J.sum(-1, keepdim=True),
            left.external_energy_W.sum(-1, keepdim=True),
            right.external_energy_W.sum(-1, keepdim=True),
        ).squeeze(-1)
        whole_m = residual(
            decoded.inventory_mass_kg.sum(-1, keepdim=True),
            left.external_mass_kg_s.sum(-1, keepdim=True),
            right.external_mass_kg_s.sum(-1, keepdim=True),
        ).squeeze(-1)
        scale_e = torch.where(
            decoded.energy_scale_J > 0,
            decoded.energy_scale_J,
            torch.ones_like(decoded.energy_scale_J),
        )
        energy_loss = (e / scale_e[:, None]).square().mean()
        mass_loss = (m / decoded.mass_scale_kg[:, None]).square().mean()
        total_energy_loss = (whole_e / scale_e.sum(-1)[:, None]).square().mean()
        loss = energy_loss + mass_loss + total_energy_loss
        finite(loss, e, m, s, whole_e, whole_m)
        return {
            "loss": loss,
            "energy_loss": energy_loss,
            "mass_loss": mass_loss,
            "total_energy_loss": total_energy_loss,
            "salt_loss": torch.zeros_like(loss),
            "per_reservoir_energy_residual_J": e,
            "per_reservoir_mass_residual_kg": m,
            "per_reservoir_salt_residual_kg": s,
            "total_external_energy_residual_J": whole_e,
            "total_external_mass_residual_kg": whole_m,
            "forcing_integration": "unchanged_native_row_at_both_interval_endpoints_no_second_shift",
            "salt_is_conditional_reference_not_measured": True,
            "energy_components_disjoint": True,
            "model_admitted": False,
        }
