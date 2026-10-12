"""S0 differentiable snow/ice building blocks; NOT an integrated lake model.

Masses are per covered-column area. Ice AREA coverage is an independent input,
never inferred by thresholding vertical solid fraction. Radiation outputs below
are per total lake area. Constants define a freshwater component test model.
Nucleation, ice-area dynamics, sublimation, snow aging and the water PDE remain
integration work. A component test cannot establish lake winter accuracy.
"""
from __future__ import annotations
import torch

LATENT_J_KG = 334000.0
CP_ICE_J_KG_K = 2100.0
CP_WATER_J_KG_K = 4186.0


def _finite(*values):
    if any(not bool(torch.isfinite(v).all()) for v in values):
        raise ValueError("Nonfinite physical input; missing forcing must not become zero")


def freshwater_phase(specific_enthalpy_j_kg):
    """Reference liquid water at 0C; latent plateau has temperature exactly 0C."""
    e = specific_enthalpy_j_kg
    _finite(e)
    solid = (-e / LATENT_J_KG).clamp(0, 1)
    temperature = torch.where(e > 0, e / CP_WATER_J_KG_K,
                             torch.where(e < -LATENT_J_KG, (e + LATENT_J_KG) / CP_ICE_J_KG_K, torch.zeros_like(e)))
    return {"phase_temperature_c": temperature, "solid_mass_fraction": solid,
            "liquid_temperature_valid": solid < 1}


def frozen_layer_step(mass_kg_m2, enthalpy_j_m2, supplied_energy_j_m2):
    """Heat a frozen snow OR ice layer and export melt at 0C plus surplus heat.

The initial retained layer is entirely solid (possibly cold). Melted mass exits
this layer into the next reservoir at zero reference enthalpy. With no layer,
all supplied heat/cooling passes through. This function does NOT nucleate ice.
No snow thickness is inferred without a separately supplied density.
"""
    m, e, q = mass_kg_m2, enthalpy_j_m2, supplied_energy_j_m2
    _finite(m, e, q)
    # Reconstructing mass from enthalpy introduces a few ulps in FP32. This is
    # an input-consistency tolerance only; neither mass nor energy is adjusted.
    tolerance = 8 * torch.finfo(e.dtype).eps * (LATENT_J_KG * m).abs().clamp_min(1)
    if bool((m < 0).any()) or bool((e > -LATENT_J_KG * m + tolerance).any()):
        raise ValueError("Initial retained mass must be frozen, with valid enthalpy")
    if bool(((m == 0) & (e != 0)).any()):
        raise ValueError("Zero mass cannot retain enthalpy")
    total = e + q
    retained_mass = torch.minimum((-total / LATENT_J_KG).clamp_min(0), m)
    retained_energy = torch.where(retained_mass > 0, total, torch.zeros_like(total))
    exported_heat = total - retained_energy
    temperature = torch.where(retained_mass > 0,
                              (retained_energy / retained_mass.clamp_min(1e-20) + LATENT_J_KG) / CP_ICE_J_KG_K,
                              torch.zeros_like(total))
    return {"solid_mass_kg_m2": retained_mass, "solid_enthalpy_j_m2": retained_energy,
            "solid_temperature_c": temperature, "meltwater_kg_m2": m - retained_mass,
            "exported_heat_j_m2": exported_heat,
            "energy_residual_j_m2": retained_energy + exported_heat - e - q,
            "mass_residual_kg_m2": retained_mass + (m - retained_mass) - m}


def conductive_flux_w_m2(upper_temperature_c, lower_temperature_c,
                         snow_thickness_m, ice_thickness_m, snow_conductivity, ice_conductivity):
    """Positive upward heat flow, per covered-column area, through two layers."""
    _finite(upper_temperature_c, lower_temperature_c, snow_thickness_m, ice_thickness_m, snow_conductivity, ice_conductivity)
    if bool((snow_thickness_m < 0).any()) or bool((ice_thickness_m < 0).any()):
        raise ValueError("Negative layer thickness")
    if bool((snow_conductivity <= 0).any()) or bool((ice_conductivity <= 0).any()):
        raise ValueError("Nonpositive conductivity")
    resistance = snow_thickness_m / snow_conductivity + ice_thickness_m / ice_conductivity
    if bool((resistance <= 0).any()):
        raise ValueError("No covered layer: use open-water boundary instead")
    return (lower_temperature_c - upper_temperature_c) / resistance


def shortwave_partition(q_w_m2, ice_area_fraction, snow_area_fraction_on_ice,
                        snow_thickness_m, ice_thickness_m, water_albedo, ice_albedo,
                        snow_albedo, snow_extinction_m1, ice_extinction_m1):
    """Conservative open-water/snow/bare-ice shortwave partition.

Snow thickness is for the snowy part of the ice-covered area. Snow-free ice
uses ice albedo. All absorption/reflection outputs use total lake area units.
"""
    values=(q_w_m2,ice_area_fraction,snow_area_fraction_on_ice,snow_thickness_m,ice_thickness_m,
            water_albedo,ice_albedo,snow_albedo,snow_extinction_m1,ice_extinction_m1)
    _finite(*values)
    for f in (ice_area_fraction,snow_area_fraction_on_ice,water_albedo,ice_albedo,snow_albedo):
        if bool(((f < 0) | (f > 1)).any()):raise ValueError("Fraction outside [0,1]")
    for positive in (q_w_m2,snow_thickness_m,ice_thickness_m,snow_extinction_m1,ice_extinction_m1):
        if bool((positive < 0).any()):raise ValueError("Negative optical input")
    q,f,s=q_w_m2,ice_area_fraction,snow_area_fraction_on_ice
    ts=torch.exp(-snow_extinction_m1*snow_thickness_m)
    ti=torch.exp(-ice_extinction_m1*ice_thickness_m)
    snow_enter=q*f*s*(1-snow_albedo)
    bare_enter=q*f*(1-s)*(1-ice_albedo)
    open_water=q*(1-f)*(1-water_albedo)
    snow_absorbed=snow_enter*(1-ts)
    ice_incident=bare_enter+snow_enter*ts
    ice_absorbed=ice_incident*(1-ti)
    under_ice=ice_incident*ti
    reflected=q*((1-f)*water_albedo+f*((1-s)*ice_albedo+s*snow_albedo))
    return {"reflected_w_m2":reflected,"snow_absorbed_w_m2":snow_absorbed,
            "ice_absorbed_w_m2":ice_absorbed,"under_ice_water_w_m2":under_ice,
            "open_water_w_m2":open_water,
            "budget_residual_w_m2":q-reflected-snow_absorbed-ice_absorbed-under_ice-open_water}


def radiative_surface_temperature_c(water_skin_c, frozen_surface_c, ice_area_fraction):
    """Equal-emissivity broadband radiative mixture; not a sensor-band inversion.

This predicts surface radiance-equivalent temperature, NOT subsurface water T.
The water argument must already be a water-skin observation-model output.
"""
    _finite(water_skin_c,frozen_surface_c,ice_area_fraction)
    if bool(((ice_area_fraction<0)|(ice_area_fraction>1)).any()):raise ValueError("Invalid area coverage")
    water_k=water_skin_c+273.15;ice_k=frozen_surface_c+273.15
    if bool(((water_k<=0)|(ice_k<=0)).any()):raise ValueError("Nonphysical absolute temperature")
    return ((1-ice_area_fraction)*water_k.pow(4)+ice_area_fraction*ice_k.pow(4)).pow(.25)-273.15
