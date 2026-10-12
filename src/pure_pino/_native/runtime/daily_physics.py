"""Differentiable, FP64 liquid-water process utilities for TRICO-R--PINO V2.

Scope: homogeneous *bulk* salinity, a reduced open-water skin boundary, and
two-band radiation over piecewise-linear hypsography. No learned Q/T correction.
This is NOT a full COARE implementation, inland-brine EOS, ice/snow model, or
sediment model. The caller routes ice-bearing columns through its existing
phase path and stores intercepted radiation in an explicit bed heat reservoir.

Primary references (equations/constants, not claims of global validation):
* UNESCO 44 (1983), EOS-80 eqs. 13--14, atmospheric pressure:
  https://www.jodc.go.jp/info/ioc_doc/UNESCO_tech/059832eb.pdf
* CTSM interface energy balance:
  https://escomp.github.io/CTSM/tech_note/Lake/CLM50_Tech_Note_Lake.html
* COARE physical skin/background:
  https://www.coaps.fsu.edu/COARE/flux_algor/flux.html
* MacKay (2019), radiation intercepted by sloping bed:
  https://gmd.copernicus.org/articles/12/3045/2019/

Inputs are on the same device; floating tensors are converted to float64 while
retaining their autograd graph. Parameter-domain checks belong to data loading,
not the hourly GPU loop. No state/flux clipping or hidden temperature repair.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import torch
from torch import Tensor

SCOPE = "liquid-only; homogeneous salinity; reduced skin; explicit bed interception"


def _64(x, ref: Tensor) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float64, device=ref.device)


def eos80_density_atm(temperature_c: Tensor, practical_salinity: Tensor) -> Tensor:
    """EOS-80 rho in kg/m3, p=0 dbar; input T is ITS-90 Celsius.

    Practical salinity is dimensionless PSS-78, NOT arbitrary g/kg or EC.
    Best supported for standard seawater composition over 0--40 C, S=0--42;
    freshwater endpoint is the UNESCO reference-water polynomial. Returned
    values outside that range are extrapolations, not validated brine values.
    Bulk-uniform salinity still changes the *temperature derivative* of rho.
    """
    t = temperature_c.to(torch.float64) * 1.00024  # ITS-90 -> IPTS-68
    s = _64(practical_salinity, t)
    rw = (((((6.536332e-9*t - 1.120083e-6)*t + 1.001685e-4)*t
             - 9.095290e-3)*t + 6.793952e-2)*t + 999.842594)
    a = (((5.3875e-9*t - 8.2467e-7)*t + 7.6438e-5)*t
         - 4.0899e-3)*t + 0.824493
    b = (-1.6546e-6*t + 1.0227e-4)*t - 5.72466e-3
    return rw + a*s + b*torch.pow(s, 1.5) + 4.8314e-4*s.square()


def bulk_salinity_density(temperature_c: Tensor, salinity_gkg: Tensor,
                          salinity_known: Tensor, *,
                          standard_seawater_composition: Tensor | bool = False):
    """Explicit g/kg adapter with source/applicability flags.

    Standard-composition reference salinity SR = (35.16504/35)*SP, hence the
    inverse conversion here. For inland water this is a declared *sea-salt
    equivalent approximation*, not a universal ion-composition correction.
    Missing values may be supplied as zero by the caller, but `salinity_known`
    remains false; output never labels that fallback as measured freshwater.
    """
    t = temperature_c.to(torch.float64)
    s = _64(salinity_gkg, t)
    sp = s * (35.0 / 35.16504)
    known = torch.as_tensor(salinity_known, dtype=torch.bool, device=t.device)
    standard = torch.as_tensor(standard_seawater_composition,
                               dtype=torch.bool, device=t.device)
    return {
        "density_kg_m3": eos80_density_atm(t, sp),
        "practical_salinity_equivalent": sp,
        "salinity_known": known,
        "composition_approximation": ~standard & (s > 0),
        "eos_documented_range": (t >= 0) & (t <= 40) & (sp >= 0) & (sp <= 42),
        "scope": "reference-water/standard-seawater EOS; no salinity stratification",
    }


@dataclass(frozen=True)
class SkinBulkConfig:
    """Shared, reviewable physical parameters, not per-hour neural outputs.

    1.3e-3 exchange/drag, 0.97 emissivity and 4180/1005 heat capacities retain
    common lake-model values. The bulk-Ri stability factors are a reduced
    candidate (stable (1+5Ri)^-2, unstable sqrt(1-16Ri)), NOT full MOST/COARE.
    Gustiness and 2-mm background skin thickness are explicit candidate scales,
    not measured lake-specific data. Wind skin thickness uses the Saunders
    scaling lambda*nu/u*w, with lambda=6; the background conductance acts in
    parallel. Skin solar absorption is omitted here; all SW goes to the separate
    radiation partition. A spectral skin absorption extension must repartition
    SW rather than double-count it.
    """
    neutral_exchange: float = 1.3e-3
    neutral_drag: float = 1.3e-3
    temperature_height_m: float = 2.0
    wind_height_m: float = 10.0
    water_roughness_m: float = 2e-4
    gustiness_m_s: float = 0.2
    stable_ri_coefficient: float = 5.0
    unstable_ri_coefficient: float = 16.0
    emissivity: float = 0.97
    cp_air_j_kg_k: float = 1005.0
    latent_heat_j_kg: float = 2.5e6
    water_thermal_conductivity_w_m_k: float = 0.6
    water_kinematic_viscosity_m2_s: float = 1e-6
    skin_saunders_constant: float = 6.0
    background_skin_thickness_m: float = 0.002
    water_density_kg_m3: float = 1000.0
    newton_iterations: int = 8


def _humidity(t: Tensor, p: Tensor):
    vapor = 610.94 * torch.exp(17.625*t/(t + 243.04))
    dv = vapor * (17.625*243.04)/(t + 243.04).square()
    denominator = p - 0.378*vapor
    return 0.622*vapor/denominator, 0.622*p*dv/denominator.square()


def _skin_core(bulk: Tensor, forcing: Tensor, config: SkinBulkConfig):
    air, dew, u, v, p, _, lw, _, _ = forcing.unbind(-1)
    qa, _ = _humidity(dew, p)
    qb, dqb = _humidity(bulk, p)
    density_air = p / (287.05*(air + 273.15))
    wind10 = torch.sqrt(u.square()+v.square()+config.gustiness_m_s**2)
    ratio = math.log(config.temperature_height_m/config.water_roughness_m) / math.log(
        config.wind_height_m/config.water_roughness_m)
    wind_t = wind10 * ratio
    virtual_air = (air+273.15)*(1+0.61*qa)
    virtual_bulk = (bulk+273.15)*(1+0.61*qb)
    ri = 9.81*config.temperature_height_m*(virtual_air-virtual_bulk) / (
        virtual_air*wind_t.square())
    # Piecewise physical regimes; operands are valid on both branches.
    positive_ri = torch.relu(ri)
    negative_ri = torch.relu(-ri)
    factor = torch.where(ri >= 0,
        (1+config.stable_ri_coefficient*positive_ri).pow(-2),
        torch.sqrt(1+config.unstable_ri_coefficient*negative_ri))
    exchange = density_air*config.neutral_exchange*factor*wind10
    dvirtual_bulk = 1+0.61*qb+0.61*(bulk+273.15)*dqb
    dri_dbulk = -9.81*config.temperature_height_m*dvirtual_bulk/(virtual_air*wind_t.square())
    dfactor_dri = torch.where(ri >= 0,
        -2*config.stable_ri_coefficient*(1+config.stable_ri_coefficient*positive_ri).pow(-3),
        -.5*config.unstable_ri_coefficient/(1+config.unstable_ri_coefficient*negative_ri).sqrt())
    dexchange_dbulk = density_air*config.neutral_exchange*wind10*dfactor_dri*dri_dbulk
    friction_water = torch.sqrt(density_air/config.water_density_kg_m3
                                *config.neutral_drag) * wind10
    conductance = config.water_thermal_conductivity_w_m_k * (
        1/config.background_skin_thickness_m
        + friction_water/(config.skin_saunders_constant
                          *config.water_kinematic_viscosity_m2_s))

    def component_and_derivative(skin):
        qs, dqs = _humidity(skin, p)
        kelvin = skin+273.15
        longwave = config.emissivity*(lw-5.670374419e-8*kelvin.pow(4))
        sensible = -exchange*config.cp_air_j_kg_k*(skin-air)
        latent = -exchange*config.latent_heat_j_kg*(qs-qa)
        components = torch.stack((longwave, sensible, latent), -1)
        derivative = (-4*config.emissivity*5.670374419e-8*kelvin.pow(3)
                      - exchange*config.cp_air_j_kg_k
                      - exchange*config.latent_heat_j_kg*dqs)
        return components, derivative

    # Numerical root iterations do not need a full backward tape. Reintroduce
    # the implicit response once at the converged root. At residual zero,
    # d[Ts - R(Ts,p)/R_Ts]/dp = -R_p/R_Ts, including forcing/stability terms.
    # This changes the graph, not the solved boundary or its first derivative.
    with torch.no_grad():
        skin = bulk
        for _ in range(config.newton_iterations):
            components, dq_dskin = component_and_derivative(skin)
            residual = components.sum(-1)-conductance*(skin-bulk)
            skin = skin-residual/(dq_dskin-conductance)
    skin = skin.detach()
    components, dq_dskin = component_and_derivative(skin)
    residual = components.sum(-1)-conductance*(skin-bulk)
    skin = skin-residual/(dq_dskin-conductance)
    components, dq_dskin = component_and_derivative(skin)
    flux = components.sum(-1)
    residual = flux-conductance*(skin-bulk)
    qs, dqs = _humidity(skin, p)
    partial_bulk = -dexchange_dbulk*(config.cp_air_j_kg_k*(skin-air)
                                    +config.latent_heat_j_kg*(qs-qa))
    dskin_dbulk = (conductance+partial_bulk)/(conductance-dq_dskin)
    component_derivative = torch.stack((
        -4*config.emissivity*5.670374419e-8*(skin+273.15).pow(3)*dskin_dbulk,
        -config.cp_air_j_kg_k*(dexchange_dbulk*(skin-air)+exchange*dskin_dbulk),
        -config.latent_heat_j_kg*(dexchange_dbulk*(qs-qa)+exchange*dqs*dskin_dbulk)), -1)
    return skin, components, flux, residual, conductance, ri, factor, component_derivative


def skin_bulk_exchange(temperature_bulk_c: Tensor, forcing_core9: Tensor,
                       top_cell_depth_m=None, config: SkinBulkConfig | None = None):
    """Solve Qns(Tskin; Tb,F) = G(F)*(Tskin-Tbulk), positive down.

    Qns is LW+H+LE. G is positive thin-layer conductance. Stability is evaluated
    at bulk temperature during this boundary solve. Numerical Newton iterations
    run without a backward tape, followed by one implicit differentiable
    correction in bulk state, forcing and tensor parameters.
    Returned dQ/dTbulk includes skin response AND stability dependence by implicit
    differentiation: Ts'=(G+partial_Q/partial_Tbulk)/(G-partial_Q/partial_Tskin).
    Every constituent derivative is analytic; finite differences are used only
    by verification, avoiding three repeated nonlinear solves per model hour.
    It can replace q_atmosphere/d_atmosphere in the existing implicit step.

    top_cell_depth_m is accepted only as interface metadata: skin thickness is
    not the resolved bulk-cell depth, and no second top-cell resistance is added.
    Caller must route ice-bearing columns to its previous phase boundary.
    """
    c = config or SkinBulkConfig()
    tb = temperature_bulk_c.to(torch.float64)
    f = forcing_core9.to(dtype=torch.float64, device=tb.device)
    skin, components, flux, residual, conductance, ri, stability, component_derivative = _skin_core(tb, f, c)
    return {
        "skin_c": skin,
        "flux_components_w_m2": components,  # [LW,H,LE], downward positive
        "dcomponent_dbulk_w_m2_k": component_derivative,
        "into_water_non_solar_w_m2": flux,
        "dflux_dbulk_w_m2_k": component_derivative.sum(-1),
        "closure_residual_w_m2": residual,
        "skin_conductance_w_m2_k": conductance,
        "bulk_richardson": ri,
        "stability_factor": stability,
        "scope": "liquid reduced skin; no skin SW absorption; ice path unchanged",
    }


def _layer_parameter(x, layers: Tensor) -> Tensor:
    value = _64(x, layers)
    if value.shape == layers.shape[:-1]:
        value = value.unsqueeze(-1)
    return torch.broadcast_to(value, layers.shape)


def _band_partition(irradiance: Tensor, dz: Tensor, area: Tensor, kd: Tensor):
    optical_depth = kd*dz
    start_tau = torch.cat((torch.zeros_like(optical_depth[..., :1]),
                           optical_depth.cumsum(-1)[..., :-1]), -1)
    beam = irradiance[..., None]*torch.exp(-start_tau)
    x = optical_depth
    small = x.abs() < 1e-3
    denominator = torch.where(small, torch.ones_like(x), x)
    exponential = torch.exp(-x)
    j0_regular = -torch.expm1(-x)/denominator
    j1_regular = (-torch.expm1(-x)-x*exponential)/denominator.square()
    j0_series = 1-x/2+x.square()/6-x.pow(3)/24+x.pow(4)/120-x.pow(5)/720
    j1_series = .5-x/3+x.square()/8-x.pow(3)/30+x.pow(4)/144-x.pow(5)/840
    j0 = dz*torch.where(small, j0_series, j0_regular)
    j1 = dz.square()*torch.where(small, j1_series, j1_regular)
    slope = area.diff(dim=-1)/dz
    water = beam*kd*(area[..., :-1]*j0+slope*j1)
    bed = beam*(-slope)*j0
    bottom = irradiance*torch.exp(-optical_depth.sum(-1))*area[..., -1]
    return water, bed, bottom


def two_band_shortwave_partition(shortwave_w_m2: Tensor, depth_edges_m: Tensor,
                                area_interfaces_m2: Tensor, kd_visible_m_inv,
                                *, kd_nir_m_inv, visible_fraction, albedo):
    """Exact piecewise-linear-A / piecewise-constant-Kd radiation partition.

    Each layer: Pwater=integral A*k*I dz; Pbed=integral -A'*I dz.
    Final horizontal bottom receives A(H)*I(H). Thus sum(water+bed)+bottom
    equals (1-albedo)*SW*A(0), including the exact transparent-water limit.
    Bed power is NOT immediate water heating. The caller must evolve its bed
    reservoir and pair any bed-water exchange with the opposite water flux.

    Optical coefficients, fraction, and albedo are mandatory explicit inputs.
    No oceanic Jerlov constants are silently assigned to all lakes. Band names
    are effective visible/NIR groups, not a claim of fully spectral transfer.
    PAR Kd is not automatically broadband heat absorption. A source conversion
    or an explicitly documented effective-band approximation is required.
    Shapes: [..., N+1] geometry; kd scalar, [...], or [...,N]; SW [...].
    """
    z = depth_edges_m.to(torch.float64)
    a = area_interfaces_m2.to(dtype=torch.float64, device=z.device)
    dz = z.diff(dim=-1)
    sw = _64(shortwave_w_m2, z)
    fraction = _64(visible_fraction, z)
    incoming = sw*(1-_64(albedo, z))
    vis = _band_partition(incoming*fraction, dz, a,
                           _layer_parameter(kd_visible_m_inv, dz))
    nir = _band_partition(incoming*(1-fraction), dz, a,
                           _layer_parameter(kd_nir_m_inv, dz))
    water, bed, bottom = [v+n for v, n in zip(vis, nir)]
    entering = incoming*a[..., 0]
    residual = water.sum(-1)+bed.sum(-1)+bottom-entering
    return {
        "water_w": water, "bed_w": bed, "bottom_w": bottom,
        "entering_w": entering, "balance_residual_w": residual,
        "visible_water_w": vis[0], "nir_water_w": nir[0],
        "visible_bed_w": vis[1], "nir_bed_w": nir[1],
        "scope": "liquid two-band vertical rays; no sediment evolution or ice transfer",
    }


def configuration_manifest(config: SkinBulkConfig | None = None):
    return {"scope": SCOPE, "skin": asdict(config or SkinBulkConfig()),
            "salinity": "EOS80/SP, g/kg adapter uses declared sea-salt equivalent",
            "radiation": "explicit optical inputs; piecewise-linear-area partition"}
