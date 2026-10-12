"""FP64 conservative implicit water/sediment-slab coupling via Schur elimination.

Each water cell can exchange with one local bed reservoir. Solar interception
belongs to bed storage, not immediately to water. Geometry uses *projected*
bed area; no unobserved bottom slope or surface area is invented.

For each cell, positive heat flux q goes into water:
    Cb (Tb_new - Tb_old)/dt = Pb - G (Tb_new - Tw_new).
Eliminating Tb_new gives q(Tw) = a*(Tb_old-Tw) + f*Pb, where
    D=Cb+dt*G, a=G*Cb/D, f=dt*G/D.
The water solver therefore receives q(Tw_ref), dq/dTw=-a. After the water
enthalpy solve, reconstruct Tb_new and retain it across every replay/window.
Zero-area cells have no reservoir (Cb=G=Pb=0); they are handled algebraically,
without epsilon heat capacities, temperature clipping or learned heat sources.

Physical reference: MacKay (2019), GMD 12, 3045--3054:
https://gmd.copernicus.org/articles/12/3045/2019/
The cited sand Cv=2.13e6 J/m3/K and k=2.5 W/m/K are explicit material candidates,
not measurements for these lakes. This module uses a 1-m single slab with an
adiabatic lower boundary, not that paper's full 10-m/bottom-temperature scheme.
It does not implement sediment freezing, sediment layers, or geothermal input.
Water phase changes remain governed by the caller's existing enthalpy solver.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class BedConfig:
    volumetric_heat_capacity_j_m3_k: float = 2.13e6
    thickness_m: float = 1.0
    conductivity_w_m_k: float = 2.5
    mean_temperature_distance_fraction: float = 0.5


def _field(x, reference: Tensor) -> Tensor:
    value = torch.as_tensor(x, dtype=torch.float64, device=reference.device)
    if value.shape == reference.shape[:-1]:
        value = value.unsqueeze(-1)
    return torch.broadcast_to(value, reference.shape)


def bed_parameters_from_hypsography(area_interfaces_m2: Tensor,
                                   config: BedConfig | None = None):
    """One bed reservoir per cell from side projection plus terminal bottom.

    Inputs require nonincreasing, nonnegative A(z). This function does not clip
    incompatible geometry; area validation belongs to the already-bound lake
    geometry dataset. All coefficients are shared physical candidates.
    """
    c = config or BedConfig()
    if min(c.volumetric_heat_capacity_j_m3_k, c.thickness_m,
           c.conductivity_w_m_k, c.mean_temperature_distance_fraction) <= 0:
        raise ValueError('Bed material/distance parameters must be positive')
    area = area_interfaces_m2.to(torch.float64)
    side = area[..., :-1]-area[..., 1:]
    bottom = F.pad(area[..., -1:], (side.shape[-1]-1, 0))
    projected = side+bottom
    capacity = projected*c.thickness_m*c.volumetric_heat_capacity_j_m3_k
    conductance = projected*c.conductivity_w_m_k / (
        c.mean_temperature_distance_fraction*c.thickness_m)
    return {'projected_area_m2': projected, 'side_projected_area_m2': side,
            'terminal_bottom_area_m2': bottom,
            'heat_capacity_j_k': capacity, 'conductance_w_k': conductance,
            'active_reservoir': projected > 0,
            'parameter_source': 'explicit shared sand-property candidate; not per-lake measurement',
            'config': asdict(c)}


def combine_bed_radiation(side_bed_power_w: Tensor,
                          bottom_power_w: Tensor) -> Tensor:
    """Attach terminal horizontal bottom absorption once, to the last cell."""
    side = side_bed_power_w.to(torch.float64)
    bottom = torch.as_tensor(bottom_power_w, dtype=torch.float64, device=side.device)
    return side+F.pad(bottom[..., None], (side.shape[-1]-1, 0))


def eliminate_bed(water_reference_c: Tensor, bed_temperature_c: Tensor,
                  heat_capacity_j_k: Tensor, conductance_w_k: Tensor,
                  absorbed_bed_power_w: Tensor, dt_seconds: Tensor | float):
    """Return bed contributions compatible with existing implicit_heat_step.

    Existing solver uses P(T)=power+derivative*(T-T_reference). Accordingly add
    `water_power_w` to water power and `water_derivative_w_k` to its diagonal
    derivative. Do NOT also add Pb to water power. Water enthalpy/ice nonlinear
    solve is unchanged. Array shape [...,N]; dt may be scalar or [...].

    A truly absent cell has Cb=G=Pb=0. A nonzero Pb supplied to absent capacity
    and conductance is reported as `unassigned_input_power_w`, not silently
    zero-filled; the caller can diagnose its radiation/geometry binding.
    """
    tw = water_reference_c.to(torch.float64)
    tb = _field(bed_temperature_c, tw)
    cb = _field(heat_capacity_j_k, tw)
    g = _field(conductance_w_k, tw)
    pb = _field(absorbed_bed_power_w, tw)
    dt = _field(dt_seconds, tw)
    denominator = cb+dt*g
    active = denominator != 0
    # 1 is used only on the algebraically absent branch, never as a minimum
    # physical capacity or denominator of an active reservoir.
    divisor = torch.where(active, denominator, torch.ones_like(denominator))
    relaxation = g*cb/divisor
    transmitted_fraction = dt*g/divisor
    power = relaxation*(tb-tw)+transmitted_fraction*pb
    derivative = -relaxation
    return {'water_power_w': power, 'water_derivative_w_k': derivative,
            'water_reference_c': tw, 'bed_previous_temperature_c': tb,
            'heat_capacity_j_k': cb, 'conductance_w_k': g,
            'bed_input_power_w': pb, 'dt_seconds': dt,
            'denominator_j_k': denominator, 'active_reservoir': active,
            'relaxation_w_k': relaxation, 'transmitted_fraction': transmitted_fraction,
            'unassigned_input_power_w': torch.where(active, torch.zeros_like(pb), pb)}


def reconstruct_bed(water_temperature_new_c: Tensor, eliminated: dict):
    """Reconstruct implicit slab state and paired water/bed energy diagnostics.

    Uses the Schur expression for water heat transfer, which remains stable for
    very high G (avoids G times the subtraction of nearly equal temperatures).
    Inactive reservoir temperature is represented as 0 with active mask false;
    it is not a measured 0-C bed state and carries exactly zero energy.
    """
    tw = water_temperature_new_c.to(torch.float64)
    cb = eliminated['heat_capacity_j_k']
    g = eliminated['conductance_w_k']
    pb = eliminated['bed_input_power_w']
    dt = eliminated['dt_seconds']
    old = eliminated['bed_previous_temperature_c']
    active = eliminated['active_reservoir']
    divisor = torch.where(active, eliminated['denominator_j_k'],
                          torch.ones_like(eliminated['denominator_j_k']))
    new = (cb*old+dt*pb+dt*g*tw)/divisor
    new = torch.where(active, new, torch.zeros_like(new))
    q = eliminated['relaxation_w_k']*(old-tw)+eliminated['transmitted_fraction']*pb
    into_water = dt*q
    change = cb*(new-old)
    absorbed = dt*pb
    return {'temperature_c': new, 'active_reservoir': active,
            'heat_content_j': cb*new, 'heat_change_j': change,
            'transfer_to_water_w': q, 'transfer_to_water_j': into_water,
            'absorbed_radiation_j': absorbed,
            'bed_balance_residual_j': change+into_water-absorbed,
            'unassigned_input_power_w': eliminated['unassigned_input_power_w']}


def configuration_manifest(config: BedConfig | None = None):
    return {'schema': 'TRICO_EXPLICIT_BED_SLAB_V2', 'parameters': asdict(config or BedConfig()),
            'material_parameters_measured_per_lake': False,
            'area': 'horizontal side interception Ai-Ai+1 plus final bottom Ai+1',
            'state': 'one sensible-heat slab temperature per water cell; persist continuously',
            'lower_boundary': 'adiabatic; no free/geothermal heat term',
            'water_phase': 'existing water enthalpy solver; no new ice/snow completion claim',
            'reference': 'https://gmd.copernicus.org/articles/12/3045/2019/'}
