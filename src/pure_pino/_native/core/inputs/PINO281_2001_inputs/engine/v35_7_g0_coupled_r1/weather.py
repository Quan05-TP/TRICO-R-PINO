"""Explicit physical interpretation of immutable hourly CORE9 observations.

This layer reconciles a noisy snowfall estimate with observed total precipitation;
it does not rewrite the archive or fill absent hours. Turbulent fluxes use a bulk
formulation, with the same mass flux in the water and energy budgets.
"""
import torch

FIELDS = ('air_temperature_c', 'dewpoint_c', 'wind_u_m_s', 'wind_v_m_s',
          'pressure_pa', 'shortwave_w_m2', 'longwave_w_m2',
          'core_precipitation_m_hour', 'snowfall_m_hour_raw')
PARTITION_POLICY = 'fixed_observed_tp_least_squares_snow_in_0_tp_v1'
SIGMA = 5.670374419e-8


def physical_weather(row, *, partition_policy):
    if row.shape != (9,) or not bool(torch.isfinite(row).all()):
        raise ValueError('Nine finite original weather fields required; no missing fill')
    if partition_policy != PARTITION_POLICY:
        raise ValueError('Explicit, logged precipitation interpretation required')
    air, dew, u, v, p, sw, lw, tp, sf = row.double().unbind()
    if bool((p <= 0) | (air <= -150) | (dew <= -150) | (sw < 0) | (lw < 0) | (tp < 0)):
        raise ValueError('Unphysical atmospheric forcing; retain and audit the original hour')
    snow = torch.minimum(sf.clamp_min(0), tp)
    return dict(air=air, dew=dew, wind=torch.hypot(u, v), pressure=p, sw=sw, lw=lw,
                rain_m_hour=tp-snow, snow_m_hour=snow,
                sf_adjustment_m_hour=snow-sf, raw_sf=sf, raw_tp=tp)


def saturation_humidity(temperature, pressure, *, over_ice=False):
    a, b = (22.587, 273.86) if over_ice else (17.625, 243.04)
    vapor = 610.94 * torch.exp(a*temperature/(temperature+b))
    dv = vapor*a*b/(temperature+b).square()
    if bool((vapor >= pressure).any()):
        raise ValueError('Surface thermodynamics outside bulk-flux applicability')
    den = pressure-.378*vapor
    return .622*vapor/den, .622*pressure*dv/den.square()


def atmosphere(surface_c, weather, coefficient, *, frozen=False, max_vapor=None):
    """Downward W/m2, its nonpositive derivative, outward vapor kg/m2/s.

    Latent heat is paid here. Advected condensed-phase enthalpy is removed in
    the mass substep, so it is not subtracted a second time in this heat flux.
    Snow sublimation uses ice saturation and sublimation latent heat.
    """
    w = weather
    absolute = surface_c+273.15
    if bool((absolute <= 100).any()):
        raise ValueError('Invalid absolute surface temperature')
    density = w['pressure']/(287.05*(w['air']+273.15))
    qs, dqs = saturation_humidity(surface_c, w['pressure'], over_ice=frozen)
    qa, _ = saturation_humidity(w['dew'], w['pressure'])
    exchange = density*coefficient*w['wind']
    vapor = exchange*(qs-qa)
    vapor_derivative = exchange*dqs
    if max_vapor is not None:
        # Supply-limited evaporation/sublimation. Both latent ENERGY and mass
        # use this same actual flux; no unpaid potential evaporation is kept.
        limited = vapor >= max_vapor
        vapor = torch.minimum(vapor, max_vapor)
        vapor_derivative = torch.where(limited, torch.zeros_like(vapor_derivative), vapor_derivative)
    latent = 2.834e6 if frozen else 2.500e6
    emissivity = .99 if frozen else .97
    flux = emissivity*(w['lw']-SIGMA*absolute.pow(4)) - exchange*1005*(surface_c-w['air']) - latent*vapor
    derivative = -4*emissivity*SIGMA*absolute.pow(3)-exchange*1005-latent*vapor_derivative
    return flux, derivative, vapor


def resistive_surface(node_c, resistance, weather, coefficient, *, max_vapor):
    """Implicit massless snow/ice skin, including supply-limited sublimation."""
    with torch.no_grad():
        skin = node_c.detach().clone()
        for _ in range(30):
            q, dq, _ = atmosphere(skin, weather, coefficient, frozen=True, max_vapor=max_vapor)
            residual = skin-node_c-resistance*q
            correction = residual/(1-resistance*dq)
            skin = skin-correction
            if bool((correction.abs() < 1e-10).all()): break
    q, dq, _ = atmosphere(skin, weather, coefficient, frozen=True, max_vapor=max_vapor)
    residual = skin-node_c-resistance*q
    skin = skin-residual/(1-resistance*dq)
    q, dq, vapor = atmosphere(skin, weather, coefficient, frozen=True, max_vapor=max_vapor)
    # Row-equilibrated root error in kelvin, not R-amplified cancellation.
    # A real failure had |F|=1.8e-6 but |F/F'|=1.5e-14 K at R=1.9e7.
    root_error = (skin-node_c-resistance*q)/(1-resistance*dq)
    if not bool(torch.isfinite(skin).all()) or bool(root_error.abs().max() > 1e-8):
        raise RuntimeError('Implicit phase solver skin boundary did not converge')
    return q, dq/(1-resistance*dq), vapor, skin
