"""Pure phase-constrained atmospheric skin boundary on actual TACO weather.

This is a component, not a phase lake adapter or a state time integrator.
Only the native bulk atmosphere is called. The safeguarded cold Robin solver
changes the numerical root search, not the atmospheric constitutive law. The
added
melting complementarity boundary conserves atmospheric energy without a free
heat source, and never clips the learned water/node-temperature prediction.
"""

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_constant(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Constant):
        raise ValueError("Actual source literal constant missing/nonunique: " + name)
    return float(assignments[0].value.value)


def original_weather_module(path):
    """Inspected standalone constitutive module, without any model imports."""
    name = "_surface_melt_native_weather_" + sha256(path)[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        previous = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
    return sys.modules[name]


class SurfaceMeltingBoundary:
    version = "TACO_NATIVE_FRESHWATER_SKIN_MELT_SAFEGUARDED_ROBIN_V2"
    full_phase_adapter_qualified = False
    # The native law rejects Ts+273.15<=100 K. This is a numerical interior
    # point for a root bracket, not a projection of the learned solid state.
    native_surface_absolute_lower_K = 100.0
    numerical_lower_margin_K = 8 * torch.finfo(torch.float64).eps * 273.15
    cold_root_search_tolerance_K = 1e-11
    cold_root_acceptance_tolerance_K = 1e-8
    cold_root_max_iterations = 96
    surface_energy_acceptance_absolute_W_m2 = 1e-7
    surface_energy_acceptance_relative = 2e-12
    native_cap_tie_roundoff_relative = 32 * torch.finfo(torch.float64).eps

    def __init__(self, taco_core_evidence_root):
        root = Path(taco_core_evidence_root).resolve()
        engine = root / "inputs/PINO281_2001_inputs/engine"
        self.weather_path = engine / "v35_7_g0_coupled_r1/weather.py"
        self.constants_path = engine / "v35_7_enthalpy_window_operator_r1/snow_ice_components.py"
        if not self.weather_path.is_file() or not self.constants_path.is_file():
            raise FileNotFoundError("Actual TACO weather/component-constant source required")
        self.native = original_weather_module(self.weather_path)
        self.latent_fusion_J_kg = original_constant(self.constants_path, "LATENT_J_KG")
        self.cp_ice_J_kg_K = original_constant(self.constants_path, "CP_ICE_J_KG_K")
        self.cp_water_J_kg_K = original_constant(self.constants_path, "CP_WATER_J_KG_K")
        if min(self.latent_fusion_J_kg, self.cp_ice_J_kg_K, self.cp_water_J_kg_K) <= 0:
            raise ValueError("Nonpositive actual source physical constants")
        self.dependency_code_hashes = {
            str(path): sha256(path)
            for path in (self.weather_path, self.constants_path, Path(__file__).resolve())
        }

    def _weather(self, core9, shape):
        """Use the unchanged original physical_weather on each original row.

        Its logged raw-SF -> [0,TP] interpretation is retained. Negative SW/TP
        remains rejected; this component does not invent a forcing repair.
        """
        core9 = torch.broadcast_to(core9, shape + (9,))
        native = [
            self.native.physical_weather(row, partition_policy=self.native.PARTITION_POLICY)
            for row in core9.reshape(-1, 9)
        ]
        return {key: torch.stack([row[key] for row in native]).reshape(shape) for key in native[0]}

    def _cold_robin(self, node, resistance, weather, coefficient, freezing, cap):
        """Solve the same native Robin law inside its cold-skin domain.

        G=(Ts-Tnode)/R-Q(Ts), G'=1/R-Q'(Ts)>0. Evaluating this row-scaled
        residual avoids R-amplified cancellation without changing its root.
        A cold active set has G(Tf)>=0. A true lower/upper sign bracket is
        mandatory; no endpoint is substituted for a missing root. Accepted
        Newton steps stay strictly in the bracket, otherwise a midpoint is
        used. All search calls use the original atmosphere, including the
        original vapor applicability check and supply-limited flux.

        The detached, converged root receives one ordinary differentiable
        Newton correction, as in the original solver: at G=0 this gives the
        implicit derivative -G_parameter/G_Ts. No straight-through state or
        altered flux is introduced.
        """
        inverse_resistance = resistance.reciprocal()
        if not bool(torch.isfinite(inverse_resistance).all()):
            raise ValueError("Native Robin inverse resistance outside finite numerical domain")

        def equation(skin):
            q, dq, _ = self.native.atmosphere(
                skin, weather, coefficient, frozen=True, max_vapor=cap
            )
            g = (skin - node) * inverse_resistance - q
            slope = inverse_resistance - dq
            if not all(bool(torch.isfinite(value).all()) for value in (q, dq, g, slope)):
                raise RuntimeError("Nonfinite native Robin equation; no state/flux repair")
            if bool((dq > 0).any()) or bool((slope <= 0).any()):
                raise RuntimeError("Native cold Robin monotonicity contract violated")
            return g, slope

        def reject_ambiguous_cap_tie(skin):
            if cap is None:
                return
            # Original torch.minimum has a half-gradient at equality, whereas
            # native dq uses the >=cap one-sided zero vapor derivative. Do not
            # silently certify a mismatched implicit derivative at that native
            # kink. A tiny relative roundoff band covers indistinguishable
            # floating equality, without modifying either original flux.
            with torch.no_grad():
                _, uncapped_dq, potential = self.native.atmosphere(
                    skin, weather, coefficient, frozen=True, max_vapor=None
                )
                _, capped_dq, _ = self.native.atmosphere(
                    skin, weather, coefficient, frozen=True, max_vapor=cap
                )
                scale = torch.maximum(potential.abs(), cap.abs()).clamp_min(
                    torch.finfo(torch.float64).tiny
                )
                near_tie = (potential - cap).abs() <= self.native_cap_tie_roundoff_relative * scale
                ambiguous = near_tie & (uncapped_dq != capped_dq)
                if bool(ambiguous.any()):
                    raise RuntimeError(
                        "NATIVE_COLD_ROBIN_VAPOR_CAP_EQUALITY_DERIVATIVE_AMBIGUITY: "
                        "original minimum tie and handwritten dq use different slopes; no derivative substitution"
                    )

        with torch.no_grad():
            lower = torch.full_like(
                node, self.native_surface_absolute_lower_K - 273.15 + self.numerical_lower_margin_K
            )
            upper = freezing.detach().clone()
            gl, _ = equation(lower)
            gu, _ = equation(upper)
            if bool((gl > 0).any()) or bool((gu < 0).any()):
                bad = (gl > 0) | (gu < 0)
                raise RuntimeError(
                    "NATIVE_COLD_ROBIN_NO_ROOT_IN_APPLICABILITY_DOMAIN: "
                    f"{int(bad.sum())} row(s); lower_G_max={float(gl.max()):.17g}; "
                    f"upper_G_min={float(gu.min()):.17g}; endpoints are not substitute solutions"
                )
            # Equality is accepted only when the original equation really
            # vanishes at that legal endpoint, including cold/warm transition.
            root = torch.where(gl == 0, lower, torch.where(gu == 0, upper, (lower + upper) * 0.5))
            done = torch.zeros_like(root, dtype=torch.bool)
            iterations = 0
            for iterations in range(1, self.cold_root_max_iterations + 1):
                g, slope = equation(root)
                done = done | (g.abs() / slope <= self.cold_root_search_tolerance_K)
                if bool(done.all()):
                    break
                lower = torch.where((~done) & (g < 0), root, lower)
                upper = torch.where((~done) & (g >= 0), root, upper)
                candidate = root - g / slope
                admissible = torch.isfinite(candidate) & (candidate > lower) & (candidate < upper)
                next_root = torch.where(admissible, candidate, (lower + upper) * 0.5)
                root = torch.where(done, root, next_root)
            else:
                raise RuntimeError("NATIVE_COLD_ROBIN_BRACKETED_ROOT_DID_NOT_CONVERGE")

        # Reattach all node/R/weather/coefficient/cap derivatives at the root.
        reject_ambiguous_cap_tie(root)
        g, slope = equation(root)
        skin = root - g / slope
        if not bool(torch.isfinite(skin).all()) or bool(((skin + 273.15) <= 100).any()):
            raise RuntimeError(
                "Native Robin implicit correction left strict 100K applicability domain"
            )
        if bool((skin > freezing).any()):
            raise RuntimeError("Native cold Robin implicit correction crossed melting point")
        reject_ambiguous_cap_tie(skin)
        final_g, final_slope = equation(skin)
        root_error = final_g / final_slope
        if bool((root_error.abs() > self.cold_root_acceptance_tolerance_K).any()):
            raise RuntimeError(
                "Native bracketed Robin boundary failed original equilibrated root tolerance"
            )
        return skin, root_error, iterations

    def evaluate(
        self,
        node_temperature_C,
        node_to_skin_resistance_m2_K_W,
        core9,
        bulk_coefficient,
        *,
        freezing_temperature_C=0.0,
        max_vapor_kg_m2_s=None,
    ):
        """Return conductive/melting powers and native vapor rate per covered area.

        Downward atmospheric power Q(Ts) = q_cond + q_melt,
        q_cond=(Ts-Tnode)/R, Ts<=Tf, q_melt>=0,
        (Tf-Ts)*q_melt=0. Positive q_cond heats the retained solid reservoir;
        negative q_cond cools it. Melt rate q_melt/Lf is NOT an additional
        energy input. The caller must transfer melt mass/enthalpy exactly once.

        Scope: the original native freshwater atmospheric law, Tf=0C, a
        present solid layer with positive supplied node-to-skin resistance.
        An absent layer needs the separate native open-water boundary. Solar
        absorption belongs to the separate snow/ice/water partition/storage
        contract. No time step or arbitrary per-query mass-supply cap is chosen.

        The active-set event uses a documented one-sided torch.where derivative.
        We reject a decoded solid node above Tf; we never project that prediction.
        """
        node = torch.as_tensor(node_temperature_C, dtype=torch.float64)

        def like(value):
            return torch.as_tensor(value, dtype=torch.float64, device=node.device)

        resistance = like(node_to_skin_resistance_m2_K_W)
        coefficient = like(bulk_coefficient)
        core9 = like(core9)
        freezing = like(freezing_temperature_C)
        if core9.ndim < 1 or core9.shape[-1] != 9:
            raise ValueError("Native CORE9 final dimension must be nine")
        shape = torch.broadcast_shapes(
            node.shape, resistance.shape, coefficient.shape, freezing.shape, core9.shape[:-1]
        )
        node, resistance, coefficient, freezing = (
            torch.broadcast_to(value, shape) for value in (node, resistance, coefficient, freezing)
        )
        if any(
            not bool(torch.isfinite(value).all())
            for value in (node, resistance, coefficient, freezing, core9)
        ):
            raise ValueError("Nonfinite physical input; no missing-field zero fill")
        if bool((resistance <= 0).any()) or bool((coefficient <= 0).any()):
            raise ValueError("Positive present-solid resistance and bulk coefficient required")
        if bool((freezing != 0).any()):
            raise ValueError(
                "Native freshwater Tf=0 only; saline freezing needs separate composition qualification"
            )
        if bool((node > freezing).any()):
            raise ValueError("Decoded solid node above melting point; prediction is not clipped")
        vapor_cap = None
        if max_vapor_kg_m2_s is not None:
            vapor_cap = torch.broadcast_to(like(max_vapor_kg_m2_s), shape)
            if not bool(torch.isfinite(vapor_cap).all()) or bool((vapor_cap < 0).any()):
                raise ValueError("Explicit nonnegative finite vapor-supply rate required")
        weather = self._weather(core9, shape)
        at_freezing = self.native.atmosphere(
            freezing, weather, coefficient, frozen=True, max_vapor=vapor_cap
        )
        conduction_at_freezing = (freezing - node) / resistance
        surplus_at_freezing = at_freezing[0] - conduction_at_freezing
        melting = surplus_at_freezing > 0

        # Only the cold active set needs a Robin root. Its numerical search is
        # bounded by the native atmosphere domain and the physical melting
        # point; neither the node prediction nor a constitutive flux is clipped.
        cold_indices = torch.nonzero((~melting).reshape(-1), as_tuple=False).flatten()
        skin = freezing.reshape(-1)
        root_error = torch.zeros_like(skin)
        root_iterations = 0
        if len(cold_indices):
            selected = {key: value.reshape(-1)[cold_indices] for key, value in weather.items()}
            cap = None if vapor_cap is None else vapor_cap.reshape(-1)[cold_indices]
            cold_skin, cold_error, root_iterations = self._cold_robin(
                node.reshape(-1)[cold_indices],
                resistance.reshape(-1)[cold_indices],
                selected,
                coefficient.reshape(-1)[cold_indices],
                freezing.reshape(-1)[cold_indices],
                cap,
            )
            skin = skin.index_copy(0, cold_indices, cold_skin)
            root_error = root_error.index_copy(0, cold_indices, cold_error)
        skin = skin.reshape(shape)
        atmosphere, atmosphere_derivative, vapor = self.native.atmosphere(
            skin, weather, coefficient, frozen=True, max_vapor=vapor_cap
        )
        conductive = (skin - node) / resistance
        melt = torch.where(melting, atmosphere - conductive, torch.zeros_like(atmosphere))
        residual = atmosphere - conductive - melt
        if not all(
            bool(torch.isfinite(value).all())
            for value in (skin, atmosphere, conductive, melt, vapor, residual)
        ):
            raise RuntimeError("Nonfinite native surface boundary result")
        if bool((skin > freezing + 1e-8).any()) or bool((melt < 0).any()):
            raise RuntimeError("Surface complementarity violated; no result clipping")
        energy_scale = torch.maximum(
            torch.ones_like(atmosphere),
            torch.maximum(atmosphere.abs(), conductive.abs() + melt.abs()),
        )
        energy_tolerance = (
            self.surface_energy_acceptance_absolute_W_m2
            + self.surface_energy_acceptance_relative * energy_scale
        )
        if bool((residual.abs() > energy_tolerance).any()):
            raise RuntimeError(
                "NATIVE_SURFACE_NUMERICAL_ENERGY_RESOLUTION_FAILURE: "
                "actual Q-conduction-melt does not close; no Kelvin-only acceptance or free heat patch"
            )
        return dict(
            surface_temperature_C=skin,
            downward_atmosphere_W_m2=atmosphere,
            downward_conductive_W_m2=conductive,
            melt_power_W_m2=melt,
            melt_mass_rate_kg_m2_s=melt / self.latent_fusion_J_kg,
            outward_native_vapor_kg_m2_s=vapor,
            surface_energy_residual_W_m2=residual,
            surface_complementarity_residual_K_W_m2=(freezing - skin) * melt,
            melting_active=melting,
            native_atmosphere_temperature_derivative_W_m2_K=atmosphere_derivative,
            surplus_at_freezing_W_m2=surplus_at_freezing,
            cold_robin_equilibrated_root_error_K=root_error.reshape(shape),
            cold_robin_numerical_solver=dict(
                version=self.version,
                iterations=root_iterations,
                method="true_sign_bracket_safeguarded_Newton_midpoint_fallback",
                strict_native_absolute_lower_K=self.native_surface_absolute_lower_K,
                numerical_lower_margin_K=self.numerical_lower_margin_K,
                root_search_tolerance_K=self.cold_root_search_tolerance_K,
                root_acceptance_tolerance_K=self.cold_root_acceptance_tolerance_K,
                energy_acceptance_absolute_W_m2=self.surface_energy_acceptance_absolute_W_m2,
                energy_acceptance_relative=self.surface_energy_acceptance_relative,
                native_cap_tie_roundoff_relative=self.native_cap_tie_roundoff_relative,
                state_or_forcing_clipped=False,
                endpoint_substituted_for_missing_root=False,
            ),
            native_precipitation_partition={
                key: weather[key] for key in ("rain_m_hour", "snow_m_hour", "sf_adjustment_m_hour")
            },
            event_derivative="cold Robin implicit derivative; warm Ts=Tf and q_melt=Q(Tf)-(Tf-Tnode)/R; one-sided active set at surplus=0",
            all_powers_per_covered_area=True,
            solar_accounted_here=False,
            mass_salt_draft_ledger_complete=False,
            full_phase_adapter_qualified=False,
        )
