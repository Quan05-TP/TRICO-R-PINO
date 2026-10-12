"""Direct-model contracts on manufactured exteriors, not fitted accuracy."""

from copy import deepcopy
from types import SimpleNamespace

import torch

from examples.external_field import (
    HOUR_S,
    coupled_exterior,
    exterior_batch,
    make_fixture_model,
)
from pure_pino import native_adapter
from pure_pino.clock_features import CLOCK_FEATURE_CONTRACT_ID


class TargetBomb:
    def __array__(self, *args, **kwargs):
        raise AssertionError("Target metadata must not become model input")

    def __float__(self):
        raise AssertionError("Target metadata must not be read")


class TargetMetadata(dict):
    def __getitem__(self, key):
        if key.startswith("target_"):
            raise AssertionError("Model requested target metadata: " + key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key.startswith("target_"):
            raise AssertionError("Model requested target metadata: " + key)
        return super().get(key, default)


class ExteriorOnlyBatch(SimpleNamespace):
    def __getattribute__(self, name):
        if name in {"values", "depth", "temperature", "target_labels"}:
            raise AssertionError("Model requested a target field: " + name)
        return super().__getattribute__(name)


def test_factory_selects_new_clock_and_float64():
    model = make_fixture_model(width=8)
    assert model.cfg.clock_feature_contract_id == CLOCK_FEATURE_CONTRACT_ID
    assert model.cfg.token_clock_feature_mode == "split_native_support_affine"
    assert next(model.parameters()).dtype == torch.float64


def test_causal_future_tokens_and_target_metadata_do_not_change_field():
    ordinary = exterior_batch(nodes=4)
    batch = ExteriorOnlyBatch(**vars(ordinary))
    batch.external = [
        SimpleNamespace(
            meta=TargetMetadata(
                {
                    "target_temperature": TargetBomb(),
                    "target_depth": TargetBomb(),
                }
            )
        )
    ]
    changed = ExteriorOnlyBatch(**vars(deepcopy(ordinary)))
    changed.external = batch.external
    changed.tokens[:, 1:] += 1000.0
    model = make_fixture_model(width=8).eval()
    query = torch.tensor([0.0, 900.0, 1800.0], dtype=torch.float64)
    with torch.no_grad():
        original_state = model.initialize(batch, 48 * HOUR_S, "causal")
        changed_state = model.initialize(changed, 48 * HOUR_S, "causal")
        original = model.decode(batch, original_state, query, 0.0, 1800.0, 48 * HOUR_S, "causal")
        mutated = model.decode(changed, changed_state, query, 0.0, 1800.0, 48 * HOUR_S, "causal")
    for name in ("node_raw", "global_raw", "memory", "process_control"):
        torch.testing.assert_close(getattr(original, name), getattr(mutated, name), rtol=0, atol=0)


def late_temperature(model, batch, adapter):
    initial = model.initialize(batch, 48 * HOUR_S)
    first = model.decode(
        batch,
        initial,
        torch.tensor([24 * HOUR_S], dtype=torch.float64),
        0.0,
        24 * HOUR_S,
        48 * HOUR_S,
    )
    carry = model.carry(first)
    second = model.decode(
        batch,
        carry,
        torch.tensor([24 * HOUR_S, 48 * HOUR_S], dtype=torch.float64),
        24 * HOUR_S,
        48 * HOUR_S,
        48 * HOUR_S,
    )
    decoded = adapter.decode(second, coupled_exterior(batch.nodes, 2))
    return decoded.observables["water_temperature_C"][:, -1].mean(), (initial, carry, second)


def test_two_block_continuity_and_late_gradient_reaches_initial_state():
    batch = exterior_batch(nodes=4)
    model = make_fixture_model(width=8)
    loss, (initial, carry, second) = late_temperature(model, batch, native_adapter())
    for name in ("node_raw", "global_raw", "memory"):
        torch.testing.assert_close(
            getattr(second, name)[:, 0], getattr(carry, name), rtol=0, atol=0
        )
    gradients = torch.autograd.grad(loss, (initial.node_raw, initial.memory))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_parameter_ad_fd_through_carry_and_actual_native_decoder():
    batch = exterior_batch(nodes=4)
    model = make_fixture_model(width=8)
    adapter = native_adapter()
    parameter = model.initial_node.bias
    loss, _ = late_temperature(model, batch, adapter)
    ad = torch.autograd.grad(loss, parameter)[0][0]
    assert ad.abs() > 1e-6
    original = parameter.detach().clone()
    epsilon = 1e-5
    values = []
    try:
        for sign in (1.0, -1.0):
            with torch.no_grad():
                parameter.copy_(original)
                parameter[0] += sign * epsilon
                value, _ = late_temperature(model, batch, adapter)
                values.append(value.detach())
    finally:
        with torch.no_grad():
            parameter.copy_(original)
    fd = (values[0] - values[1]) / (2 * epsilon)
    torch.testing.assert_close(ad, fd, rtol=2e-4, atol=1e-6)
