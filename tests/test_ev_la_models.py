"""Tests for Task 3: EV/LA outcome model replacement in causal_models.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

# Add src/models to path so causal_models can be imported directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "models"))

from causal_models import (  # noqa: E402
    fit_outcome_models,
    _xrv_from_shape,
    disruption_tax_split,
    indirect_effect,
    OUTCOME_FEATURES,
    ANGULAR_DEVS,
)


def make_synthetic_df(n=300, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "vert_attack_angle_dev": rng.normal(0, 2, n),
        "horz_attack_angle_dev": rng.normal(0, 2, n),
        "swing_path_tilt_dev":   rng.normal(0, 2, n),
        "plate_x":  rng.uniform(-1.5, 1.5, n),
        "plate_z":  rng.uniform(1.0, 4.0, n),
        "balls":    rng.integers(0, 4, n),
        "strikes":  rng.integers(0, 3, n),
        "is_bip":   rng.integers(0, 2, n),
        "is_contact": rng.integers(0, 2, n),
        "is_whiff":  rng.integers(0, 2, n),
        "is_swing":  np.ones(n, dtype=int),
        "launch_speed": rng.uniform(60, 110, n),
        "launch_angle": rng.uniform(-30, 50, n),
        "delta_run_exp": rng.uniform(-0.5, 0.5, n),
        "pc150_emp_dev_x": rng.normal(0, 0.05, n),
        "pc150_emp_dev_z": rng.normal(0, 0.05, n),
        "pc150_emp_x_proj": rng.uniform(-1.5, 1.5, n),
        "pc150_emp_z_proj": rng.uniform(1.0, 4.0, n),
    })
    # Non-BIP rows should have NaN EV/LA
    df.loc[df["is_bip"] == 0, ["launch_speed", "launch_angle"]] = np.nan
    return df


@pytest.fixture(scope="module")
def fitted_models():
    df = make_synthetic_df()
    return df, fit_outcome_models(df)


def test_fit_outcome_models_returns_6_items(fitted_models):
    _, result = fitted_models
    assert len(result) == 6, f"Expected 6 returns, got {len(result)}"


def test_fit_outcome_models_no_xwoba_model(fitted_models):
    """Confirm xwoba_model is gone — 6-tuple should be (bip, foul, ev, la, ev_la_rv, whiff_rv)."""
    _, result = fitted_models
    bip_m, foul_m, ev_m, la_m, ev_la_rv_m, whiff_rv = result
    # ev_m and la_m should be regressors (predict returns continuous values)
    feat = make_synthetic_df()[OUTCOME_FEATURES].fillna(0)
    ev_preds = ev_m.predict(feat)
    la_preds = la_m.predict(feat)
    # Regressors output floats, not integer class labels
    assert ev_preds.dtype in (np.float32, np.float64)
    assert la_preds.dtype in (np.float32, np.float64)


def test_bip_and_foul_models_still_work(fitted_models):
    df, result = fitted_models
    bip_m, foul_m, ev_m, la_m, ev_la_rv_m, whiff_rv = result
    feat = df[OUTCOME_FEATURES].fillna(0)
    bip_proba = bip_m.predict_proba(feat)[:, 1]
    foul_proba = foul_m.predict_proba(feat)[:, 1]
    assert len(bip_proba) == len(df)
    assert ((bip_proba >= 0) & (bip_proba <= 1)).all()
    assert ((foul_proba >= 0) & (foul_proba <= 1)).all()


def test_whiff_rv_is_dict(fitted_models):
    _, result = fitted_models
    whiff_rv = result[5]
    assert isinstance(whiff_rv, dict)
    assert "default" in whiff_rv


def test_xrv_from_shape_uses_ev_la_chain(fitted_models):
    """_xrv_from_shape should return a Series of length n using EV/LA chain."""
    df, result = fitted_models
    bip_m, foul_m, ev_m, la_m, ev_la_rv_m, whiff_rv = result
    foul_rv = {(b, s): -0.08 for b in range(4) for s in range(3)}
    foul_rv["default"] = -0.08

    xrv = _xrv_from_shape(
        df, bip_m, foul_m, ev_m, la_m, ev_la_rv_m,
        whiff_rv, foul_rv,
        commit_ms=150, zero_angular=False, zero_spatial=False, empirical=True,
    )
    assert len(xrv) == len(df)
    assert xrv.notna().all()


def test_xrv_from_shape_zero_angular_differs_from_realized(fitted_models):
    df, result = fitted_models
    bip_m, foul_m, ev_m, la_m, ev_la_rv_m, whiff_rv = result
    foul_rv = {(b, s): -0.08 for b in range(4) for s in range(3)}
    foul_rv["default"] = -0.08
    kwargs = dict(bip_model=bip_m, foul_model=foul_m, exit_velo_model=ev_m,
                  launch_angle_model=la_m, ev_la_rv_model=ev_la_rv_m,
                  whiff_rv=whiff_rv, foul_rv=foul_rv, commit_ms=150, empirical=True)

    xrv_realized = _xrv_from_shape(df, **kwargs, zero_angular=False, zero_spatial=False)
    xrv_intended = _xrv_from_shape(df, **kwargs, zero_angular=True,  zero_spatial=True)
    # They should differ (non-zero deviations exist in synthetic data)
    assert not np.allclose(xrv_realized.values, xrv_intended.values, atol=1e-6)


def _make_mock_mediator_models(df):
    """Return a dict of mock mediator models with .params attribute."""
    dev_x = "pc150_emp_dev_x"
    dev_z = "pc150_emp_dev_z"
    mock_models = {}
    for dev_col in ANGULAR_DEVS:
        m = MagicMock()
        m.params = {dev_x: 0.5, dev_z: 0.3}
        mock_models[dev_col] = m
    return mock_models


def test_disruption_tax_split_new_ev_columns(fitted_models):
    df, result = fitted_models
    bip_m, foul_m, ev_m, la_m, ev_la_rv_m, whiff_rv = result
    foul_rv = {(b, s): -0.08 for b in range(4) for s in range(3)}
    foul_rv["default"] = -0.08
    mediator_models = _make_mock_mediator_models(df)

    out = disruption_tax_split(
        df, bip_m, foul_m, ev_m, la_m, ev_la_rv_m,
        whiff_rv, foul_rv, mediator_models,
        commit_ms=150, empirical=True,
    )

    for col in ["ev_realized", "la_realized", "ev_intended", "la_intended", "ev_suppression"]:
        assert col in out.columns, f"Missing column: {col}"


def test_disruption_tax_split_ev_columns_nan_on_non_bip(fitted_models):
    df, result = fitted_models
    bip_m, foul_m, ev_m, la_m, ev_la_rv_m, whiff_rv = result
    foul_rv = {(b, s): -0.08 for b in range(4) for s in range(3)}
    foul_rv["default"] = -0.08
    mediator_models = _make_mock_mediator_models(df)

    out = disruption_tax_split(
        df, bip_m, foul_m, ev_m, la_m, ev_la_rv_m,
        whiff_rv, foul_rv, mediator_models,
        commit_ms=150, empirical=True,
    )

    non_bip_mask = df["is_bip"] != 1
    if non_bip_mask.any():
        assert out.loc[non_bip_mask, "ev_realized"].isna().all()
        assert out.loc[non_bip_mask, "la_realized"].isna().all()
        assert out.loc[non_bip_mask, "ev_suppression"].isna().all()


def test_disruption_tax_split_ev_suppression_formula(fitted_models):
    df, result = fitted_models
    bip_m, foul_m, ev_m, la_m, ev_la_rv_m, whiff_rv = result
    foul_rv = {(b, s): -0.08 for b in range(4) for s in range(3)}
    foul_rv["default"] = -0.08
    mediator_models = _make_mock_mediator_models(df)

    out = disruption_tax_split(
        df, bip_m, foul_m, ev_m, la_m, ev_la_rv_m,
        whiff_rv, foul_rv, mediator_models,
        commit_ms=150, empirical=True,
    )

    bip_mask = df["is_bip"] == 1
    bip_rows = out[bip_mask]
    expected = bip_rows["ev_realized"] - bip_rows["ev_intended"]
    np.testing.assert_allclose(bip_rows["ev_suppression"].values, expected.values, rtol=1e-5)


def test_disruption_tax_split_existing_columns_unchanged(fitted_models):
    df, result = fitted_models
    bip_m, foul_m, ev_m, la_m, ev_la_rv_m, whiff_rv = result
    foul_rv = {(b, s): -0.08 for b in range(4) for s in range(3)}
    foul_rv["default"] = -0.08
    mediator_models = _make_mock_mediator_models(df)

    out = disruption_tax_split(
        df, bip_m, foul_m, ev_m, la_m, ev_la_rv_m,
        whiff_rv, foul_rv, mediator_models,
    )

    for col in ["disruption_tax", "distortion_tax", "selection_tax",
                "spatial_distortion_tax", "distortion_share", "_xrv_intended"]:
        assert col in out.columns, f"Missing existing column: {col}"
