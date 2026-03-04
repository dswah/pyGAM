import json
import os
import pickle
import tempfile

import numpy as np
import pytest

from pygam import (
    GAM,
    ExpectileGAM,
    GammaGAM,
    InvGaussGAM,
    LinearGAM,
    LogisticGAM,
    PoissonGAM,
    SERIALIZATION_VERSION,
)
from pygam.terms import f, l, s, te


def _roundtrip_file(model, *, compress=True):
    fd, path = tempfile.mkstemp(suffix=".pkl" if not compress else ".pkl.gz")
    os.close(fd)
    try:
        model.save(path, compress=compress)
        loaded = GAM.load(path)
    finally:
        if os.path.exists(path):
            os.remove(path)
    return loaded


def _fit_linear_gam(mcycle_X_y):
    X, y = mcycle_X_y
    return LinearGAM().fit(X, y)


def _fit_logistic_gam(default_X_y):
    X, y = default_X_y
    return LogisticGAM().fit(X, y)


def _fit_poisson_gam(coal_X_y):
    X, y = coal_X_y
    return PoissonGAM().fit(X, y)


def _fit_gamma_gam(hepatitis_X_y):
    X, y = hepatitis_X_y
    return GammaGAM().fit(X, np.clip(y, a_min=1e-6, a_max=None))


def _fit_invgauss_gam(mcycle_X_y):
    X, y = mcycle_X_y
    return InvGaussGAM().fit(X, np.abs(y) + 1)


def _fit_expectile_gam(mcycle_X_y):
    X, y = mcycle_X_y
    return ExpectileGAM().fit(X, y)


def test_save_load_unfitted(tmp_path):
    gam = LinearGAM(lam=0.1, max_iter=42)
    path = tmp_path / "model.pkl"
    gam.save(path)
    loaded = LinearGAM.load(path)

    assert isinstance(loaded, LinearGAM)
    assert loaded._is_fitted is False
    assert loaded.max_iter == gam.max_iter
    assert np.allclose(loaded.lam, gam.lam)


def test_save_load_fitted_linear_gam(mcycle_X_y):
    X, y = mcycle_X_y
    gam = _fit_linear_gam(mcycle_X_y)
    loaded = _roundtrip_file(gam)

    assert isinstance(loaded, LinearGAM)
    assert loaded._is_fitted
    assert np.allclose(gam.predict(X), loaded.predict(X))


def test_save_load_fitted_logistic_gam(default_X_y):
    X, _ = default_X_y
    gam = _fit_logistic_gam(default_X_y)
    loaded = _roundtrip_file(gam)

    assert isinstance(loaded, LogisticGAM)
    assert loaded._is_fitted
    assert np.allclose(gam.predict_proba(X), loaded.predict_proba(X))


def test_save_load_fitted_poisson_gam(coal_X_y):
    X, _ = coal_X_y
    gam = _fit_poisson_gam(coal_X_y)
    loaded = _roundtrip_file(gam)

    assert isinstance(loaded, PoissonGAM)
    assert loaded._is_fitted
    assert np.allclose(gam.predict(X), loaded.predict(X))


def test_save_load_fitted_gamma_gam(hepatitis_X_y):
    X, _ = hepatitis_X_y
    gam = _fit_gamma_gam(hepatitis_X_y)
    loaded = _roundtrip_file(gam)

    assert isinstance(loaded, GammaGAM)
    assert loaded._is_fitted
    assert np.allclose(gam.predict(X), loaded.predict(X))


def test_save_load_fitted_invgauss_gam(mcycle_X_y):
    X, _ = mcycle_X_y
    gam = _fit_invgauss_gam(mcycle_X_y)
    loaded = _roundtrip_file(gam)

    assert isinstance(loaded, InvGaussGAM)
    assert loaded._is_fitted
    assert np.allclose(gam.predict(X), loaded.predict(X))


def test_save_load_fitted_expectile_gam(mcycle_X_y):
    X, _ = mcycle_X_y
    gam = _fit_expectile_gam(mcycle_X_y)
    loaded = _roundtrip_file(gam)

    assert isinstance(loaded, ExpectileGAM)
    assert loaded._is_fitted
    assert np.allclose(gam.predict(X), loaded.predict(X))


def test_save_load_compressed(mcycle_X_y, tmp_path):
    X, _ = mcycle_X_y
    gam = _fit_linear_gam(mcycle_X_y)
    path = tmp_path / "model.pkl.gz"
    gam.save(path, compress=True)
    loaded = LinearGAM.load(path)

    assert isinstance(loaded, LinearGAM)
    assert np.allclose(gam.predict(X), loaded.predict(X))
    assert path.stat().st_size > 0


def test_save_load_uncompressed(mcycle_X_y, tmp_path):
    X, _ = mcycle_X_y
    gam = _fit_linear_gam(mcycle_X_y)
    path = tmp_path / "model.pkl"
    gam.save(path, compress=False)
    loaded = LinearGAM.load(path)

    assert isinstance(loaded, LinearGAM)
    assert np.allclose(gam.predict(X), loaded.predict(X))
    assert path.stat().st_size > 0


def test_load_class_dispatch(mcycle_X_y):
    gam = _fit_linear_gam(mcycle_X_y)
    loaded = _roundtrip_file(gam)

    assert isinstance(loaded, LinearGAM)


def test_load_class_mismatch(mcycle_X_y, tmp_path):
    gam = _fit_linear_gam(mcycle_X_y)
    path = tmp_path / "model.pkl"
    gam.save(path)

    with pytest.raises(TypeError):
        LogisticGAM.load(path)


def test_save_load_with_custom_terms(mcycle_X_y):
    X, y = mcycle_X_y
    gam = LinearGAM(s(0) + l(0) + f(0) + te(0, 0)).fit(X, y)
    loaded = _roundtrip_file(gam)

    assert isinstance(loaded, LinearGAM)
    assert np.allclose(gam.predict(X), loaded.predict(X))


def test_save_load_statistics_preserved(mcycle_X_y):
    gam = _fit_linear_gam(mcycle_X_y)
    loaded = _roundtrip_file(gam)

    for key in gam.statistics_.keys():
        assert key in loaded.statistics_


def test_save_load_predictions_match(mcycle_X_y):
    X, _ = mcycle_X_y
    gam = _fit_linear_gam(mcycle_X_y)
    loaded = _roundtrip_file(gam)

    assert np.allclose(gam.predict(X), loaded.predict(X))


def test_save_load_confidence_intervals(mcycle_X_y):
    X, _ = mcycle_X_y
    gam = _fit_linear_gam(mcycle_X_y)
    loaded = _roundtrip_file(gam)

    conf_a = gam.confidence_intervals(X)
    conf_b = loaded.confidence_intervals(X)
    assert np.allclose(conf_a, conf_b)


def test_to_dict_from_dict(mcycle_X_y):
    X, _ = mcycle_X_y
    gam = _fit_linear_gam(mcycle_X_y)
    data = gam.to_dict()
    loaded = GAM.from_dict(data)

    assert isinstance(loaded, LinearGAM)
    assert loaded._is_fitted
    assert hasattr(loaded, "coef_")
    assert loaded.coef_.shape == gam.coef_.shape


def test_to_dict_json_serializable(mcycle_X_y):
    gam = _fit_linear_gam(mcycle_X_y)
    data = gam.to_dict()
    # this should not raise
    json.dumps(data)


def test_future_version_error(tmp_path):
    gam = LinearGAM()
    path = tmp_path / "future.pkl"

    payload = {
        "pygam_version": "0.0.0",
        "serialization_version": SERIALIZATION_VERSION + 1,
        "gam_class": "LinearGAM",
        "params": gam.get_params(deep=True),
        "is_fitted": False,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)

    with pytest.raises(ValueError):
        GAM.load(path)


def test_direct_pickle_with_versioning(mcycle_X_y):
    gam = _fit_linear_gam(mcycle_X_y)
    dumped = pickle.dumps(gam)
    loaded = pickle.loads(dumped)

    assert isinstance(loaded, LinearGAM)
    assert loaded._is_fitted
