import numpy as np
from pygam import LinearGAM


def test_json_save_load(tmp_path):
    X = np.random.randn(50, 2)
    y = np.random.randn(50)

    gam = LinearGAM().fit(X, y)

    file_path = tmp_path / "model.json"
    gam.save(file_path)

    loaded = LinearGAM.load(file_path)

    assert np.allclose(gam.coef_, loaded.coef_)