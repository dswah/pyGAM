import numpy as np

from pygam import LinearGAM, s


def test_summary_returns_dictionary():
    """
    Test that the summary() method returns a dictionary containing
    the correct model and term statistics.
    """
    # Setup: Create a simple model
    X = np.random.rand(100, 1)
    y = np.random.rand(100)
    gam = LinearGAM(s(0)).fit(X, y)

    # Execution: Call the method we refactored
    res = gam.summary()

    # Check that it is a dictionary
    assert isinstance(res, dict)

    # Check that top-level model keys exist
    assert "model" in res
    assert "terms" in res

    # Check that raw data matches internal statistics
    assert res["model"]["AIC"] == gam.statistics_["AIC"]
    assert res["model"]["loglikelihood"] == gam.statistics_["loglikelihood"]

    # Check that term-level data is captured
    assert len(res["terms"]) == len(gam.terms)

    # Cross-check EDoF precision (rounded in table, raw in dict)
    dict_edof = res["terms"][0]["edof"]
    stat_edof = gam.statistics_["edof_per_coef"][gam.terms.get_coef_indices(0)].sum()

    assert np.isclose(dict_edof, stat_edof)
