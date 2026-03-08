def test_invgauss_sample_with_default_scale():
    from pygam.distributions import InvGaussDist

    dist = InvGaussDist(scale=None)
    sample = dist.sample(mu=1.0)

    assert sample is not None
