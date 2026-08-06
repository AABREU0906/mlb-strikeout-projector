from app.features.shrinkage import shrink_named, shrink_rate


def test_small_sample_pulls_toward_prior():
    result = shrink_named(observed_rate=0.40, observed_n=10, prior_rate=0.224, key="batter_k_rate_overall")
    assert result.reliability < 0.2
    assert abs(result.shrunk_rate - 0.224) < abs(0.40 - 0.224)


def test_large_sample_stays_close_to_observed():
    result = shrink_named(observed_rate=0.30, observed_n=600, prior_rate=0.224, key="batter_k_rate_overall")
    assert result.reliability > 0.85
    assert abs(result.shrunk_rate - 0.30) < 0.02


def test_zero_observations_returns_pure_prior():
    result = shrink_rate(observed_rate=0.5, observed_n=0, prior_rate=0.224, stabilization_n=60)
    assert result.shrunk_rate == 0.224
    assert result.reliability == 0.0


def test_is_small_sample_flag():
    small = shrink_named(0.3, 5, 0.224, "batter_k_rate_overall")
    large = shrink_named(0.3, 1000, 0.224, "batter_k_rate_overall")
    assert small.is_small_sample is True
    assert large.is_small_sample is False


def test_unknown_key_raises():
    import pytest
    with pytest.raises(KeyError):
        shrink_named(0.3, 100, 0.224, "not_a_real_key")
