from app.simulation.monte_carlo import run_monte_carlo


def _probs():
    return [0.30, 0.27, 0.25, 0.23, 0.22, 0.20, 0.18, 0.16, 0.14]


def test_reproducible_with_seed():
    r1 = run_monte_carlo(5.5, 1.2, 4.3, _probs(), n_simulations=5000, seed=7)
    r2 = run_monte_carlo(5.5, 1.2, 4.3, _probs(), n_simulations=5000, seed=7)
    assert r1.mean == r2.mean
    assert r1.median == r2.median
    assert r1.probability_by_k == r2.probability_by_k


def test_probabilities_sum_to_one():
    r = run_monte_carlo(5.5, 1.2, 4.3, _probs(), n_simulations=10000, seed=1)
    total = sum(r.probability_by_k.values())
    assert abs(total - 1.0) < 1e-9


def test_higher_innings_increases_mean_strikeouts():
    low = run_monte_carlo(3.0, 1.0, 4.3, _probs(), n_simulations=8000, seed=2)
    high = run_monte_carlo(7.0, 1.0, 4.3, _probs(), n_simulations=8000, seed=2)
    assert high.mean > low.mean


def test_minimum_25000_simulations_supported():
    r = run_monte_carlo(5.5, 1.2, 4.3, _probs(), n_simulations=25000, seed=3)
    assert r.n_simulations == 25000
    assert len(r.raw_strikeouts) == 25000


def test_percentiles_ordered():
    r = run_monte_carlo(5.5, 1.2, 4.3, _probs(), n_simulations=10000, seed=4)
    p = r.percentiles
    assert p[10] <= p[25] <= p[50] <= p[75] <= p[90]


def test_fifteen_or_more_bucket_exists():
    r = run_monte_carlo(9.0, 0.5, 5.5, [0.4] * 9, n_simulations=10000, seed=5)
    assert 15 in r.probability_by_k
