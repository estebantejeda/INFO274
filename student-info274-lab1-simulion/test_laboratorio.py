import pytest
import numpy as np
import parte1
import parte2

@pytest.fixture()
def mock_prices():
    return np.array([10.0, 90.0, 6.0, 40.0])

def test_part1():
    assert len(parte1.crear_naipes()) == 52
    assert parte1.reyes_juntos(['TK', 'DK', 'DQ']) == 1
    assert parte1.reyes_juntos(['TK', 'DQ', 'DK']) == 0

def test_part2(mock_prices):
    params = parte2.fit_model(mock_prices)
    assert len(params) == 2
    assert params[0] == pytest.approx(0.46209812037329695, rel=1e-5)
    assert params[1] == pytest.approx(2.244978986793704, rel=1e-5)
    assert parte2.brownian_motion(mock_prices[-1], params[0], 0.0) == pytest.approx(63.49604207872798, rel=1e-5)
