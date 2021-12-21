import pytest
import numpy as np
import laboratorio

@pytest.fixture()
def mock_knapsack():
    return {'C': 2, 'v': np.array([0., 2.]), 'w': np.array([2., 2.])}

def test_knapsack_utility(mock_knapsack):
    assert laboratorio.knapsack_utility([0, 0], mock_knapsack) == 0
    assert laboratorio.knapsack_utility([1, 0], mock_knapsack) == 0
    assert laboratorio.knapsack_utility([0, 1], mock_knapsack) == 2

def test_knapsack_is_valid(mock_knapsack):
    assert laboratorio.knapsack_is_valid([0, 0], mock_knapsack)
    assert laboratorio.knapsack_is_valid([0, 1], mock_knapsack)
    assert not laboratorio.knapsack_is_valid([1, 1], mock_knapsack)

def test_knapsack_propose():
    x = np.zeros(shape=(10,))
    x_new = laboratorio.knapsack_propose(x)
    assert len(x_new) == len(x)
    assert min(x_new) == 0
    assert max(x_new) == 1

