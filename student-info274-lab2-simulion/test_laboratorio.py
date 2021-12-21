import pytest
import numpy as np
import laboratorio


@pytest.mark.parametrize("input, expected", [(1, np.array([0.75, 0.25])),
                                             (2, np.array([0.625, 0.375])),
                                             (3, np.array([0.5625, 0.4375]))])
def test_transicion_x_pasos(input, expected):
    P = np.array([[0.75, 0.25], [0.25, 0.75]])
    answer = laboratorio.transicion_x_pasos(input, 0, P)
    np.testing.assert_allclose(answer, expected, rtol=1e-5)


def test_markov_monte_carlo():
    P = np.array([[0.75, 0.25], [0.25, 0.75]])
    answer = laboratorio.markov_monte_carlo(2, 2, 0, P, rseed=12345)
    np.testing.assert_allclose(answer, np.array([[1, 0], [1, 1]]), rtol=1e-5)