import pickle
import numpy as np
import laboratorio

def test_part1():
    assert laboratorio.hello_world() == "Hola Mundo"

def test_part2():
    with open("data/mistery_data.pkl", "rb") as f:
        x = pickle.load(f)
    ans = laboratorio.fit_mistery_data(x)
    assert len(ans) == 3
    res = (6.746014286918019, 0.41389920532236746, 1.0104850789126822)
    return np.testing.assert_allclose(ans, res, rtol=1e-5)
