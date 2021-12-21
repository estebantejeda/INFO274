import numpy as np
from typing import Tuple


def fit_model(prices: np.ndarray) -> Tuple[float, float]:
    """
    Returna una tupla con los parámetros alpha y sigma calculados a partir de los precios de cierre
    """
    vec = []
    vec1 = []
    for i in range(1, len(prices)):
        precio_actual = prices[i]
        precio_anterior = prices[i-1]
        vec.append(np.log(precio_actual/precio_anterior))
    vec1 = [np.mean(vec), np.std(vec)]
    return vec1


def brownian_motion(past_price: float, alpha: float, sigma: float) -> float:
    """
    Una función que retorna el precio Pt dado el Pt-1 y los parámetros alpha y beta
    Hint: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html?highlight=randn#numpy.random.randn
    """
    z_t = np.random.randn()
    k = alpha+z_t*sigma
    e = np.exp(k)
    return past_price*e
    raise NotImplementedError("Debe implementar esta función")
    


"""
import datetime as dt
import pandas as pd
import pandas_datareader.data as web
def get_yahoo_stocks_close_price(start = dt.datetime(2009, 12, 30), end = dt.datetime(2019, 5, 29), plot=False):
    df = web.DataReader('AAPL', 'yahoo', start, end)['Close']
    return  df.values
"""
