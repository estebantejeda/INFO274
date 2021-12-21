import numpy as np
from scipy.stats import gamma

def hello_world() -> str:
    """
    Escriba una función que retorne el string Hola Mundo
    """
    return("Hola Mundo")
    #raise NotImplementedError("Implemente esta función")


def fit_mistery_data(data: np.ndarray) -> np.ndarray:
    
    """
    Input: Un ndarray con datos numéricos
    Output: Los parámetros de una distribución gamma ajustada mediante máxima verosimilitud sobre los datos
    Hint: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
    """
    args = gamma.fit(data);
    #raise NotImplementedError("Implemente esta función")
    return args
