import numpy as np
import scipy.stats

def transicion_x_pasos(steps: int, s0_index: int, P: np.ndarray) -> np.ndarray:
    """Retorna el estado de la cadena x pasos en el futuro

    Argumentos:
        steps (int): El número de pasos a simular
        s0_index (int): El índice del estado inicial
        P (np.ndarray): La matriz de transición

    Se recomienda haber revisado al menos hasta: 
    https://phuijse.github.io/MonteCarloBook/lectures/markov.html#ejemplo-cadena-de-dos-estados
    """
    return np.dot(s0_index, np.linalg.matrix_power(P, steps))


def markov_monte_carlo(chains: int, steps: int, s0: int, P: np.ndarray, rseed:int=None):
    """Retorna una matriz de tamaño chains x horizon con las cadenas simuladas
    
    Argumentos:
        chains (int): La cantidad de cadenas a simular
        steps (int): El número de pasos a simular (horizonte de predicción)
        s0_index (int): El índice del estado inicial
        P (np.ndarray): La matriz de transición
        rseed (int): La semilla para el generador de números aleatorios

    Se recomienda haber revisado al menos hasta: 
    https://phuijse.github.io/MonteCarloBook/lectures/markov.html#algoritmo-general-para-simular-una-cadena-de-markov-discreta
    """
    if rseed is not None:
        np.random.seed(rseed)

   
    horizon = steps
    states = np.zeros(shape=(chains, horizon), dtype='int')
    states[:, 0] = 1 # Estado inicial para todas las cadenas
    for i in range(chains):
        for j in range(1, horizon):
            states[i, j] = np.argmax(scipy.stats.multinomial.rvs(n=1, p=P[states[i, j-1], :], size=1))

    return states

