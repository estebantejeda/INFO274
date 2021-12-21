from typing import Dict
import numpy as np
import scipy.stats


def knapsack_utility(x: np.ndarray, config: Dict):
    """
    Entradas:
    x: Vector de configuración de elementos en la mochila
    config: Diccionario de configuración del problema

    La función debe retornar la utilidad, es decir la función U(x)
    """
    return np.sum(x*config['v'])

def knapsack_is_valid(x: np.ndarray, config: Dict):
    """
    Entradas:
    x: Vector de configuración de elementos en la mochila 
    config: Diccionario de configuración del problema

    La función debe retornar True si el peso combinado de los elementos en la mochila es menor o igual que la capacidad de la mochila
    """
    return np.sum(x*config['w']) <= config["C"]

def knapsack_propose(x: np.ndarray):
    """
    Entradas:
    x: Vector de configuración de elementos en la mochila

    La función retorna una configuración modificada. Para esto tome la configuración original como base y luego seleccione un número al azar entre 0 y len(x). Modifique el elemento de x en la posición seleccionada. Si el objeto estaba en la mochila remuevalo, de lo contrario agréguelo.
    """
    x_new = x.copy()
    idx = np.random.randint(len(x_new))
    x_new[idx] = 1 - x_new[idx]
    return x_new

def knapsack_acceptance_criterion(x: np.ndarray, x_new: np.ndarray, config: Dict, T: float):
    """
    Entradas
    x: Vector de configuración actual
    x_new: Vector de configuración propuesto
    config: Diccionario de configuración del problema
    T: Temperatura 

    La función debe retornar alpha, es decir el mínimo entre 1 y p(x_new)/p(x)
    """
    U = knapsack_utility(x, config)
    U_new = knapsack_utility(x_new, config)    
    alpha = np.amin([1, np.exp((U_new - U)/T)])
    return alpha

def simulated_annealing(mix_time: int, problem_config: Dict, T0: float, beta: float):
    """
    Entradas:
    mix_time: Largo de la cadena
    problem_config: Diccionario de configuración del problema
    T: Temperatura para el criterio de aceptación
    """
    # Inicialización
    m = len(problem_config["v"])
    x = np.zeros(shape=(m,))
    best_x = x
    utility_history = np.zeros(shape=(mix_time,))
    # Precalcular los valores de u
    us = scipy.stats.uniform.rvs(size=mix_time)
    # Ciclo principal
    for i in range(1, mix_time): 
        T = T0*beta**i
        x_new = knapsack_propose(x)
        if knapsack_is_valid(x_new, problem_config):            
            if us[i] <= knapsack_acceptance_criterion(x, x_new, problem_config, T):
                x = x_new
                if np.amax(utility_history) < knapsack_utility(x, problem_config):
                    best_x = x

        utility_history[i] = knapsack_utility(x, problem_config)

    return utility_history, best_x, knapsack_utility(best_x, problem_config)