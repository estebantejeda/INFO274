from typing import Dict
import random as rnd
import numpy as np
import scipy.stats


def knapsack_utility(x: np.ndarray, config: Dict):
    """
    Entradas:
    x: Vector de configuración de elementos en la mochila
    config: Diccionario de configuración del problema

    La función debe retornar la utilidad, es decir la función U(x)
    """
    u_x=0
    for v in range(0,len(x)):
        u_x = u_x + config["v"][v]*x[v]
    
    return u_x    

def knapsack_is_valid(x: np.ndarray, config: Dict):
    """
    Entradas:
    x: Vector de configuración de elementos en la mochila 
    config: Diccionario de configuración del problema

    La función debe retornar True si el peso combinado de los elementos en la mochila es menor o igual que la capacidad de la mochila
    """
    acm=0
    for e in range(0,len(x)):
        if(x[e]==1):
            acm = config["w"][e] + acm
    
    if acm <= config["C"]:
        return True
    else:
        return False
    

def knapsack_propose(x: np.ndarray):
    """
    Entradas:
    x: Vector de configuración de elementos en la mochila

    La función retorna una configuración modificada. Para esto tome la configuración original como base y luego seleccione un número al azar entre 0 y len(x). Modifique el elemento de x en la posición seleccionada. Si el objeto estaba en la mochila remuevalo, de lo contrario agréguelo.
    

    """
    x_cp = x.copy()
    
    azar = rnd.randint(0,len(x)-1)
    if(x_cp[azar]):
        x_cp[azar]=0
    else:
        x_cp[azar]=1
    
    return x_cp
    
    

def knapsack_acceptance_criterion(x: np.ndarray, x_new: np.ndarray, config: Dict, T: float):
    """
    Entradas
    x: Vector de configuración actual
    x_new: Vector de configuración propuesto
    config: Diccionario de configuración del problema
    T: Temperatura 

    La función debe retornar alpha, es decir el mínimo entre 1 y p(x_new)/p(x)
    """
    
    utilidad1 = knapsack_utility(x,config)
    utilidad2 = knapsack_utility(x_new,config)
    p1 = np.exp(utilidad1/T)
    p2 = np.exp(utilidad2/T)
    rst = p1 / p2
    return rst
    
    

def simulated_annealing(mix_time: int, problem_config: Dict, T: float):
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
        x_new = knapsack_propose(x)
        if knapsack_is_valid(x_new, problem_config):
            if us[i] < knapsack_acceptance_criterion(x, x_new, problem_config, T):
                x = x_new
                if np.amax(utility_history) < knapsack_utility(x, problem_config):
                    best_x = x

        utility_history[i] = knapsack_utility(x, problem_config)

    return utility_history, best_x, knapsack_utility(best_x, problem_config)






