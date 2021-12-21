import numpy as np
from random import shuffle
from typing import List

def crear_naipes() -> List[str]:
    """
    Retorna una lista con los 52 naipes de la baraja francesa

    Cada carta debe ser un string. La primera letra debe ser T, D, C o P (palos). A continuación debe ir una A, J, Q, K o un número del 2 al 10
    """
    cartas=[]
    palos = ["T","D","C","P"]
    secuencia = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
    for a in palos:
        for b in secuencia:
            cartas.append(a+b)
    return cartas

def reyes_juntos(naipes: List) -> int:
    """
    Returna 1 si hay al menos dos reyes juntos o 0 en caso contrario
    """
    acm=0
    for i in range(1,len(naipes)):
        if naipes[i-1][1] == "K" and naipes[i][1]== "K":
            acm=acm+1
    if acm>0:
        return 1
    else:
        return 0


def barajar(naipes: List) -> List:
    """
    Retorna una versión "barajada" de naipes
    HINT: https://docs.python.org/3/library/random.html#random.shuffle
    """
    shuffle(naipes)

    return naipes
