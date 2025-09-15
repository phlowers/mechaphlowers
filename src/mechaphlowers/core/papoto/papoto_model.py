# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
from scipy import optimize


_ZETA = 1.
def papoto(
        a: np.ndarray,
        HG: np.ndarray,
        VG: np.ndarray,
        HD: np.ndarray,
        VD: np.ndarray,
        H1: np.ndarray,
        V1: np.ndarray,
        H2: np.ndarray,
        V2: np.ndarray
    ) -> np.ndarray:

    # déclaration pour l'appel aux fonctions hyperboliques

    # Dim np
    # Set np = WorksheetFunction

    # déclaration des angle horizontaux formés par le triangle station, pylônes G et D
    # seul alpha est connu (angle vu de la station)
    # l'angle alphaD (vu du pylône D) est recherché par dichotomie à partir de l'hypothèse initiale d'un triangle isocèle alphaD = alphaG = (np.pi - alpha)/2
    # les angles alpha1 et alpha2 sont connus et issus de la mesure comme alpha


    # conversion des grades en radians

    Alpha = (HD - HG) / 200 * np.pi
    Alpha1 = (H1 - HG) / 200 * np.pi
    Alpha2 = (H2 - HG) / 200 * np.pi
    VG = (100 - VG) / 200 * np.pi # angle nul = horizon (et non l'azimuth comme avec la station)
    VD = (100 - VD) / 200 * np.pi
    V1 = (100 - V1) / 200 * np.pi
    V2 = (100 - V2) / 200 * np.pi

    # initialisation des premières valeurs

    puissance = 1

    iteration = (np.pi - Alpha) / 2 ** puissance

    dif = 1000

    nb_loops = 100
    AlphaD = 0.

    for _ in range(nb_loops):

        AlphaD = AlphaD + iteration  # pour l'initialisation, cela revient à alphaD = (np.pi - alpha) / 2
        AlphaG = np.pi - Alpha - AlphaD # pour l'initialisation, cela revient à alphaG = (np.pi - alpha) / 2

        # premier calcul : les distances entre la station et les pylônes G et D

        distG = a / np.sin(Alpha) * np.sin(AlphaD)
        distD = distG * np.cos(Alpha) + a * np.cos(AlphaD)

        # deuxième calcul : les hauteurs des points d'accrochages et le dénivelé

        zG = distG * np.tan(VG)
        zD = distD * np.tan(VD)
        h = zD - zG


        # troisième calcul : les distances entre la station et les points 1 et 2 mais aussi a1, a2, z1, z2

        dist1 = distG / np.sin(np.pi - Alpha1 - AlphaG) * np.sin(AlphaG)
        dist2 = distG / np.sin(np.pi - Alpha2 - AlphaG) * np.sin(AlphaG)

        a1 = distG * np.cos(AlphaG) + dist1 * np.cos(np.pi - Alpha1 - AlphaG)
        a2 = distG * np.cos(AlphaG) + dist2 * np.cos(np.pi - Alpha2 - AlphaG)

        z1 = dist1 * np.tan(V1)
        z2 = dist2 * np.tan(V2)

        # appel à la fonction pour le calcul du paramètre à partir de la différence (zG - z1) et de a1
        # avec utilisation d'une valeur initiale p0 de p déjà proche de la solution

        p0 = a1 * (a - a1) / 2 / ((zG - z1) + h * a1 / a)   # estimation p0 du paramètre à partir de la flèche en utilisant l'approximation parabolique
        p = parameter_solver(a, h, zG - z1, a1, p0)                # affinage de la valeur du paramètre p à la fonction "parametre"
            
        # calcul d'une nouvelle différence à partir du paramètre calculé, comparaison avec (zG - z2) et bouclage avec nouvelle valeur pour iteration
        # val correspond à la valeur positive de la distance entre le point bas et le pylône de gauche (elle peut devenir négative avec un dénivelé fort et un câble très tendu)

        val = a / 2 - p * np.asinh(h / (2 * p * np.sinh(a / (2 * p))))
        dif = p * (np.cosh(val / p) - np.cosh((a2 - val) / p))
            
        puissance = puissance + 1
                
        iteration = np.sign(dif - (zG - z2)) * (np.pi - Alpha) / 2 ** puissance

        stop_variable = abs(dif - (zG - z2))
        stop_value = 0.001
        if (np.logical_or(stop_variable < stop_value, np.isnan(stop_variable))).all():
            break


    return p


def parameter_solver(a, h, delta, x, p0, solver: str = "newton",):
    solver_dict = {"newton": optimize.newton}
    try:
        solver_method = solver_dict[solver]
    except KeyError:
        raise ValueError(f"Incorrect solver name: {solver}")

    solver_result = solver_method(
        function_f,
        p0,
        fprime=function_f_prime,
        args=(a, h, delta, x),
        maxiter=10,
        tol=1e-5,
        full_output=True
    )
    if not solver_result.converged.all():
        raise ValueError("Solver did not converge")
    return solver_result.root


def function_f(p, a, h, delta, x):
    val = a / 2 - p * np.asinh(h / (2 * p * np.sinh(a / 2 / p)))
    f = p * (np.cosh(val / p) - np.cosh((x - val) / p)) - delta  
    return f

def function_f_prime(p, a, h, delta, x):
    # define f' with cosh
    return (function_f(p + _ZETA, a, h, delta, x) - function_f(p, a, h, delta, x)) / _ZETA


def parametre(a, h, delta, x, p0):

    # initialisation de la boucle

    p = p0                                                                      # avec la valeur initiale estimée proche de la solution
    pmem = p0 * (1 + 0.01)
    compteur = 0                                                                # le compteur n'est qu'une sécurité pour sortir de la boucle après 10 itérations

    while abs(pmem - p) > 0.001 and compteur < 10:                        # tant que pmem et p restent éloignés, il faut boucler

        val = a / 2 - p * np.asinh(h / (2 * p * np.sinh(a / 2 / p)))            # calcul de la valeur de la distance entre point bas et pylône G
        
        f = p * (np.cosh(val / p) - np.cosh((x - val) / p)) - delta             # calcul d'une fonction f qu'il faut annuler
        
        p = p + 1                                                               # signifie (p + epsillon) avec epsillon petit
        
        val = a / 2 - p * np.asinh(h / (2 * p * np.sinh(a / 2 / p)))
        
        fprim = p * (np.cosh(val / p) - np.cosh((x - val) / p)) - delta - f     # calcul de la valeur de la dérivée de la fonction en p par (f(p+1) - f(p)) / 1
        
        pmem = p - 1                                                            # mise en memoire de p
        
        p = (p - 1) - f / fprim                                                 # calcul de type Newton pour estimer le nouveau p

        compteur = compteur + 1

    return p

