import sympy as sp
from sympy.printing import pprint
import matplotlib.pyplot as plt
import makePlots
import Beispiele_diskret
import numpy as np

from IPython.display import display

plt.style.use("default")
sp.init_printing()


x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)


def get_h_fy_definition(fxy, supp_X):
    """
    returns functions h (so that h(y) = E( X | Y = y) für y in supp_Y) 
    and fy (mass function of random variable Y),
    supposing the supports of random variables X and Y are sets of values 
    that the variables assume with probability >0

    :param fxy: common mass function of random variables X and Y
    :param supp_X: support of X
    """
    fy = sp.Add(*[fxy.subs(x, xi) for xi in supp_X]).doit()  # calculates fy from fxy
    h = sp.Add(*[(x * fxy / fy).subs(x, xi) for xi in supp_X]).doit()  # calculates h from definition of conditional expectation
    return (h, fy)


def get_punktweise_h_fy_MMSE(fxy, supp_X):
    """
    returns functions h (pointwise, so that h(y) = E( X | Y = y) für y in supp_Y) 
    and fy (mass function of random variable Y)
    calculating h using MMSE method

    :param fxy: common mass function of random variables X and Y
    :param supp_X: support of X
    """
    h = sp.Function("h")
    inner_integral = sp.Add(*[((x - h(y)) ** 2 * fxy).subs(x, xi) for xi in supp_X]).doit()
    L = sp.diff(inner_integral, h(y)).doit()
    def h_punktweise(z):
        return sp.solve(L.subs(y, z), h(z))[0]

    fy = sp.Add(*[fxy.subs(x, xi) for xi in supp_X])  # calculates fy from fxy
    return h_punktweise, fy


def get_h_punktweise_fy_projection(fxy, supp_X):
    """
    returns functions h (pointwise, so that h(y) = E( X | Y = y) für y in supp_Y) 
    and fy (mass function of random variable Y)
    calculating h using projection method

    :param fxy: common mass function of random variables X and Y
    :param supp_X: support of X
    """
    h = sp.Function("h")
    fy = sp.Add(*[fxy.subs(x, xi) for xi in supp_X]).doit()  # calculates fy from fxy
    inner_integral = sp.Add(*[((x - h(y)) * y * fxy).subs(x, xi) for xi in supp_X]).doit()

    def h_punktweise(z):
        return sp.solve(inner_integral.subs(y, z), h(z))[0]

    return (h_punktweise, fy)


def get_h_plot(fxy, supp_X, supp_Y, mode):
    """
    plots h (so that h(y) = E( X | Y = y) für y in supp_Y) using the specified method 

    :param fxy: common mass function of random variables X and Y
    :param supp_X: support of X
    :param supp_Y: support of Y
    :param mode: string "def", "mmse" or "proj" that specifies the method for calculating h
    """
    if mode == "def":
        h, fy = get_h_fy_definition(fxy, supp_X)
        funktionswerte_h = [h.subs(y, yi) for yi in supp_Y]
    elif mode == "mmse":
        h, fy = get_punktweise_h_fy_MMSE(fxy, supp_X)
        funktionswerte_h = [h(z) for z in supp_Y]
    elif mode == "proj":
        h, fy = get_h_punktweise_fy_projection(fxy, supp_X)
        funktionswerte_h = [h(z) for z in supp_Y]
    else:
        raise ValueError("Mode nicht zulässig angegeben")
    fig = makePlots.plot_h_colored_discrete(funktionswerte_h, fy, supp_Y)
    return fig,h,fy

