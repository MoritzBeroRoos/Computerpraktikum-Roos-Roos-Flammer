from sympy.printing import pprint
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import makePlots
import Beispiele_stetig


from IPython.display import display

plt.style.use("default")
sp.init_printing()
x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)
z = sp.Symbol("z", real=True)


def unHeaviSideify(func, dic):
    """
    replaces Heaviside functions in func with values specified in dic and returns new function
    e.g. func = Heaviside(x)*Heaviside(x+1), dic = {x:1} -> return 1*Heaviside(x+1)

    :param func: function using sp.Heaviside
    :param dic: dictionary with entries of form: "value to be replaced : replacement"
    """
    for argument, replacement in dic.items():
        func = func.subs(sp.Heaviside(argument), replacement)
    return func


def get_h_fy_definition(fxy):
    """
    returns functions h
    and fy (density function of random variable Y)
    e.g. fxy = (x + y) * sp.Heaviside(x) * sp.Heaviside(-(x - 1)) * sp.Heaviside(y) * sp.Heaviside(-(y - 1))
    
    :param fxy: common density function of random variables X and Y
    """
    fy = sp.integrate(fxy, (x, -sp.oo, sp.oo)) # calculates fy from fxy
    h = sp.integrate(x * fxy / fy, (x, -sp.oo, sp.oo)) # calculates h using definition of conditional expectation
    return h, fy


def get_h_fy_MMSE(fxy):
    """
    returns functions h
    and fy (density function of random variable Y)
    calculating h using MMSE method
    e.g. fxy = (x + y) * sp.Heaviside(x) * sp.Heaviside(-(x - 1)) * sp.Heaviside(y) * sp.Heaviside(-(y - 1))
    :param fxy: common density function of random variables X and Y
    """
    h = sp.Function("h")
    L = sp.diff(sp.integrate((x - h(y)) ** 2 * fxy, (x, -sp.oo, sp.oo)), h(y))
    solutions = sp.solve(L, h(y))
    if len(solutions) == 0:
        raise ValueError("{} konnte nicht nach {} aufgelöst werden".format(L,h(y)))
    elif len(solutions)>1:
        raise ValueError("{} hatte mehr als eine Lösung beim auflösen nach {}".format(L,h(y)))
    fy = sp.integrate(fxy, (x, -sp.oo, sp.oo)) # calculates fy from fxy
    return solutions[0], fy


def get_h_fy_projection(fxy):
    """
    returns functions h
    and fy (density function of random variable Y)
    calculating h using projection method
    e.g. fxy = (x + y) * sp.Heaviside(x) * sp.Heaviside(-(x - 1)) * sp.Heaviside(y) * sp.Heaviside(-(y - 1))
    :param fxy: common density function of random variables X and Y
    """
    h = sp.Function("h")
    L = sp.integrate((x - h(y)) * y * fxy, (x, -sp.oo, sp.oo))
    solutions = sp.solve(L, h(y))
    if len(solutions) == 0:
        raise ValueError("{} konnte nicht nach {} aufgelöst werden".format(L,h(y)))
    elif len(solutions)>1:
        raise ValueError("{} hatte mehr als eine Lösung beim auflösen nach {}".format(L,h(y)))
    fy = sp.integrate(fxy, (x, -sp.oo, sp.oo)) # calculates fy from fxy
    return solutions[0], fy


def get_h_fy(fxy, mode):
    """
    returns h using the specified method 
    and density function fy of random variable Y

    :param fxy: common density function of random variables X and Y
    :param mode: string "def", "mmse" or "proj" that specifies the method for calculating h
    """
    if mode == "def":
        h, fy = get_h_fy_definition(fxy)
    elif mode == "mmse":
        h, fy = get_h_fy_MMSE(fxy)
    elif mode == "proj":
        h, fy = get_h_fy_projection(fxy)
    else:
        raise ValueError("Mode nicht zulässig angegeben")
    return h, fy


def get_h_plot(h, lower_boundary_plot_supp_Y, upper_boundary_plot_supp_Y):
    """
    returns plot for h in given interval (lower_boundary_plot_supp_Y, upper_boundary_plot_supp_Y) subset supp_Y

    :param h: function h
    :param lower_boundary_plot_supp_Y: lower limit on x-axis of the plot (dependent on supp_Y)
    :param upper_boundary_plot_supp_Y: upper limit on x-axis of the plot (dependent on supp_Y)
    """
    fig3 = plt.figure()
    x_werte = np.linspace(lower_boundary_plot_supp_Y, upper_boundary_plot_supp_Y, 1000)[1:-1]
    y_werte = [h.subs(y, sp.Rational(xi)).evalf() for xi in x_werte]
    plt.xlabel("y in supp_Y")
    plt.ylabel("h_opt(y)")
    plt.title("h_opt")
    plt.plot(x_werte, y_werte)
    return fig3


def try_plot_h_and_F_h_Y_z(h, fy, epsilon, lower_boundary_supp_Y, upper_boundary_supp_Y, lower_boundary_plot_supp_Y, upper_boundary_plot_supp_Y):
    """
    tries to plot the density-function of h(Y)=E(X|Y), or the cumulative distribution function.
    the result depends on sympys ability to calculate h^(-1) or inverse Image of h
    function values z of h will be colored with color determining probability of h(Y) in [z-epsilon, z + epsilon]

    :param h: function h (often oversimplified h works better than h containing Indicatorfunctions/heaviside-functions)
    :param fy: density function of random variable Y
    :param epsilon: defines how big the examined neighbourhood of h(y) should be
    :param lower_boundary_supp_Y: lower limit of support of Y , can also be set to -oo
    :param upper_boundary_supp_Y: upper limit of support of Y , can also be set to  oo
    :param lower_boundary_plot_supp_Y: lower limit on x-axis of the plot (dependent on supp_Y)
    :param upper_boundary_plot_supp_Y: upper limit on x-axis of the plot (dependent on supp_Y)
    """
    try:
        F_h_Y_z = makePlots.get_F_h_Y_z(sp.simplify(h), sp.simplify(fy), lower_boundary_supp_Y, upper_boundary_supp_Y)
        fig1, fig2 = makePlots.plot_h_colored_stetig_Symbolic(sp.simplify(h), lower_boundary_plot_supp_Y, upper_boundary_plot_supp_Y, epsilon, F_h_Y_z)

        return fig1, fig2
    except BaseException:
        print("could not calculate a formula for F_h_Y, trying with inverse Image: ")
        try:
            fig1, fig2 = makePlots.plot_h_colored_stetig_WithoutInverse(
                sp.simplify(h), sp.simplify(fy), lower_boundary_supp_Y, upper_boundary_supp_Y, lower_boundary_plot_supp_Y, upper_boundary_plot_supp_Y, epsilon
            )
            return fig1, fig2
        except BaseException:
            print("Couldn't do it with inverse Image")

