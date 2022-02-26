import sympy as sp
from sympy.printing import pprint
import matplotlib.pyplot as plt
import makePlots
import Beispiele_diskret_Infinite
from IPython.display import display

plt.style.use("default")
sp.init_printing()

# symbols:
x = sp.Symbol("x", real=True, integer=True, negative=False)
y = sp.Symbol("y", real=True, integer=True, negative=False)
h = sp.Function("h")


def get_h_fy_definition(fxy, fy, upper_bound = sp.oo):
    """
    returns functions h (so that h(y) = E( X | Y = y) für y in supp_Y),
    supposing supports of random variables X and Y are subsets of sp.Naturals0

    :param fxy: common mass function of random variables X and Y
    :param fy: mass function of random variable Y
    :param upper_bound: upper limit of support(X), can be set to a natural number
    """
    h = sp.Sum((x * fxy / fy), (x, 0, upper_bound)).simplify()  # calculates h from definition of conditional expectation
    return h


def get_L_MMSE(fxy, upper_bound=sp.oo):
    h = sp.Function("h")
    inner_integral = sp.Sum((x - h(y)) ** 2 * fxy, (x, 0, upper_bound))
    L = sp.diff(inner_integral, h(y)).doit()
    return L


def get_L_proj(fxy, upper_bound=sp.oo):
    h = sp.Function("h")
    L = sp.Sum((x - h(y)) * y * fxy, (x, 0, upper_bound))
    return L


def get_h_punktweise(L_manuell):
    def h_punktweise(z):
        L_manuell_subs = L_manuell.subs(y, z).doit()
        solutions = sp.solve(L_manuell_subs, h(z))
        if len(solutions) == 0:
            raise ValueError("{} konnte nicht nach h({}) aufgelöst werden".format(L_manuell_subs, z))
        if len(solutions) > 1:
            raise ValueError("{} besitzt mehrere Lösungen für h({})".format(L_manuell_subs, z))
        return solutions[0]

    return h_punktweise


def get_h_plot(fxy, upper_bound, mode):
    """
    param: upper_bound function in y, chosen so that fxy(x,y) = 0 for x > upper_bound,e.g. for sum Y := X1+X2 it holds that fxy(x,y) = 0 for x>y
    returns fig with h_plot and h or L


    """

    h = sp.Function("h")
    if mode == "def":
        fy = sp.Sum(fxy, (x, 0, upper_bound)).doit().refine(x<=upper_bound).simplify()
        h = get_h_fy_definition(fxy, fy,upper_bound).refine(x<=upper_bound).simplify()
        x_Achse = sp.Range(1, 25)
        h_funktionswerte = [h.subs(y, z) for z in x_Achse]
        ret = h
    elif mode == "mmse":
        L = get_L_MMSE(fxy, upper_bound).refine(x<=upper_bound).simplify()
        h_punktweise = get_h_punktweise(L)
        x_Achse = sp.Range(1, 25)
        h_funktionswerte = [h_punktweise(z) for z in x_Achse]
        ret = L
    elif mode == "proj":
        L = get_L_proj(fxy, upper_bound).refine(x<=upper_bound).simplify()
        h_punktweise = get_h_punktweise(L)
        x_Achse = sp.Range(1, 25)
        h_funktionswerte = [h_punktweise(z) for z in x_Achse]
        ret = L
    fig = plt.figure()
    plt.scatter(x_Achse, h_funktionswerte)
    plt.xlabel("y in supp Y")
    plt.ylabel("h_opt(y)")
    return fig,ret
