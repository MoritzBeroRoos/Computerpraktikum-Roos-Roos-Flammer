import sympy as sp
from sympy.printing import pprint
import matplotlib.pyplot as plt
import numpy as np

x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)
z = sp.Symbol("z", real=True)
u = sp.Symbol("u", real=True)


def get_wahrscheinlichkeiten_für_h_discrete(
    funktionswerte_h, fy, supp_Y,
):
    def F_h_of_Y(u):
        # requires funktionswerte_h = [h.subs(y, yi) for yi in supp_Y] in der richtigen reihenfolge
        # dann äquivalent zu Add(*[(fy.subs(y,yi)) for yi in [k for k in supp_Y if (h.subs(y,k) <= u) ] ])
        return sp.Add(*[(fy.subs(y, yi)) for yi in [k for i, k in enumerate(supp_Y) if (funktionswerte_h[i] <= u)]]).doit()

    distances_funktionswerte_h = [sp.Integer(0)]#garantiert mindestens 1 Element in der Liste
    for i in range(0, len(funktionswerte_h)):
        for j in range(0, i):
            distances_funktionswerte_h.append(sp.Abs(funktionswerte_h[i] - funktionswerte_h[j]))
    sorted_distances=sorted(distances_funktionswerte_h)
    min_notzero_index=0
    #schließe aus dass minimale Distanz 0 ist
    for i,element in enumerate(sorted_distances):
        if element != sp.Integer(0):
            min_notzero_index=i
            break
    min_distances_of_funktionswerte = sorted_distances[min_notzero_index]
    if min_distances_of_funktionswerte == sp.Integer(0):
        #falls dies eintritt sind alle Funktionswerte von h gleich also treten diese zu 100 % auf
        return [1 for u in funktionswerte_h]
    wahrscheinlichkeiten_für_h = [F_h_of_Y(u) - F_h_of_Y(u - min_distances_of_funktionswerte / 2) for u in funktionswerte_h]
    return wahrscheinlichkeiten_für_h


def get_normed_list(wahrscheinlichkeiten_für_h):
    m = max(wahrscheinlichkeiten_für_h)
    return [300 / m * x + 1 for x in wahrscheinlichkeiten_für_h]


def plot_h_colored_discrete(funktionswerte_h, fy, supp_Y):
    wahrscheinlichkeiten_für_h = get_wahrscheinlichkeiten_für_h_discrete(funktionswerte_h, fy, supp_Y)
    wahrscheinlichkeiten_für_h_floats = [float(x) for x in get_normed_list(wahrscheinlichkeiten_für_h)]
    fig = plt.figure()
    sc = plt.scatter(supp_Y, funktionswerte_h,s=wahrscheinlichkeiten_für_h_floats, c=wahrscheinlichkeiten_für_h, cmap="plasma")
    plt.title("Farbe von (y,E(X|Y=y)) anhand Größe von P(E(X|Y) = h_opt(y))")
    plt.xlabel("supp_Y")
    plt.ylabel("E(X|Y=y)=h_opt(y)")
    plt.colorbar(sc)
    return fig


def get_F_h_Y_z(h, fy, lower_boundary, upper_boundary):
    """
    requires h to be stricly monotone
    """
    inverse_u_candidates = sp.solve(h - u, y)
    assert len(inverse_u_candidates) == 1
    inverse_u = inverse_u_candidates[0]
    if sp.is_strictly_decreasing(h, sp.Interval.open(lower_boundary, upper_boundary)):
        F_hY_z = sp.integrate(fy, (y, sp.Min(sp.Max(inverse_u.subs(u, z), lower_boundary), upper_boundary), upper_boundary)).doit().simplify()
        return F_hY_z
    elif sp.is_strictly_increasing(h, sp.Interval.open(lower_boundary, upper_boundary)):
        F_hY_z = sp.integrate(fy, (y, lower_boundary, sp.Max(lower_boundary, sp.Min(inverse_u.subs(u, z), upper_boundary)))).doit().simplify()
        return F_hY_z
    else:
        raise ValueError("h war nicht streng monoton")


def get_F_h_Y_punktweise(z, h, fy, lower_boundary, upper_boundary):
    urbildmenge = sp.solveset(h <= z, y, sp.S.Reals)
    if isinstance(urbildmenge,sp.sets.sets.Union):
        ergebnis = sp.Integer(0)
        for interval in urbildmenge.args:
            int_lower_bound = sp.Max(interval.inf, lower_boundary)
            int_upper_bound = sp.Min(upper_boundary, interval.sup)
            if int_upper_bound >= int_lower_bound:
                ergebnis += sp.integrate(fy, (y, int_lower_bound, int_upper_bound))
        return ergebnis
    else:
        int_lower_bound = sp.Max(urbildmenge.inf, lower_boundary)
        int_upper_bound = sp.Min(upper_boundary, urbildmenge.sup)
        return sp.integrate(fy, (y, int_lower_bound, int_upper_bound))


def get_probability_of_neighbourhood_of_a(a, epsilon, h, fy, lower_boundary, upper_boundary):
    return get_F_h_Y_punktweise(a + epsilon, h, fy, lower_boundary, upper_boundary) - get_F_h_Y_punktweise(a - epsilon, h, fy, lower_boundary, upper_boundary)


def plot_h_colored_stetig_WithoutInverse(h, fy, lower_boundary, upper_boundary, lower_boundary_plot, upper_boundary_plot, epsilon):
    stützstellen = np.linspace(lower_boundary_plot, upper_boundary_plot, 200)[1:-1]  # x-Werte für h-plot
    h_values = [h.subs(y, z) for z in stützstellen]  # Werte für h-plot
    h_values_min = min(h_values)  # Wertebereich von Bild(h)
    h_values_max = max(h_values)

    x_werte = np.linspace(lower_boundary_plot, upper_boundary_plot, 100)[1:-1]
    stützstellen_und_h_values = [(z, h.subs(y, z)) for z in x_werte]
    stützstellen_und_h_values_für_h_Y_wahrscheinlichkeiten = [tup for tup in stützstellen_und_h_values if ((sp.Abs(tup[1] - h_values_min) > epsilon) & (sp.Abs(tup[1] - h_values_max) > epsilon))]
    stützstellen_für_h_Y_wahrscheinlichkeiten = [tup[0] for tup in stützstellen_und_h_values_für_h_Y_wahrscheinlichkeiten]
    h_values_für_h_Y_wahrscheinlichkeiten = [tup[1] for tup in stützstellen_und_h_values_für_h_Y_wahrscheinlichkeiten]
    h_wahrscheinlichkeiten = [get_probability_of_neighbourhood_of_a(h_yi, epsilon, h, fy, lower_boundary, upper_boundary) for h_yi in h_values_für_h_Y_wahrscheinlichkeiten]

    fig1 = get_h_figure_stetig(stützstellen, h_values, stützstellen_für_h_Y_wahrscheinlichkeiten, h_values_für_h_Y_wahrscheinlichkeiten, h_wahrscheinlichkeiten)
    fig1.title("h_opt mit Färbung von Punkt (y,h_opt(y)) anhand Größe von P(E(X|Y) in (h_opt - epsilon, h_opt + epsilon])")
    fig1.xlabel("y in supp_Y")
    fig1.ylabel("h_opt(y)")
    fig2 = get_distr_figure_without_inverse(h, fy, lower_boundary, upper_boundary, h_values_min, h_values_max)
    return fig1, fig2


def get_distr_figure_without_inverse(h, fy, lower_boundary, upper_boundary, h_values_min, h_values_max):
    x_Achse = np.linspace(float(h_values_min), float(h_values_max), 100)[1:-1]  # x-Werte für Verteilungsfunktion/Dichte von h(Y)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title("Verteilungsfunktion von h(Y)")
    ax2.set_xlabel("Bildwerte y von h(Y)")
    ax2.set_ylabel("P(h(Y) <= y)")
    Verteilungsfunktionswerte = [get_F_h_Y_punktweise(h_yi, h, fy, lower_boundary, upper_boundary) for h_yi in x_Achse]
    ax2.plot(x_Achse, Verteilungsfunktionswerte)
    return fig2


def plot_h_colored_stetig_Symbolic(h, lower_boundary_plot, upper_boundary_plot, epsilon, F_h_Y_z):
    stützstellen = np.linspace(lower_boundary_plot, upper_boundary_plot, 2000)[1:-1]
    h_values = [h.subs(y, z) for z in stützstellen]
    h_values_min = min(h_values)
    h_values_max = max(h_values)
    x_werte = np.linspace(lower_boundary_plot, upper_boundary_plot, 1000)
    stützstellen_und_h_values = [(z, h.subs(y, z)) for z in x_werte]
    stützstellen_und_h_values_für_h_Y_wahrscheinlichkeiten = [tup for tup in stützstellen_und_h_values if ((sp.Abs(tup[1] - h_values_min) > epsilon) & (sp.Abs(tup[1] - h_values_max) > epsilon))]
    stützstellen_für_h_Y_wahrscheinlichkeiten = [tup[0] for tup in stützstellen_und_h_values_für_h_Y_wahrscheinlichkeiten]
    h_values_für_h_Y_wahrscheinlichkeiten = [tup[1] for tup in stützstellen_und_h_values_für_h_Y_wahrscheinlichkeiten]
    h_wahrscheinlichkeiten_symbol = [F_h_Y_z.subs(z, h_yi + epsilon) - F_h_Y_z.subs(z, h_yi - epsilon) for h_yi in h_values_für_h_Y_wahrscheinlichkeiten]
    
    fig1 = get_h_figure_stetig(stützstellen, h_values, stützstellen_für_h_Y_wahrscheinlichkeiten, h_values_für_h_Y_wahrscheinlichkeiten, h_wahrscheinlichkeiten_symbol)
    fig2 = get_distr_figure_symbolic(F_h_Y_z, h_values_min, h_values_max)
    return fig1, fig2


def get_distr_figure_symbolic(F_h_Y_z, h_values_min, h_values_max):
    x_Achse = np.linspace(float(h_values_min), float(h_values_max), 100)[1:-1]
    f_h_Y_z = sp.diff(F_h_Y_z, z)
    dichtewerte_symbol = [f_h_Y_z.subs(z, h_yi) for h_yi in x_Achse]

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title("Dichte")
    ax2.set_xlabel("Bildwerte y von h(Y)")
    ax2.set_ylabel("")
    ax2.plot(x_Achse, dichtewerte_symbol)
    return fig2


def get_h_figure_stetig(stützstellen, h_values, stützstellen_für_h_Y_wahrscheinlichkeiten, h_values_für_h_Y_wahrscheinlichkeiten, h_wahrscheinlichkeiten):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(stützstellen, h_values)
    ax.set_title("h-plot")
    ax.set_xlabel("y in supp(Y)")
    ax.set_ylabel("h_opt(y)")
    radiusList_floats = [float(a) for a in get_normed_list(h_wahrscheinlichkeiten)]
    sc = ax.scatter(stützstellen_für_h_Y_wahrscheinlichkeiten, h_values_für_h_Y_wahrscheinlichkeiten,s =radiusList_floats , c=h_wahrscheinlichkeiten, cmap="plasma", zorder=10)
    plt.colorbar(sc)
    return fig

