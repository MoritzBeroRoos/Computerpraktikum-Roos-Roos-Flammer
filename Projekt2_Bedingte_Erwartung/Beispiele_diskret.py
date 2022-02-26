import sympy as sp

x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)

# functions for generating density functions and supports of discrete random variables:


def discrete_equal_distrib(a,b):
    """
    returns mass function for discrete equally distributed random variable 
    with support [a,...,b]
    """
    mass_func = sp.Rational(1, b - a + 1)
    supp = [k for k in sp.Range(a, b+1)]
    return mass_func, supp


def binomial_distrib(n, p):
    """
    returns mass function (in Symbol x) for binomially distributed random variable 
    with support [0,1,...,n]

    :param n: cardinality of support
    :param p: probability of success
    """
    mass_func = sp.factorial(n) / (sp.factorial(n - x) * sp.factorial(x)) * p ** x * (1 - p) ** (n - x)
    supp = [a for a in sp.Range(0, n + 1)]
    return mass_func, supp


def hypergeometric(N, M, n):
    """
    returns mass function (in Symbol x) for hypergeometric distributed random variable 
    with support [max(0, n+M-N),...,min(n,M)]

    :param N: number of elements in total
    :param M: number of elements with desired property
    :param n: number of experiment realisations
    """
    mass_func = sp.binomial(M, x) * sp.binomial(N - M, n - x) / sp.binomial(N, n)
    supp = [a for a in sp.Range(max(0, n + M - N), min(n, M) + 1)]
    return mass_func, supp


def get_common_density_func_and_supp(fx1, fx2, supp_X1, supp_X2, operation):
    """
    operation can be "X1+X2","X2+X1", "X1-X2", "X2-X1", "X1*X2", "X2*X1", "X1/X2", "X2/X1
    returns common density of (X1, X1 _operation_ X2)
    """
    if operation == "X1+X2" or operation == "X2+X1":
        supp_Y_sum = [a for a in sp.Range(min(supp_X1) + min(supp_X2), max(supp_X1) + max(supp_X2) + 1)]
        fx1y_sum = sp.Piecewise((fx1 * fx2.subs(x, y - x), sp.FiniteSet(*supp_X1).contains(x) & sp.FiniteSet(*[x + k for k in supp_X2]).contains(y)), (sp.Integer(0), True))
        return fx1y_sum, supp_Y_sum
    elif operation == "X1-X2":
        supp_Y_minus = [a for a in sp.Range(min(supp_X1) - max(supp_X2), max(supp_X1) - min(supp_X2) + 1)]
        fx1y_minus = sp.Piecewise((fx1 * fx2.subs(x, x - y), sp.FiniteSet(*supp_X1).contains(x) & sp.FiniteSet(*[x - k for k in supp_X2]).contains(y)), (sp.Integer(0), True))
        return fx1y_minus, supp_Y_minus
    elif operation == "X2-X1":
        supp_Y_minus = [a for a in sp.Range(min(supp_X2) - max(supp_X1), max(supp_X2) - min(supp_X1) + 1)]
        fx1y_minus = sp.Piecewise((fx1 * fx2.subs(x, y+x), sp.FiniteSet(*supp_X1).contains(x) & sp.FiniteSet(*[k-x for k in supp_X2]).contains(y)), (sp.Integer(0), True))
        return fx1y_minus,supp_Y_minus
    elif operation == "X1*X2" or operation == "X2*X1":
        supp_Y_mult = list(set([x1 * x2 for x1 in supp_X1 for x2 in supp_X2]))#entferne doppelte Elemente
        # fx1y_mult = sp.Piecewise((fx1 * fx2.subs(x, y / x), sp.FiniteSet(*supp_X1).contains(x) & sp.FiniteSet(*[x * k for k in supp_X2]).contains(y)), (sp.Integer(0), True))
        fx1y_mult = sp.Piecewise(
            (fx1 * fx2.subs(x, y / x), (sp.FiniteSet(*[a for a in supp_X1 if a != sp.Integer(0)]).contains(x) & sp.FiniteSet(*[x * k for k in supp_X2]).contains(y))),
            (fx1.subs(x, 0), (x == sp.Integer(0)) & (y == sp.Integer(0)) & (sp.FiniteSet(*supp_X1).contains(sp.Integer(0)))),
            (sp.Integer(0), True),
        )
        return fx1y_mult, supp_Y_mult
    elif operation == "X1/X2":
        #X2 darf nie 0 sein, bzw 0 notin supp_X2
        if sp.Integer(0) in supp_X2:
            raise ValueError("0 ist in supp_X2 enthalten, damit ist X1/X2 nicht sinnvoll definiert")
        supp_Y_div = list(set([x1 / x2 for x1 in supp_X1 for x2 in supp_X2]))#entferne doppelte Elemente
        #fx1y_div = sp.Piecewise((fx1 * fx2.subs(x, x / y), sp.FiniteSet(*supp_X1).contains(x) & sp.FiniteSet(*[x - k for k in supp_X2]).contains(y)), (sp.Integer(0), True))
        fx1y_div = sp.Piecewise(
            (fx1 * fx2.subs(x, x / y), sp.FiniteSet(*supp_X1).contains(x) & sp.FiniteSet(*[x / k for k in supp_X2]).contains(y)),
            (fx1.subs(x, 0), (y == sp.Integer(0)) & (sp.FiniteSet(*supp_X1).contains(sp.Integer(0)))),
            (sp.Integer(0), True)
        )
        return fx1y_div,supp_Y_div
    elif operation == "X2/X1":
        #X1 darf nie 0 sein, bzw 0 notin supp_X1
        if sp.Integer(0) in supp_X1:
            raise ValueError("0 ist in supp_X1 enthalten, damit ist X2/X1 nicht sinnvoll definiert")
        supp_Y_div = list(set([x2/x1 for x2 in supp_X2 for x1 in supp_X1]))#entferne doppelte Elemente
        fx1y_div = sp.Piecewise(
            (fx1*fx2.subs(x,y*x), sp.FiniteSet(*supp_X1).contains(x) & (sp.FiniteSet(*[k/x for k in supp_X2]).contains(y))),
            (0,True)
        )
        return fx1y_div,supp_Y_div
    else:
        raise ValueError("Specified Operation '{}' was invalid".format(operation))


