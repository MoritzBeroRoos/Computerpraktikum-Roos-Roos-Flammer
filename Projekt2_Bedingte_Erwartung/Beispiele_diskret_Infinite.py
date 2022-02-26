import sympy as sp

x = sp.Symbol("x", real=True, integer=True, negative=False)
y = sp.Symbol("y", real=True, integer=True, negative=False)

# functions for generating density functions and supports of discrete random variables:
def geometric_distrib(p):
    """
    returns mass function (in Symbol x) for geometric distributed random variable 
    with support sp.Naturals

    :param p: probability of success, 0<p<=1
    """
    mass_func = p * (1 - p) ** (x - 1) * sp.Piecewise((0, x == sp.Integer(0)), (1, True))
    return mass_func


def poisson_distrib(lmbda):
    """
    returns mass function (in Symbol x) for poisson distributed random variable 
    with support sp.Naturals0

    :param lmbda: average event rate > 0
    """
    mass_func = lmbda ** x * sp.exp(-lmbda) / sp.factorial(x)
    return mass_func


def logarithmic_distrib(p):
    """
    returns mass function (in Symbol x) for logarithmic distributed random variable 
    with support sp.Naturals

    :param p: probability of success, 0<p<1
    """
    mass_func = sp.Piecewise((0, x<=0), (p ** x / x * 1 / (-sp.ln(1 - p)), True))
    return mass_func


def common_density_func_sum(fx1, fx2):
    """
    returns common density of x1 and x1+x2 and upper_bound for get_h_plot
    """
    return sp.Piecewise((fx1 * fx2.subs(x, y - x), (y >= x)), (0, True)).simplify(), y


def common_density_func_mult(fx1, fx2):
    """
    returns common density of x1 and x1+x2 and upper_bound for get_h_plot, requires 0 notin supp X2
    """
    return sp.Piecewise((fx1 * fx2.subs(x, y / x), (x>0)),(fx1.subs(x,0),(x==0) & (y == 0)), (0, True)).simplify(), y

def pascal_distrib(p, r):
    """
    returns mass function (in Symbol x) for Pascal distributed random variable 
    with support [r,...] 

    :param p: probability of success
    :param r: number of successes expected
    """
    mass_func = sp.Piecewise((sp.factorial(x - 1) / (sp.factorial(x - r) * sp.factorial(r - 1)) * p ** r * (1 - p) ** (x - r), x >= r), (0, True))
    return mass_func


# Verteilungen für endliche, eingebettet in N_0:


def get_discrete_equal_distrib(n):
    """
    returns mass function for discrete equally distributed random variable 
    with support [1,...,n]
    
    :param n: cardinality of support
    """
    mass_func = sp.Piecewise((sp.Rational(1, n), (1 <= x) & (x <= n)), (0, True))
    return mass_func


def binomial_distrib(n, p):
    """
    returns mass function (in Symbol x) for binomially distributed random variable 
    with support [0,1,...,n]

    :param n: cardinality of support
    :param p: probability of success
    """
    mass_func = sp.Piecewise((sp.factorial(n) / (sp.factorial(n - x) * sp.factorial(x)) * p ** x * (1 - p) ** (n - x), (x <= n)), (0, True))
    return mass_func


def hypergeometric(N, M, n):
    """
    returns mass function (in Symbol x) for hypergeometric distributed random variable 
    with support [max(0, n+M-N),...,min(n,M)]

    :param N: number of elements in total
    :param M: number of elements with desired property
    :param n: number of experiment realisations
    """
    mass_func = sp.Piecewise(sp.binomial(M, x) * sp.binomial(N - M, n - x) / sp.binomial(N, n), ((sp.Max(0, n - M - N) <= x) & x <= sp.Min(n, M)), (0, True))
    return mass_func

