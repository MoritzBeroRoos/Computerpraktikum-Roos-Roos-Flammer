import sympy as sp

x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)

def continuous_equal_distrib(a,b):
    """
    """
    density_func = sp.Rational(1, b-a)*sp.Heaviside(x-a)*sp.Heaviside(b-x)
    return density_func, a,b


def normal_distrib(mu, sigma):
    """
    """
    density_func = 1/(sigma*sp.sqrt(2*sp.pi))*sp.exp(-sp.Rational(1,2)*((x-mu)/sigma)**2)
    return density_func, -sp.oo, sp.oo


def logarithmic_distrib(mu, sigma):
    """
    """
    density_func = 1/(sigma*sp.sqrt(2*sp.pi))*(1/x)*sp.exp(-sp.Rational(1,2)*((sp.ln(x)-mu)/sigma)**2)*sp.Heaviside(x)
    return density_func, 0, sp.oo


def exponential_distrib(alpha):
    assert alpha > 0
    density_func = alpha * sp.exp(-alpha*x)*sp.Heaviside(x)
    return density_func, 0, sp.oo


def common_density_func(fx1,fx2, operation):
    """
    returns common density of (X1,X1_operationX2)
    """
    if operation == "X1+X2" or operation == "X2+X1":
        return fx1*fx2.subs(x, y-x).simplify() 
    elif operation == "X1-X2":
        return fx1*fx2.subs(x, x-y).simplify()
    elif operation == "X1*X2_H" or operation == "X2*X1_H":
        return fx1*fx2.subs(x, y/x)*sp.Heaviside(x,0) + fx1*fx2.subs(x,y/x)*sp.Heaviside(-x, 0).simplify()
    elif operation == "X1*X2" or operation == "X2*X1":
        return fx1*fx2.subs(x, y/x).simplify()
    elif operation == "X1/X2_H":
        return fx1*fx2.subs(x, x/y)*sp.Heaviside(y,0) + fx1*fx2.subs(x,x/y)*sp.Heaviside(-y, 0).simplify()
    elif operation == "X1/X2":
        return fx1*fx2.subs(x, x/y).simplify()
    elif operation == "X2/X1":
        return fx1*fx2.subs(x, y*x).simplify()
    else:
        raise ValueError(" '{}' is not a valid operation".format(operation))

