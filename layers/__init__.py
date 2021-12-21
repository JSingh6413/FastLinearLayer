from .Linear import Linear


def CustomLinear(k, method):
    return lambda *params: Linear(*params, k=k, method=method)