"""Browser stand-in for numba (no Pyodide build): shap's jit decorators become no-ops."""


def njit(*args, **kwargs):
    """Return the function unchanged, with or without decorator arguments."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda f: f


jit = njit
prange = range
