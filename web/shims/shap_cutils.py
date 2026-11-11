"""Pure-Python replacement for shap's compiled _cutils (copied over shap/_cutils.py by build.py)."""


def compute_exp_val(nsamples_run, nsamples_added, D, N, weights, y, ey):
    """Expected model output per sample, as in shap's C version (used by KernelExplainer)."""
    ey[nsamples_run:nsamples_added] = (
        y[nsamples_run * N:nsamples_added * N].reshape(-1, N, D) *
        weights[None, :, None]).sum(1)
    return nsamples_added


def compute_grey_code_row_values(*args, **kwargs):
    """Only used by ExactExplainer, which dianna does not use."""
    raise NotImplementedError(
        'shap ExactExplainer is not available in the browser')
