"""Share fixed material quadrature while independently compiling shorter routes."""
from copy import copy
from .ehss_factored_local_samples import MaterialSamples

def material_bank_with_cap(bank,cap):
    """Keep every sampled cell/weight; require a fresh bound and cache scope.

    This changes local and missing-target route budgets, not the training
    ensemble or its occupation weights. A bound from the parent bank cannot
    be passed into this bank's bind method because bank identity differs.
    """
    if not isinstance(bank,MaterialSamples): raise TypeError('MaterialSamples required')
    if type(cap) is not int or not 2<=cap<=bank.cap:
        raise ValueError('Compilation cap must be an integer from 2 to the existing cap')
    result=copy(bank); result.cap=cap
    return result
