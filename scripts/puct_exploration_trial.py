"""Inference helpers from puct_exploration_trial.py."""

ABI='public-puct-exploration-v1'


COEFFICIENTS=(.5,1.,2.)


def coefficient_config(c):
    if type(c) not in (int,float) or c not in COEFFICIENTS:raise ValueError('undeclared exploration coefficient')
    return dict(version=ABI,coefficient=c)
