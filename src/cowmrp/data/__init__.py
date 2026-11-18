"""Example data shipped with cowmrp: a small synthetic herd for the tutorials (no real farm records).

``example_herd_path()``      the 200-cow herd in the model's own schema (cow_id, state_id, lactation, uim, up, dim_days, dp_days, status),
                             made with the package's own generator (``generate_herd``, equilibrium mode, seed 42, month resolution)
``example_raw_herd_path()``  the same cows as a farmer would export them: cow_id, lactation, dim_days, dp_days only
Both are synthetic: cows drawn from the model's steady-state herd, MIT like the package.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def example_herd_path():
    return os.path.join(_HERE, "example_herd.csv")


def example_raw_herd_path():
    return os.path.join(_HERE, "example_herd_raw.csv")
