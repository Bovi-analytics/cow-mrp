"""cowmrp: Markov reward process (MRP) models of dairy cattle.

Layout:
    cowmrp.mrp      the domain-free MRP engine (core math, spec orchestration, simulator)
    cowmrp.domains  one sub-package per domain (cow, youngstock; add your own)
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("cowmrp")
except PackageNotFoundError:   # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
