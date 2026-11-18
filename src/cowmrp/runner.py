"""cowmrp.runner -- DEPRECATED compatibility alias. Use ``cowmrp.domain`` (the interface) and ``cowmrp.model`` (build / solve)."""
import warnings

from cowmrp.domain import Domain, load_domain, validate_domain                                          # noqa: F401
from cowmrp.model import (Model, build_model, per_step_gamma, evaluate, initial_distribution, evolve_distribution,   # noqa: F401
                          replacement_split, simulate, monte_carlo, group_summary, matrix_checksum, row_sum_report)

warnings.warn("cowmrp.runner is deprecated: import from cowmrp.domain and cowmrp.model", DeprecationWarning, stacklevel=2)
