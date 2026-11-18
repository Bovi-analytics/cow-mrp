"""audit_registry.py -- print the method registry (cow/registry.py) and fail if it is out of step with the specs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import pandas as pd
from cowmrp.domains.cow.registry import check_registry, describe
check_registry()
pd.set_option("display.width", 250, "display.max_colwidth", 110, "display.max_rows", 100)
d = describe()
print(d[["spec", "method", "active", "available", "evidence"]].to_string(index=False))
print("\nREGISTRY_OK: every probability spec is recorded and its method availability matches")
