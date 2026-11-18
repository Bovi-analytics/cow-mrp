"""Shared fixtures. The cow and youngstock models are built once per session (month resolution, the default)."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)          # tests/machine_domain (the toy domain) only; the package itself is imported as installed (pip install -e .)
os.environ.setdefault("COW_TIME_UNIT", "month")


def pytest_configure(config):
    for m in ("slow: long-running", "install: builds a wheel and installs it in a fresh venv (needs network for dependencies)",
              "performance: timing / memory budgets"):
        config.addinivalue_line("markers", m)


@pytest.fixture(scope="session")
def golden():
    with open(os.path.join(HERE, "baselines", "golden_month.json"), encoding="utf8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def cow():
    from cowmrp.domain import load_domain
    from cowmrp.model import build_model
    dom = load_domain("cow")
    return dom, build_model(dom)


@pytest.fixture(scope="session")
def youngstock():
    from cowmrp.domain import load_domain
    from cowmrp.model import build_model
    dom = load_domain("youngstock")
    return dom, build_model(dom)


@pytest.fixture(scope="session")
def machine():
    from cowmrp.domain import load_domain
    from cowmrp.model import build_model
    dom = load_domain("machine_domain")
    return dom, build_model(dom)
