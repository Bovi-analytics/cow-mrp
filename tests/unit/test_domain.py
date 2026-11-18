"""The domain interface: loading by name or dotted path, validation, hooks and their defaults."""
import pytest

from cowmrp.domain import Domain, load_domain, validate_domain


def test_shipped_domains_load_and_validate(cow, youngstock):
    for dom, _ in (cow, youngstock):
        assert isinstance(dom, Domain) and validate_domain(dom) == []
    assert cow[0].name == "cow" and cow[0].has_rewards and cow[0].time_unit == "month" and cow[0].unit_days == 30
    assert not youngstock[0].has_rewards


def test_hooks_and_defaults(cow, youngstock):
    c, y = cow[0], youngstock[0]
    assert c.strict is True and c.export_column_names == {"dim": "uim", "dp": "up"}
    assert y.export_column_names == {"dp": "up"} and y.hook("replacement_state_id") is None
    assert y.hook("INITIAL_STATE", "fallback") == "fallback"


def test_custom_domain_by_dotted_path(machine):
    dom, model = machine
    assert dom.name == "machine_domain" and validate_domain(dom) == [] and len(model.states) == 3


def test_load_domain_is_idempotent(cow):
    assert load_domain(cow[0]) is cow[0]


def test_unknown_domain_raises():
    with pytest.raises(ImportError, match="not found"):
        load_domain("no_such_domain_xyz")


def test_incomplete_domain_reports_what_is_missing(tmp_path, monkeypatch):
    pkg = tmp_path / "half_domain"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "parameters.py").write_text("TIME_UNIT = 'month'\nUNIT_DAYS = 30\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ImportError, match="states, probabilities, transitions"):
        load_domain("half_domain")


def test_validate_domain_flags_missing_functions_and_bad_parameters(tmp_path, monkeypatch):
    pkg = tmp_path / "bad_domain"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "parameters.py").write_text("TIME_UNIT = 'fortnight'\nUNIT_DAYS = 14.5\n")
    for f in ("states", "probabilities", "transitions"):
        (pkg / f"{f}.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    text = " ".join(validate_domain("bad_domain"))
    assert "build_bad_domain_states" in text and "compute_probabilities" in text and "compute_transitions" in text
    assert "UNIT_DAYS" in text and "TIME_UNIT" in text
