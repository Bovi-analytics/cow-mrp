"""Documentation synchronization (guide/sync.py): facts come from the package, stale documents are detected, a rebuilt site is current."""
import os
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, "guide"))
import sync  # noqa: E402


def test_facts_match_the_built_model(cow):
    _, m = cow
    f = sync.facts()
    assert f["COW_STATES"] == f"{len(m.states):,}" and f["COW_NNZ"] == f"{m.P.nnz:,}" and f["CO_STATES"] == f["COW_STATES"]


def test_generated_blocks_are_current_and_a_stale_block_is_detected():
    text = open(os.path.join(ROOT, "README.md"), encoding="utf8", newline="").read()
    assert sync.render_blocks(text) == text                                   # `python guide/sync.py write` leaves nothing to do
    stale = text.replace(f"{sync.facts()['COW_STATES']} states", "1 states", 1)
    assert sync.render_blocks(stale) == text and stale != text                # a drifted number is rewritten from the package


def test_tokens_expand_and_unknown_tokens_are_left_for_the_builders():
    assert sync.fill_tokens("n={{COW_STATES}} {{CODE}}") == f"n={sync.facts()['COW_STATES']} {{{{CODE}}}}"


def test_hand_typed_model_numbers_are_reported(monkeypatch):
    real = sync._read
    monkeypatch.setattr(sync, "_read", lambda rel: real(rel) + f"\nthe model has {sync.facts()['COW_STATES']} states" if rel == "runners/README.md" else real(rel))
    problems = []
    sync.check_hardcoded(problems)
    assert any("runners/README.md" in p and "typed by hand" in p for p in problems)


def test_unknown_token_in_a_content_page_is_reported(monkeypatch):
    real = sync._read
    monkeypatch.setattr(sync, "_read", lambda rel: real(rel) + " {{NOT_A_FACT}}" if rel == "guide/web/content/home.html" else real(rel))
    problems = []
    sync.check_tokens(problems)
    assert any("NOT_A_FACT" in p for p in problems)


def test_source_change_makes_a_site_stale(monkeypatch, tmp_path):
    site = tmp_path / "site"
    sync.write_stamp(str(site))
    monkeypatch.setitem(sync.SITES, "dummy", "none")
    monkeypatch.setattr(sync, "ROOT", str(tmp_path))                          # a fake root holding only the dummy site
    (tmp_path / "dummy").mkdir()
    (tmp_path / "dummy" / sync.STAMP).write_text('{"sources_sha256": "0"}')
    problems = []
    monkeypatch.setattr(sync, "SITES", {"dummy": "none"})
    sync.check_sites(problems)
    assert any("older sources" in p for p in problems)


def test_impact_lists_dependents(capsys):
    sync.impact("src/cowmrp/domains/cow/rewards/base.py")
    out = capsys.readouterr().out
    assert "reward_table" in out and "baselines" in out


def test_examples_page_lists_every_example_notebook():
    import glob
    if not os.path.isdir(os.path.join(ROOT, "user_examples")):
        pytest.skip("user_examples/ is obtained from GitHub; it is not part of the source distribution")
    page =open(os.path.join(ROOT, "guide", "web", "content", "notebooks.html"), encoding="utf8").read()
    notebooks = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "user_examples", "[0-9][0-9]_*.ipynb")))
    assert len(notebooks) == 5
    assert "{{NOTEBOOK_ROWS}}" in page                      # the notebook list is generated from user_examples/README.md at build time
    readme = open(os.path.join(ROOT, "user_examples", "README.md"), encoding="utf8").read()
    assert [n for n in notebooks if n not in readme] == []   # every notebook file is in the README table that feeds the page
    assert os.path.exists(os.path.join(ROOT, "user_examples", "custom_domain", "insemination", "states.py"))
