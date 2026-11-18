"""The current licence of the project is MIT, stated the same way in every current metadata file."""
import json
import os
import re

from conftest import ROOT


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf8") as fh:
        return fh.read()


def test_license_file_is_mit_with_one_copyright_line():
    text = _read("LICENSE")
    assert text.startswith("MIT License")
    assert "Permission is hereby granted, free of charge" in text
    assert not re.search(r"GNU|General Public License|Affero", text)
    assert re.search(r"^Copyright \(c\) \d{4} \S", text, flags=re.M)


def test_metadata_and_readme_agree_on_mit_and_the_copyright_line():
    assert re.search(r'^license = "MIT"', _read("pyproject.toml"), flags=re.M)
    assert re.search(r"^license: MIT$", _read("CITATION.cff"), flags=re.M)
    assert json.loads(_read(".zenodo.json"))["license"].lower() == "mit"
    line = re.search(r"^(Copyright \(c\) \d{4} .+)$", _read("LICENSE"), flags=re.M).group(1)
    readme = _read("README.md")
    assert "MIT licence" in readme and line in readme


def test_no_current_project_file_names_another_licence_for_cowmrp():
    for rel in ("pyproject.toml", "CITATION.cff", ".zenodo.json", "README.md"):
        text = _read(rel)
        assert not re.search(r"\b(AGPL|GPL-?3|GPLv3|GNU (Affero )?General Public)", text, flags=re.I), rel
