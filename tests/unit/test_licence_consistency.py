"""The current licence of the project is Apache-2.0, stated the same way in every current metadata file."""
import json
import os
import re

from conftest import ROOT


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf8") as fh:
        return fh.read()


def test_license_file_is_the_apache_2_text():
    text = _read("LICENSE")
    assert "Apache License" in text and "Version 2.0, January 2004" in text
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in text
    assert "APPENDIX: How to apply the Apache License to your work." in text
    assert not re.search(r"^MIT License|GNU (Affero )?General Public|Permission is hereby granted, free of charge", text, flags=re.M)


def test_metadata_and_readme_agree_on_apache_and_carry_a_copyright_line():
    assert re.search(r'^license = "Apache-2\.0"', _read("pyproject.toml"), flags=re.M)
    assert re.search(r"^license: Apache-2\.0$", _read("CITATION.cff"), flags=re.M)
    assert json.loads(_read(".zenodo.json"))["license"].lower() == "apache-2.0"
    readme = _read("README.md")
    assert "Apache License 2.0" in readme and "Apache-2.0" in readme
    assert re.search(r"Copyright \(c\) \d{4} \S", readme)


def test_no_current_project_file_names_another_licence_for_cowmrp():
    for rel in ("pyproject.toml", "CITATION.cff", ".zenodo.json"):
        text = _read(rel)
        assert not re.search(r"\b(AGPL|GPL-?3|GPLv3|GNU (Affero )?General Public)", text, flags=re.I), rel
        assert not re.search(r"\bMIT\b", text, flags=re.I), rel
