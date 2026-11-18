#!/usr/bin/env python3
"""
validate_model_documentation.py
================================
Lightweight, dependency-free STRUCTURAL check that the documentation
system in docs/ hasn't silently fallen out of sync with the code. This
checks PRESENCE, not correctness -- it can never tell you a citation is
wrong or an equation is scientifically invalid. That always requires a
human (or agent) doing the primary-source audit process described in
docs/evidence/REFERENCES.md. See PRIMARY.md's documentation-maintenance section
for what is and isn't automatable.

Checks:
1. Every docs/ file listed in REQUIRED_DOCS exists.
2. Every environmental *_METHOD_INFO entry (cow/rewards/environment/*.py)
   has all the required template fields (docs/model/ENVIRONMENTAL_EMISSIONS.md
   schema, docs/model/THESIS_MODEL.md's method template).
3. Every environmental reward spec name (reward_nitrogen, ...) and every
   ENABLE_REWARD_* flag name is mentioned somewhere in
   docs/model/REWARD_SYSTEM.md.
4. Every environmental bio-quantity name (nitrogen_intake_g, ...) is
   mentioned somewhere in docs/model/ENVIRONMENTAL_EMISSIONS.md.

Exits 0 and prints DOC_CHECK_OK on success; exits 1 and prints every gap
found on failure (does not stop at the first one).

USAGE
-----
    python validate_model_documentation.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))   # project root (mrp/, cow/)
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DOCS = os.path.join(ROOT, "docs")
DOC_GROUPS = ["architecture", "model", "assumptions", "evidence", "project"]   # docs/<group>/NAME.md


def _doc_path(name):
    for g in DOC_GROUPS:
        p = os.path.join(DOCS, g, name)
        if os.path.isfile(p):
            return p
    return os.path.join(DOCS, name)

REQUIRED_DOCS = [
    "THESIS_MODEL.md", "ARCHITECTURE.md", "BIOLOGICAL_ASSUMPTIONS.md",
    "ENVIRONMENTAL_EMISSIONS.md", "REWARD_SYSTEM.md", "REFERENCES.md",
    "ASSUMPTIONS_REGISTER.md", "LIMITATIONS_REGISTER.md", "VALIDATION.md",
    "MODEL_CHANGELOG.md", "SYSTEM_WORKFLOW.md",
]

REQUIRED_METHOD_INFO_FIELDS = [
    "citation", "applicable_animals", "scope", "required_inputs",
    "optional_inputs", "equation", "units", "parameter_sources",
    "assumptions", "limitations",
]

errors = []


def check_docs_exist():
    for name in REQUIRED_DOCS:
        path = _doc_path(name)
        if not os.path.isfile(path):
            errors.append(f"Missing required doc: docs/<group>/{name}")


def check_method_info_fields():
    sys.path.insert(0, ROOT)
    from cowmrp.domains.cow.rewards.environment.nitrogen import NITROGEN_METHOD_INFO
    from cowmrp.domains.cow.rewards.environment.enteric_methane import ENTERIC_METHANE_METHOD_INFO
    from cowmrp.domains.cow.rewards.environment.manure_methane import MANURE_METHANE_METHOD_INFO
    from cowmrp.domains.cow.rewards.environment.co2 import CO2_METHOD_INFO
    from cowmrp.domains.cow.rewards.environment.n2o import N2O_METHOD_INFO

    registries = {
        "nitrogen": NITROGEN_METHOD_INFO, "enteric_methane": ENTERIC_METHANE_METHOD_INFO,
        "manure_methane": MANURE_METHANE_METHOD_INFO, "co2": CO2_METHOD_INFO,
        "n2o": N2O_METHOD_INFO,
    }
    for component, registry in registries.items():
        for method_name, info in registry.items():
            missing = [f for f in REQUIRED_METHOD_INFO_FIELDS if f not in info]
            if missing:
                errors.append(
                    f"{component}::{method_name}'s *_METHOD_INFO entry is missing "
                    f"field(s): {missing}"
                )
    return registries


def check_reward_system_doc_mentions():
    from cowmrp.domains.cow.parameters import (
        ENABLE_REWARD_NITROGEN, ENABLE_REWARD_ENTERIC_METHANE,
        ENABLE_REWARD_MANURE_METHANE, ENABLE_REWARD_N2O,
    )
    reward_doc = _read("REWARD_SYSTEM.md")
    for flag_name in ["ENABLE_REWARD_NITROGEN", "ENABLE_REWARD_ENTERIC_METHANE",
                       "ENABLE_REWARD_MANURE_METHANE", "ENABLE_REWARD_N2O"]:
        if flag_name not in reward_doc:
            errors.append(f"docs/model/REWARD_SYSTEM.md doesn't mention flag: {flag_name}")

    from cowmrp.domains.cow.rewards.environment.reward_nitrogen import reward_nitrogen_spec
    from cowmrp.domains.cow.rewards.environment.reward_enteric_methane import reward_enteric_methane_spec
    from cowmrp.domains.cow.rewards.environment.reward_manure_methane import reward_manure_methane_spec
    from cowmrp.domains.cow.rewards.environment.reward_n2o import reward_n2o_spec
    for spec in [reward_nitrogen_spec, reward_enteric_methane_spec,
                 reward_manure_methane_spec, reward_n2o_spec]:
        if spec['name'] not in reward_doc:
            errors.append(f"docs/model/REWARD_SYSTEM.md doesn't mention reward spec: {spec['name']}")
    if 'reward_environment_total' not in reward_doc:
        errors.append("docs/model/REWARD_SYSTEM.md doesn't mention derived output: reward_environment_total")


def check_environmental_quantities_documented(registries):
    env_doc = _read("ENVIRONMENTAL_EMISSIONS.md")
    bio_names = [
        "nitrogen_intake_g", "fecal_n_g", "urinary_n_g", "milk_n_g", "manure_n_g",
        "enteric_ch4_g", "manure_ch4_kg", "respiratory_co2_kg",
        "n2o_direct_kg", "n2o_indirect_volatilization_kg", "n2o_indirect_leaching_kg",
        "n2o_total_kg", "bw_kg", "dmi_kg", "milk_yield_kg",
    ]
    for name in bio_names:
        if name not in env_doc:
            errors.append(f"docs/model/ENVIRONMENTAL_EMISSIONS.md doesn't mention quantity: {name}")


def _read(doc_name):
    path = _doc_path(doc_name)
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    check_docs_exist()
    if not errors:  # only proceed to code-dependent checks if docs/ itself is intact
        registries = check_method_info_fields()
        check_reward_system_doc_mentions()
        check_environmental_quantities_documented(registries)

    if errors:
        print(f"DOC_CHECK_FAILED: {len(errors)} issue(s)\n")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("DOC_CHECK_OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
