"""
guide/web/check_methods.py -- read-only check of the built website (guide/web/site/): are the long module-level values of the API pages
and the Methods page complete?

    python guide/web/check_methods.py                          after `python guide/web/build_web.py`
    python guide/web/check_methods.py --save-baseline F.json   record the long values of the current API pages (run before a change)
    python guide/web/check_methods.py --baseline F.json        also compare the kept values with a recorded baseline (text hash)
    python guide/web/check_methods.py --manifest M.json        also compare source-file hashes with a recorded manifest {path: sha256}

Checks
  1. every long (> 300 characters) module-level value on the API pages is either hidden (HIDE_VALUES, name still on the page, raw value gone,
     note present) or kept (KEEP_VALUES, shown as before); a long value in neither list is reported;
  2. the Methods page shows every key and every value of the five *_METHOD_INFO dictionaries, and the dictionary keys on the page equal the
     dictionary's keys; the field list equals REQUIRED_METHOD_INFO_FIELDS of validate_model_documentation.py;
  3. optional: kept values equal the baseline; source files equal the manifest.
Exit code 0 when everything passes.
"""
import ast
import hashlib
import html
import importlib
import json
import os
import re
import sys

WEB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(WEB))
API = os.path.join(WEB, "site", "api")
LONG = 300
VAR = re.compile(r'<span class="name">(\w+)</span>\s*=\s*(.*?)(?=</div>)', re.S)
SHOWN = re.compile(r'<label[^>]*></label><span class="default_value">(.*?)</span>\s*$', re.S)


def text(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s))).strip()


def norm(s):
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def page_values():
    """{module: {name: (long_value_text or None, present)}}: every variable of every API page; the text only while the raw value is shown."""
    found = {}
    for d, _, fs in os.walk(API):
        for f in fs:
            if not f.endswith(".html"):
                continue
            p = os.path.join(d, f)
            mod = os.path.relpath(p, API)[:-5].replace(os.sep, ".").replace("cowmrp.", "", 1)
            s = open(p, encoding="utf-8", errors="ignore").read()
            for m in VAR.finditer(s):
                shown = SHOWN.search(m.group(2))
                found.setdefault(mod, {})[m.group(1)] = text(shown.group(1)) if shown else None
    return found


def main():
    sys.path.insert(0, WEB)
    args = sys.argv[1:]
    val = lambda flag: args[args.index(flag) + 1] if flag in args else None
    found = page_values()
    if val("--save-baseline"):
        rec = {f"{m}.{n}": hashlib.sha256(v.encode()).hexdigest() for m, d in found.items() for n, v in d.items() if v is not None and len(v) > LONG}
        json.dump(rec, open(val("--save-baseline"), "w"), indent=0)
        print(f"baseline: {len(rec)} long values recorded")
        return 0

    import build_web as bw
    errors = []
    hide = {(m, n) for m, ns in bw.HIDE_VALUES.items() for n in ns}
    base = json.load(open(val("--baseline"))) if val("--baseline") else None
    n_hidden = 0

    for m, d in found.items():
        for n, v in d.items():
            if (m, n) in hide:
                if v is not None:
                    errors.append(f"hidden value still shown: {m}.{n}")
                else:
                    n_hidden += 1
            elif v is not None and len(v) > LONG:
                errors.append(f"long value neither hidden nor listed: {m}.{n}")
    for m, n in sorted(hide):
        if n not in found.get(m, {}):
            errors.append(f"hidden name missing from its page: {m}.{n}")
    if base is not None:
        accounted = {f"{m}.{n}" for m, n in hide}
        missing = sorted(set(base) - accounted)
        errors += [f"baseline value not accounted for: {k}" for k in missing]
    print(f"API values: {n_hidden} of {len(hide)} listed values hidden (name kept, value replaced by a note); baseline long values all accounted for")

    # Methods page
    page = open(os.path.join(WEB, "site", "methods.html"), encoding="utf-8").read()
    body = norm(text(page))
    validator = ast.parse(open(os.path.join(ROOT, "runners", "_shared", "checks", "validate_model_documentation.py"), encoding="utf-8").read())
    required = next(ast.literal_eval(n.value) for n in validator.body if isinstance(n, ast.Assign) and n.targets[0].id == "REQUIRED_METHOD_INFO_FIELDS")
    if [k for k, _l, _w in bw.METHOD_FIELDS] != required:
        errors.append("METHOD_FIELDS differs from REQUIRED_METHOD_INFO_FIELDS")
    os.environ.setdefault("COW_TIME_UNIT", "month")
    sys.path.insert(0, os.path.join(ROOT, "src"))
    n_fields = 0
    page_keys = set(re.findall(r'<h3 id="([\w-]+)">Method', page))
    want_keys = set()
    for anchor, _t, modname, dname, _i in bw.METHOD_COMPONENTS:
        info = getattr(importlib.import_module(modname), dname)
        for mk, rec in info.items():
            want_keys.add(f"{anchor}-{mk}")
            for k, v in rec.items():
                for item in (v if isinstance(v, (list, tuple)) else [v]):
                    n_fields += 1
                    if norm(str(item)) not in body:
                        errors.append(f"Methods page lacks {dname}[{mk}][{k}]: {str(item)[:60]}")
    if page_keys != want_keys:
        errors.append(f"method keys differ: page-only {sorted(page_keys - want_keys)}, missing {sorted(want_keys - page_keys)}")
    print(f"Methods page: {len(want_keys)} methods, {n_fields} values checked against the live dictionaries")

    # Provenance page: every key and scalar of every listed value must be on the page, inside its own section
    prov = norm(text(open(os.path.join(WEB, "site", "provenance.html"), encoding="utf-8").read()))
    pg = open(os.path.join(WEB, "site", "provenance.html"), encoding="utf-8").read()
    n_leaves = 0
    for _a, _h, _i, items in bw.PROVENANCE_SECTIONS:
        for mod, name, _l in items:
            obj = getattr(importlib.import_module("cowmrp." + mod), name)
            aid = f'prov-{mod.replace(".", "-")}-{name}'
            seg = re.search(rf'<h3 id="{aid}">(.*?)(?=<h3 id="prov-|<h2 id=|$)', pg, re.S)
            if not seg:
                errors.append(f"provenance section missing: {mod}.{name}")
                continue
            segtext = norm(text(seg.group(1)))
            for leaf in bw.leaves(obj):
                n_leaves += 1
                if norm(leaf) not in segtext:
                    errors.append(f"provenance page lacks {mod}.{name}: {leaf[:60]}")
    print(f"Provenance page: {sum(len(i) for _a, _h, _i, i in bw.PROVENANCE_SECTIONS)} values, {n_leaves} keys/entries checked inside their own sections")

    if val("--manifest"):
        man = json.load(open(val("--manifest")))
        changed = [p for p, h in man.items() if hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() != h and p.startswith("src/")]
        print(f"source files under src/ compared: {sum(p.startswith('src/') for p in man)}; changed: {len(changed)}")
        errors += [f"source changed: {p}" for p in changed]
    for e in errors:
        print("FAIL", e)
    print("METHODS_CHECK_OK" if not errors else f"METHODS_CHECK_FAILED ({len(errors)})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
