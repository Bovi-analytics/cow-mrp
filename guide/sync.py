"""guide/sync.py -- keep the documentation in step with the code.

The idea: a fact that the documentation repeats (state counts, the reward components, the overridable economic settings, ...) is computed
from the package and written into the documents by this tool; it is not typed by hand. A document that has gone out of date is detected.

    python guide/sync.py write            rewrite the generated blocks of the hand-written documents (README.md) from the live package
    python guide/sync.py check            exit 1 if anything is stale or inconsistent (see below); safe for CI
    python guide/sync.py impact PATH      which documents and outputs depend on a source file (e.g. src/cowmrp/domains/cow/rewards/base.py)
    python guide/sync.py facts            print the facts

How it works
    * Facts (``facts()``) are read from the package: sizes of the models, the reward components and signs, the economic settings with their
      defaults, units and recorded sources. The documents use them in two ways: ``{{NAME}}`` tokens (tutorial and site pages, expanded at build time by
      ``fill_tokens``) and generated blocks in README.md between ``<!-- GEN:name -->`` and ``<!-- /GEN -->`` (rewritten by ``write``).
    * Every site build records a stamp (``.sync_stamp.json``) with a hash of everything the site is built from (the package source, the build scripts,
      the hand-written pages, the examples, CITATION.cff, pyproject.toml, README.md). ``check`` recomputes it: a site built from older sources is reported as stale.
    * ``check`` also reports: a generated README block that no longer matches the package; a ``{{TOKEN}}`` that nothing defines or that was left unexpanded in a built
      page; a model-size number typed by hand in a hand-written document (it should be a token or a block); generated outputs of the runners
      (review pages, dashboards, projections, reward-vector tables) that are older than the model sources they were computed from (a warning, because outputs are local).
    * ``guide/build_all.py`` runs: write, both site builds, then check and the other guide checks.

Nothing here changes the model: it only reads the package and writes documents.
"""
from __future__ import annotations

import ast
import contextlib
import fnmatch
import glob
import hashlib
import html
import io
import json
import os
import re
import sys
import time

GUIDE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(GUIDE)
sys.path.insert(0, os.path.join(ROOT, "src"))

STAMP = ".sync_stamp.json"
SITES = {"guide/site": "guide/build.py", "guide/web/site": "guide/web/build_web.py"}

# Hand-written documents that carry generated blocks or tokens (blocks are rewritten by ``write``).
BLOCK_DOCS = ["README.md"]
TOKEN_DOCS = ["guide/tutorial.html", "guide/web/content/home.html", "guide/web/content/science.html", "guide/web/content/getting-started.html"]
# Tokens that the site builders fill from other sources (not facts of this file).
BUILD_TOKENS = {"CODE", "OUT", "STATS"}

# What a built site is made from.
SITE_SOURCES = ["src/**/*.py", "guide/build.py", "guide/sync.py", "guide/web/build_web.py", "guide/web/content/*", "guide/web/theme/*", "guide/tutorial.html",
                "examples/*.py", "CITATION.cff", "pyproject.toml", "README.md"]

# Model sources -> what is computed from them (for ``impact`` and the output-freshness check).
MODEL_SOURCES = ["src/cowmrp/domains/cow/parameters.py", "src/cowmrp/domains/cow/rewards/**/*.py", "src/cowmrp/domains/cow/probabilities/**/*.py",
                 "src/cowmrp/domains/cow/states.py", "src/cowmrp/domains/cow/tables.py", "src/cowmrp/domains/cow/transitions.py",
                 "src/cowmrp/domains/cow/state_constraints.py", "src/cowmrp/domains/cow/lookup_tables.py", "src/cowmrp/mrp/*.py"]
GENERATED_OUTPUTS = ["runners/cow/outputs/review", "runners/cow/outputs/dashboard", "runners/cow/outputs/projection", "runners/cow/outputs/model",
                     "runners/cow/outputs/notebook", "runners/cow/outputs/farm/results", "runners/youngstock/outputs"]
IMPACT = [   # (source pattern, documents and outputs that depend on it)
    ("src/cowmrp/domains/cow/rewards/**", ["README.md blocks: reward_table, econ_table", "guide/web site: Methods, Provenance, Science, API pages", "guide/site: same pages",
                                           "runners/cow/outputs: review, dashboard, projection, model/*/reward_vector*.csv, notebook outputs",
                                           "checks: runners/cow/checks/verify_reward_vector.py, verify_value_comparisons.py", "tests/baselines/*.json (golden numbers)",
                                           "docs/model/REWARD_SYSTEM.md (hand-written, local)"]),
    ("src/cowmrp/domains/cow/economics.py", ["README.md block: econ_table", "guide pages: API, Methods (economic settings)"]),
    ("src/cowmrp/domains/cow/parameters.py", ["README.md block: state_counts", "guide pages: tokens CO_STATES, CO_NNZ; Provenance page", "all runners/cow/outputs",
                                              "docs/assumptions/ASSUMPTIONS_REGISTER.md (hand-written, local)"]),
    ("src/cowmrp/domains/cow/probabilities/**", ["all runners/cow/outputs", "tests/baselines/*.json", "guide pages: Provenance, API"]),
    ("src/cowmrp/domains/cow/{states,tables,transitions,state_constraints}.py", ["README.md block: state_counts", "tokens CO_STATES, CO_NNZ, CO_REACHABLE", "all runners/cow/outputs", "tests/baselines/*.json"]),
    ("src/cowmrp/domains/youngstock/**", ["tokens YS_STATES, YS_NNZ", "runners/youngstock/outputs", "guide pages: Science"]),
    ("src/cowmrp/**/*.py", ["API pages of both sites (docstrings)", "site stamps (rebuild needed)"]),
    ("guide/tutorial.html", ["guide/site/tutorial.html", "guide/web/site/tutorial.html"]),
    ("runners/cow/farm/keepcull.py", ["README.md (commands)", "guide tutorial (keep/cull section)"]),
    ("README.md", ["site stamps (rebuild needed)"]),
]


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# facts
# ----------------------------------------------------------------------------------------------------------------------------------------------------

_FACTS = None


def _first_sentence(doc):
    doc = " ".join((doc or "").split())
    m = re.match(r"(.+?[.!?])(\s|$)", doc)
    return (m.group(1) if m else doc)[:200]


def facts():
    """Every fact the documents repeat, computed from the package at this moment."""
    global _FACTS
    if _FACTS is not None:
        return _FACTS
    with contextlib.redirect_stdout(io.StringIO()):
        from cowmrp.model import build_model
        from cowmrp.mrp.utils import reachable_states
        from cowmrp.domains.cow import economics
        from cowmrp.domains.cow.rewards import BASE_REWARD_SPECS
        from cowmrp.domains.cow.transitions import get_replacement_state_id
        c, y = build_model("cow"), build_model("youngstock")
    reach = len(reachable_states(c.P, get_replacement_state_id(c.states)))
    econ_rows = [(s["name"], s["default"], s["unit"], s["currency"], s["price_year"], s["geography"], s["affects"], s["source"]) for s in economics.ECONOMIC_SETTINGS.values()]
    rew_rows = []
    for spec in BASE_REWARD_SPECS:
        fn = spec["methods"].get("formula")
        mod = sys.modules.get(fn.__module__) if fn else None
        rew_rows.append((spec["name"], "+" if spec["sign"] > 0 else "-", _first_sentence(mod.__doc__) if mod else ""))
    from cowmrp.domains.cow.parameters import TIME_UNIT
    _FACTS = {
        "TIME_UNIT": TIME_UNIT, "COW_STATES": f"{len(c.states):,}", "COW_REACHABLE": f"{reach:,}", "COW_NNZ": f"{c.P.nnz:,}",
        "CO_STATES": f"{len(c.states):,}", "CO_NNZ": f"{c.P.nnz:,}", "YS_STATES": f"{len(y.states):,}", "YS_NNZ": f"{y.P.nnz:,}",   # CO_* are the names the site content already uses
        "_econ_rows": econ_rows, "_reward_rows": rew_rows,
    }
    _FACTS["EXCEL_COLUMNS_TABLE"] = excel_columns_html()
    _FACTS["ECON_TABLE"] = econ_table_html()
    return _FACTS


def _fmt(v):
    return f"{v:,.4g}" if abs(v) < 1000 else f"{v:,.0f}"


def _md(s):
    return str(s).replace("|", "\\|")


def block_state_counts():
    f = facts()
    return (f"State counts ({f['TIME_UNIT']} resolution): {f['COW_STATES']} states, of which {f['COW_REACHABLE']} are reachable from the replacement state "
            f"(the default is the full table), {f['COW_NNZ']} transitions.")


def block_reward_table():
    rows = ["| Reward component | Sign | What it is |", "|---|---|---|"]
    rows += [f"| `{n}` | {s} | {_md(d)} |" for n, s, d in facts()["_reward_rows"]]
    return ("The reward vector `total` is the signed sum of these components (environmental terms only when switched on). "
            "Replacement cost is not in it; it is charged in the keep/cull comparison.\n\n" + "\n".join(rows))


def block_econ_table():
    rows = ["| Setting | Default | Unit | Affects | Source recorded in the repository (price year; geography) |", "|---|---|---|---|---|"]
    for n, d, unit, cur, yr, geo, aff, src in facts()["_econ_rows"]:
        meta = "; ".join(x for x in (f"price year {yr}" if yr else "price year not recorded", geo or "geography not recorded") if x)
        rows.append(f"| `{n}` | {_fmt(d)} | {_md(unit)} ({cur}) | {_md(aff)} | {_md(src)} ({meta}) |")
    return "Overridable economic settings (defaults are those of `cowmrp.domains.cow.rewards.base`):\n\n" + "\n".join(rows)


def econ_table_html():
    """The same table for the site pages (Methods)."""
    rows = []
    for n, d, unit, cur, yr, geo, aff, src in facts()["_econ_rows"]:
        meta = "; ".join(x for x in (f"price year {yr}" if yr else "price year not recorded", geo or "geography not recorded") if x)
        rows.append(f"<tr><td><code>{n}</code></td><td>{html.escape(_fmt(d))}</td><td>{html.escape(unit)} ({cur})</td><td>{html.escape(aff)}</td>"
                    f"<td>{html.escape(src)} <span class=\"mut\">({html.escape(meta)})</span></td></tr>")
    return ('<h2 id="economic-settings">Economic settings that can be overridden</h2>'
            '<p>These prices and costs can be changed for one run with <code>economic_overrides</code>; anything not named keeps the default shown. '
            'The table is read from the package when the site is built. Biological and management values are not overridable. '
            'The discount rate and the sale offer are arguments of the keep/cull functions.</p>'
            '<table><tr><th>Setting</th><th>Default</th><th>Unit</th><th>Affects</th><th>Source recorded in the repository</th></tr>' + "".join(rows) + "</table>")


def _excel_rows():
    from cowmrp.domains.cow.keepcull_excel import HEADERS, WORKBOOK_COLUMNS
    kinds = {"herd": "from the herd file", "model": "model value (pasted)", "input": "input (yellow)", "formula": "Excel formula", "check": "check"}
    return [(letter, HEADERS[i], kinds[kind], formula.format(r="<row>", h="<horizon row>") if formula else "", meaning)
            for i, (letter, kind, formula, meaning) in enumerate(WORKBOOK_COLUMNS)]


def block_excel_columns():
    rows = ["| Col | Header | Kind | Formula (row r) | Meaning |", "|---|---|---|---|---|"]
    rows += [f"| {a} | {_md(b)} | {c} | {('`' + _md(d).replace('<row>', 'r') + '`') if d else ''} | {_md(e)} |" for a, b, c, d, e in _excel_rows()]
    return chr(10).join(rows)


def excel_columns_html():
    rows = "".join(f"<tr><td>{a}</td><td>{html.escape(b)}</td><td>{c}</td><td>{('<code>' + html.escape(d).replace('&lt;row&gt;', 'r') + '</code>') if d else ''}</td><td>{html.escape(e)}</td></tr>"
                   for a, b, c, d, e in _excel_rows())
    return ('<table><tr><th>Col</th><th>Header</th><th>Kind</th><th>Formula (row r)</th><th>Meaning</th></tr>' + rows + "</table>")


BLOCKS = {"state_counts": block_state_counts, "reward_table": block_reward_table, "econ_table": block_econ_table, "excel_columns": block_excel_columns}
MARK = re.compile(r"(<!-- GEN:(\w+) -->\r?\n)(.*?)(\r?\n<!-- /GEN -->)", re.S)
TOKEN = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


def fill_tokens(text):
    """Expand ``{{NAME}}`` tokens that are facts; tokens the site builders fill themselves are left for them."""
    f = facts()
    return TOKEN.sub(lambda m: str(f[m.group(1)]) if m.group(1) in f and not m.group(1).startswith("_") else m.group(0), text)


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf8", newline="") as fh:
        return fh.read()


def _write(rel, text):
    with open(os.path.join(ROOT, rel), "w", encoding="utf8", newline="") as fh:
        fh.write(text)


def render_blocks(text):
    nl = "\r\n" if "\r\n" in text else "\n"

    def one(m):
        name = m.group(2)
        if name not in BLOCKS:
            raise SystemExit(f"unknown generated block {name!r}; known: {', '.join(BLOCKS)}")
        return m.group(1) + BLOCKS[name]().replace("\n", nl) + m.group(4)
    return MARK.sub(one, text)


def write():
    changed = []
    for rel in BLOCK_DOCS:
        old = _read(rel)
        new = render_blocks(old)
        if new != old:
            _write(rel, new)
            changed.append(rel)
    print("sync write: " + (", ".join(changed) + " updated" if changed else "generated blocks already current"))
    return changed


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# stamps and checks
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def _files(patterns):
    out = set()
    for pat in patterns:
        for p in glob.glob(os.path.join(ROOT, pat), recursive=True):
            if os.path.isfile(p) and "__pycache__" not in p:
                out.add(os.path.relpath(p, ROOT).replace(os.sep, "/"))
    return sorted(out)


def source_hash():
    h = hashlib.sha256()
    for rel in _files(SITE_SOURCES):
        h.update(rel.encode())
        with open(os.path.join(ROOT, rel), "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def write_stamp(site_dir):
    """Called by the site builders: record what the site was built from."""
    os.makedirs(site_dir, exist_ok=True)
    with open(os.path.join(site_dir, STAMP), "w", encoding="utf8") as fh:
        json.dump({"sources_sha256": source_hash(), "built_unix": int(time.time()), "n_source_files": len(_files(SITE_SOURCES))}, fh, indent=1)


def check_blocks(problems):
    for rel in BLOCK_DOCS:
        text = _read(rel)
        if render_blocks(text) != text:
            problems.append(f"{rel}: a generated block is out of date (run: python guide/sync.py write)")
        for m in MARK.finditer(text):
            if m.group(2) not in BLOCKS:
                problems.append(f"{rel}: unknown generated block {m.group(2)!r}")


def check_tokens(problems):
    known = set(k for k in facts() if not k.startswith("_")) | BUILD_TOKENS
    for rel in TOKEN_DOCS:
        if os.path.exists(os.path.join(ROOT, rel)):
            for t in sorted(set(TOKEN.findall(_read(rel)))):
                if t not in known:
                    problems.append(f"{rel}: token {{{{{t}}}}} is not defined by guide/sync.py or the site builders")


def check_hardcoded(problems):
    """A model-size number typed by hand in a hand-written document (outside generated blocks) goes stale; it should be a token or a block."""
    f = facts()
    numbers = {k: f[k] for k in ("COW_STATES", "COW_REACHABLE", "COW_NNZ", "YS_STATES", "YS_NNZ")}
    for rel in BLOCK_DOCS + TOKEN_DOCS + ["runners/README.md"]:
        if not os.path.exists(os.path.join(ROOT, rel)):
            continue
        text = MARK.sub("", _read(rel))
        text = TOKEN.sub("", text)
        for name, val in numbers.items():
            if re.search(rf"(?<![\d,.]){re.escape(val)}(?![\d,]|\.\d)", text):
                problems.append(f"{rel}: the number {val} ({name}) is typed by hand; use {{{{{name}}}}} or a generated block so it cannot go stale")


def check_sites(problems):
    cur = None
    for site, builder in SITES.items():
        d = os.path.join(ROOT, site)
        if not os.path.isdir(d):
            continue                                   # a site that was never built is not stale
        stamp = os.path.join(d, STAMP)
        if not os.path.exists(stamp):
            problems.append(f"{site}: no build stamp; rebuild it (python {builder})")
            continue
        cur = cur or source_hash()
        if json.load(open(stamp, encoding="utf8")).get("sources_sha256") != cur:
            problems.append(f"{site}: built from older sources than the current ones; rebuild it (python {builder}, or python guide/build_all.py)")
        for dirpath, _, files in os.walk(d):
            if os.path.relpath(dirpath, d).split(os.sep)[0] == "api":
                continue
            for fn in files:
                if fn.endswith(".html"):
                    with open(os.path.join(dirpath, fn), encoding="utf8") as fh:
                        txt = fh.read()
                    left = [t for t in TOKEN.findall(txt) if t not in ("url",)]
                    if left or "<!-- GEN:" in txt:
                        problems.append(f"{site}/{os.path.relpath(os.path.join(dirpath, fn), d)}: unexpanded token or generated-block marker {left[:3]}")


def check_links(problems):
    """Every relative link and image of the built sites points to a file that exists, and every #fragment to an id in that file."""
    href = re.compile(r'(?:href|src)="([^"#:?][^"]*?)(#[^"]*)?"')
    ids_cache = {}

    def ids(path):
        if path not in ids_cache:
            with open(path, encoding="utf8", errors="ignore") as fh:
                ids_cache[path] = set(re.findall(r'\sid="([^"]+)"', fh.read()))
        return ids_cache[path]

    for site in SITES:
        base = os.path.join(ROOT, site)
        if not os.path.isdir(base):
            continue
        for dirpath, _, files in os.walk(base):
            if os.path.relpath(dirpath, base).split(os.sep)[0] == "api":
                continue                                                      # pdoc pages are generated and checked by pdoc itself
            for fn in files:
                if not fn.endswith(".html"):
                    continue
                src = os.path.join(dirpath, fn)
                with open(src, encoding="utf8", errors="ignore") as fh:
                    txt = fh.read()
                for m in href.finditer(txt):
                    target = os.path.normpath(os.path.join(dirpath, m.group(1)))
                    where = f"{site}/{os.path.relpath(src, base).replace(os.sep, '/')}"
                    if not os.path.exists(target):
                        problems.append(f"{where}: broken link {m.group(1)}")
                    elif m.group(2) and len(m.group(2)) > 1 and target.endswith(".html") and m.group(2)[1:] not in ids(target):
                        problems.append(f"{where}: link {m.group(1)}{m.group(2)} - no such anchor")


def check_outputs(warnings_):
    src = _files(MODEL_SOURCES)
    if not src:
        return
    newest = max(os.path.getmtime(os.path.join(ROOT, p)) for p in src)
    for rel in GENERATED_OUTPUTS:
        d = os.path.join(ROOT, rel)
        if not os.path.isdir(d):
            continue
        files = [os.path.join(dp, fn) for dp, _, fs in os.walk(d) for fn in fs]
        if files and max(os.path.getmtime(p) for p in files) < newest:
            warnings_.append(f"{rel}: older than the model sources it was computed from; regenerate it with its runner (see `python guide/sync.py impact <file>`)")


def check(strict_outputs=False):
    problems, warn = [], []
    check_blocks(problems)
    check_tokens(problems)
    check_hardcoded(problems)
    check_sites(problems)
    check_links(problems)
    check_outputs(problems if strict_outputs else warn)
    for w in warn:
        print("WARNING:", w)
    for p in problems:
        print("STALE:  ", p)
    print(f"sync check: {len(problems)} problem(s), {len(warn)} warning(s)")
    return 1 if problems else 0


def impact(path):
    rel = os.path.relpath(os.path.abspath(path), ROOT).replace(os.sep, "/") if os.path.isabs(path) or os.path.exists(path) else path.replace("\\", "/")
    hits = []
    for pat, consumers in IMPACT:
        alts = [pat]
        m = re.search(r"\{([^}]*)\}", pat)
        if m:
            alts = [pat.replace(m.group(0), a) for a in m.group(1).split(",")]
        if any(fnmatch.fnmatch(rel, a.replace("**", "*")) for a in alts):
            hits.append((pat, consumers))
    print(f"{rel}")
    if not hits:
        print("  no registered dependents (add the file to IMPACT in guide/sync.py if documents are computed from it)")
    for pat, consumers in hits:
        print(f"  matches {pat}:")
        for c in consumers:
            print(f"    - {c}")
    return 0


def _runner_options(rel):
    """The command-line options of a runner, read from its source (no import)."""
    tree = ast.parse(_read(rel))
    return [n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument" and n.args
            and isinstance(n.args[0], ast.Constant) and str(n.args[0].value).startswith("--")]


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "check"
    if cmd == "write":
        write()
        return 0
    if cmd == "check":
        return check(strict_outputs="--strict-outputs" in argv)
    if cmd == "impact" and len(argv) > 2:
        return impact(argv[2])
    if cmd == "facts":
        for k, v in facts().items():
            if not k.startswith("_") and len(str(v)) < 60:
                print(f"{k} = {v}")
        print("runner options (runners/cow/farm/keepcull.py):", ", ".join(_runner_options("runners/cow/farm/keepcull.py")))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
