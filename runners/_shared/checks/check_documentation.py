"""
runners/_shared/checks/check_documentation.py
=============================================
Read-only consistency check of the public documentation. It writes nothing and imports nothing from the model.

    python runners/_shared/checks/check_documentation.py [--strict]

The source code (docstrings, comments, constants) is the authoritative text; the API pages are generated from it by
`python guide/build.py` and `python guide/web/build_web.py`. This script detects where the generated pages, the READMEs or
the guide have drifted from it:

  1. PRIVATE   public files (source, runners, tests, README files, guide sources) that point at the private `docs/` folder,
               `PRIMARY.md`, `CLAUDE.md` or `notes/`.
  2. HISTORY   wording that belongs in private working notes (dated change notes, "this pass", private register IDs).
  3. STALE     a generated API page that is older than its source file, a module missing from the API pages, or a module whose
               docstring opening words are not on its page (the page was built from different text).
  4. LINKS     local href/src targets that do not exist, in both generated sites.
  5. NAMES     CONSTANT_NAMES quoted in README files that no longer exist in the package.
  6. GUIDE     the existing `guide/audit_guide.py` (paths and file lists quoted by the guide).

Exit code 1 when any check reports a finding (with --strict, 0 findings are required for every check; without it, items that
are known and reported separately are still printed but only STALE/LINKS/NAMES fail). Run it after rebuilding the sites.
"""
import html
import importlib
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.dont_write_bytecode = True
PUBLIC_DIRS = ("src", "runners", "tests", "examples")
PUBLIC_FILES = ("README.md", "pyproject.toml")
SKIP_DIRS = {"__pycache__", "outputs", ".venv", "build", "site", ".git", "notes", "older_outputs", "dist", ".pytest_cache"}
TEXT_EXT = (".py", ".md", ".toml", ".html", ".txt", ".cfg", ".yml", ".yaml")
SITES = ("guide/site", "guide/web/site")

# Files that legitimately mention the private `docs/` folder because their function depends on it. They are listed here with the
# reason, so the exception is visible, and they are reported as ALLOWED instead of being hidden.
ALLOWED_PRIVATE = {
    "runners/_shared/checks/validate_model_documentation.py": "compares the code with the maintainers' local documentation folder (maintainer-only check)",
    "guide/audit_guide.py": "treats docs/ and PRIMARY.md as local-only paths when auditing the guide",
    "tests/install/test_clean_install.py": "asserts that nothing from docs/ is packaged in the wheel",
}

PRIVATE = re.compile(r"(?<![\w/.-])(?:docs/(?!Literature/?\b)\S*|PRIMARY\.md|CLAUDE\.md|notes/)")
HISTORY = re.compile(
    r"\bthis pass\b|\bCITATION REMOVED\b|\bVERIFIED exact\b|\buser will supply\b|\bscientific audit\b|\(audit P\d+\)"
    r"|\b(?:A|L)-\d{3}\b|(?<!report of )\b20\d\d-\d\d-\d\d\b", re.I)


def public_files():
    for d in PUBLIC_DIRS:
        for dp, dirs, files in os.walk(os.path.join(ROOT, d)):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for fn in files:
                if fn.endswith(TEXT_EXT):
                    yield os.path.join(dp, fn)
    for fn in PUBLIC_FILES:
        if os.path.isfile(os.path.join(ROOT, fn)):
            yield os.path.join(ROOT, fn)
    gd = os.path.join(ROOT, "guide")
    for dp, dirs, files in os.walk(gd):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and x not in ("assets",)]
        for fn in files:
            if fn.endswith(TEXT_EXT):
                yield os.path.join(dp, fn)


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def scan(pattern, allowed=()):
    hits = []
    me = os.path.abspath(__file__)
    for p in public_files():
        if os.path.abspath(p) == me or rel(p) in allowed:
            continue
        try:
            lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            continue
        for n, line in enumerate(lines, 1):
            m = pattern.search(line)
            if m:
                hits.append(f"{rel(p)}:{n}: {m.group(0)}")
    return hits


def module_pages(site):
    """{module name: page path} for every API page of one generated site."""
    base = os.path.join(ROOT, site, "api")
    out = {}
    for dp, _, files in os.walk(os.path.join(base, "cowmrp")):
        for fn in files:
            if fn.endswith(".html"):
                r = os.path.relpath(os.path.join(dp, fn), base)[:-5].replace("\\", ".").replace("/", ".")
                out[r[:-len(".index")] if r.endswith(".index") else r] = os.path.join(dp, fn)
    for top in ("cowmrp.html", "index.html"):
        if os.path.isfile(os.path.join(base, top)):
            out.setdefault("cowmrp", os.path.join(base, top))
    return out


def words(text, n):
    text = re.sub(r"`|\*|_{2}|\[|\]|\(|\)|<[^>]+>", " ", text)
    return re.findall(r"[A-Za-z0-9]+", text)[:n]


def page_words(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S)
    return re.findall(r"[A-Za-z0-9]+", html.unescape(re.sub(r"<[^>]+>", " ", t)))


def source_modules():
    src = os.path.join(ROOT, "src")
    for dp, dirs, files in os.walk(os.path.join(src, "cowmrp")):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for fn in files:
            if fn.endswith(".py"):
                p = os.path.join(dp, fn)
                name = os.path.relpath(p, src)[:-3].replace("\\", ".").replace("/", ".")
                yield (name[:-len(".__init__")] if name.endswith(".__init__") else name), p


def module_doc(path):
    import ast
    try:
        return ast.get_docstring(ast.parse(open(path, encoding="utf-8").read())) or ""
    except SyntaxError:
        return ""


def parent_all(path):
    """The names in the parent package's __all__, or None when it has none (pdoc then lists every submodule)."""
    import ast
    init = os.path.join(os.path.dirname(path), "__init__.py")
    if os.path.abspath(init) == os.path.abspath(path) or not os.path.isfile(init):
        return None
    for node in ast.parse(open(init, encoding="utf-8").read()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in node.targets):
            try:
                return set(ast.literal_eval(node.value))
            except ValueError:
                return None
    return None


def intentionally_unpaged(name, path):
    """Modules the API pages leave out on purpose: deprecated aliases, and submodules re-exported by a sub-package's __all__."""
    if module_doc(path).lstrip().upper().startswith(("DEPRECATED", "COWMRP.RUNNER -- DEPRECATED")) or "DEPRECATED" in module_doc(path)[:80].upper():
        return True
    allowed = parent_all(path)
    return allowed is not None and name.count(".") >= 2 and name.rsplit(".", 1)[-1] not in allowed


def check_stale():
    out = []
    for site in SITES:
        pages = module_pages(site)
        if not pages:
            out.append(f"{site}: no API pages found (run the build for this site)")
            continue
        for name, path in source_modules():
            if name.rsplit(".", 1)[-1].startswith("_") and not name.endswith("__init__"):
                continue
            page = pages.get(name)
            if page is None:
                if module_doc(path) and not intentionally_unpaged(name, path):
                    out.append(f"{site}: module {name} has no API page")
                continue
            if os.path.getmtime(path) > os.path.getmtime(page) + 1:
                out.append(f"{site}: {name} page is older than {rel(path)} (rebuild)")
                continue
            if os.path.basename(page) == "index.html" and name == "cowmrp":
                continue                       # the API index page, not the package module page
            doc = module_doc(path)
            w = words(doc.split("\n\n")[0], 6)
            if len(w) >= 3:
                pw = page_words(page)
                if not any(pw[i:i + len(w)] == w for i in range(len(pw))):
                    out.append(f"{site}: {name} page does not start from the current docstring ('{' '.join(w)}')")
    return out


def check_links():
    out = []
    for site in SITES:
        base = os.path.join(ROOT, site)
        if not os.path.isdir(base):
            continue
        for dp, _, files in os.walk(base):
            for fn in files:
                if not fn.endswith(".html"):
                    continue
                p = os.path.join(dp, fn)
                text = open(p, encoding="utf-8", errors="replace").read()
                for t in re.findall(r'(?:href|src)="([^"#?]+)', text):
                    if t.startswith(("http", "mailto", "data:", "//", "javascript")) or "${" in t:
                        continue
                    if not os.path.exists(os.path.normpath(os.path.join(dp, t))):
                        out.append(f"{rel(p)} -> {t}")
    return out


def check_names():
    out = []
    sys.path.insert(0, os.path.join(ROOT, "src"))
    known = set()
    for name, _ in source_modules():
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        known.update(a for a in dir(mod) if a.isupper() or "_" in a)
    ignore = {"COW_TIME_UNIT", "MRP_DOMAIN", "HIDE_VALUES", "KEEP_VALUES"}   # the last two live in guide/web/build_web.py (documentation build), not in the package
    for fn in ("README.md",) + tuple(rel(p) for p in public_files() if p.endswith("README.md")):
        p = os.path.join(ROOT, fn)
        if not os.path.isfile(p):
            continue
        text = open(p, encoding="utf-8", errors="replace").read()
        for tok in sorted(set(re.findall(r"`([A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+)`", text))):
            if tok not in known and tok not in ignore:
                out.append(f"{fn}: `{tok}` is not defined in the package")
    return sorted(set(out))


def check_guide():
    p = os.path.join(ROOT, "guide", "audit_guide.py")
    if not os.path.isfile(p):
        return []
    r = subprocess.run([sys.executable, "-B", p], capture_output=True, text=True, cwd=ROOT)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]
    return [l for l in lines if re.search(r"stale|missing|not found|no longer|does not exist", l, re.I)]


def report(title, items, fails):
    print(f"[{'OK' if not items else ('FAIL' if fails else 'INFO')}] {title}: {len(items)} finding(s)")
    for i in items[:40]:
        print("   ", i)
    if len(items) > 40:
        print(f"    ... {len(items) - 40} more")
    return bool(items) and fails


def main(argv):
    strict = "--strict" in argv
    bad = False
    bad |= report("PRIVATE references in public files", scan(PRIVATE, ALLOWED_PRIVATE), strict)
    print(f"[ALLOWED] functional references to private material ({len(ALLOWED_PRIVATE)} files):")
    for f, why in ALLOWED_PRIVATE.items():
        print("    ", f, "--", why)
    bad |= report("HISTORY wording in public files", scan(HISTORY), strict)
    bad |= report("STALE generated API pages", check_stale(), True)
    bad |= report("LINKS in generated sites", check_links(), True)
    bad |= report("NAMES quoted in README files", check_names(), True)
    bad |= report("GUIDE audit (guide/audit_guide.py)", check_guide(), strict)
    print("DOCUMENTATION_CHECK_" + ("FAILED" if bad else "OK"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
