"""guide/check_links.py -- broken-link check: every href/src in guide/site/*.html that points at a local file must exist. Run: python guide/check_links.py"""
import os
import re

SITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site")
checked = broken = 0
for dirpath, _, files in os.walk(SITE):
    for fn in files:
        if not fn.endswith(".html"):
            continue
        p = os.path.join(dirpath, fn)
        text = open(p, encoding="utf8").read()
        for target in re.findall(r'(?:href|src)="([^"#?]+)', text):
            if target.startswith(("http", "mailto", "data:", "//", "javascript")) or "${" in target:   # '${url}' is a template string inside pdoc's script
                continue
            checked += 1
            if not os.path.exists(os.path.normpath(os.path.join(dirpath, target))):
                broken += 1
                print("BROKEN:", os.path.relpath(p, SITE), "->", target)
print(f"{checked} local links checked, {broken} broken")
