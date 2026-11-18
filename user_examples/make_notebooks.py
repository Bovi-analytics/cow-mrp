"""Turns each NN_*.py example (cells marked with '# %%', markdown cells with '# %% [markdown]') into a Jupyter notebook next to it.

    python make_notebooks.py        (needs nbformat; the .ipynb files are already included, rerun after editing a .py file)

The .py file is the source of truth. Notebooks are saved without outputs.
"""
from pathlib import Path

import nbformat

HERE = Path(__file__).parent


def cells_of(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        elif body.strip():
            if body.lstrip().startswith("# %pip"):                      # the install cell: '# %pip ...' in the .py file, '%pip ...' in the notebook
                cell = nbformat.v4.new_code_cell(body.replace("# %pip", "%pip", 1))
                cell.metadata["tags"] = ["pip-install"]                  # check_installed.py skips this cell (the wheel is already installed)
                cells.append(cell)
            else:
                cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


for path in sorted(HERE.glob("[0-9][0-9]_*.py")):
    nb = nbformat.v4.new_notebook(cells=cells_of(path.read_text(encoding="utf-8")))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(nb, path.with_suffix(".ipynb"))
    print("wrote", path.with_suffix(".ipynb").name)
