"""Plotly figures for how the herd distribution evolves (the notebook's Step 10 as functions). Requires ``plotly`` (extra ``plot``)."""
from __future__ import annotations

import numpy as np

from cowmrp.domain import load_domain


def _go():
    try:
        import plotly.graph_objects as go
    except ImportError as e:   # pragma: no cover
        raise ImportError("plotly is required for cowmrp.viz (pip install cowmrp[plot])") from e
    return go


def distribution_by_group(model, dom, never_replaced, replaced):
    """Stacked area of the herd mass per group label over time (reconstructs the undivided distribution from the two replacement-split trajectories)."""
    go, dom = _go(), load_domain(dom)
    fn = dom.hook("group_labels")
    if fn is None:
        raise ValueError(f"domain {dom.name!r} has no group_labels hook")
    labels = np.asarray(fn(model.states)).astype(str)
    x = list(range(len(never_replaced)))
    total = np.asarray(never_replaced) + np.asarray(replaced)                      # (steps + 1, n_states)
    fig = go.Figure()
    for lab in sorted(set(labels)):
        fig.add_trace(go.Scatter(x=x, y=total[:, labels == lab].sum(axis=1), mode="lines", stackgroup="one", name=lab,
                                 hovertemplate=f"{dom.time_unit.capitalize()} %{{x}}<br>%{{y:.3f}}<extra></extra>"))
    fig.update_layout(title=f"Herd distribution by {dom.hook('GROUP_NAME', 'group')} ({dom.time_unit} resolution, {len(x) - 1} steps)",
                      xaxis_title=f"{dom.time_unit.capitalize()}s elapsed", yaxis_title="Probability mass", template="plotly_white", height=420)
    return fig


def replacement_split_figure(dom, never_replaced, replaced):
    """Never-replaced versus replaced-at-least-once share of the herd over time."""
    go, dom = _go(), load_domain(dom)
    x = list(range(len(never_replaced)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=np.asarray(never_replaced).sum(axis=1), mode="lines", name="Regular (never replaced)"))
    fig.add_trace(go.Scatter(x=x, y=np.asarray(replaced).sum(axis=1), mode="lines", name="Replaced at least once"))
    fig.update_layout(title=f"Replacement-aware population split ({dom.time_unit} resolution)", xaxis_title=f"{dom.time_unit.capitalize()}s elapsed",
                      yaxis_title="Probability mass", template="plotly_white", height=420)
    return fig
