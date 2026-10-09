"""
audit_milk.py
=============
Milk-curve audit (audit P6): for lactations 1..MAX_LACTATION under BOTH curve methods print peak DIM, peak milk,
milk at DIM 1 / 5 / 30, at 210 d and at the dry-off DIM (DRY_OFF_DP_DAYS before calving at the model's mean
calving interval), 305-day milk, and the behaviour at DIM 0 and at the end of lactation. The Wood parameters a, peak
and time-to-peak come from Evangelista et al. (2024) (lactations 1-3, a non-US population); b and c are derived
(cow/parameters.py _WOOD_SOURCE). MilkBot values are the previous unverified placeholders.

    python verify/audit_milk.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cow.parameters as P
from cow.curves import curve_kg

def row(method, L):
    t = np.arange(0, 600)
    y = curve_kg(t.astype(float), L, method)
    pk = int(np.argmax(y))
    return dict(peak_dim=pk, peak_kg=y[pk], kg1=curve_kg(1.0, L, method), kg5=curve_kg(5.0, L, method), kg30=curve_kg(30.0, L, method),
                kg210=y[210], kg_dry=y[min(int(P.GESTATION_LENGTH_DAYS + 140 - P.DRY_OFF_DAYS_BEFORE_CALVING * 1), 599)], total305=float(y[:305].sum()),
                kg0=float(curve_kg(0.0, L, method)))

print(f"{'method':7s} {'lac':>3s} {'peakDIM':>7s} {'peak':>6s} {'DIM0':>5s} {'DIM1':>5s} {'DIM5':>5s} {'DIM30':>6s} {'DIM210':>7s} {'305-d kg':>9s} {'basis'}")
for method in ("wood", "milkbot"):
    for L in range(1, P.MAX_LACTATION + 1):
        r = row(method, L)
        basis = P.MILK_CURVE_BASIS[method][L]
        print(f"{method:7s} {L:>3d} {r['peak_dim']:>7d} {r['peak_kg']:6.1f} {r['kg0']:5.1f} {r['kg1']:5.1f} {r['kg5']:5.1f} {r['kg30']:6.1f} {r['kg210']:7.1f} {r['total305']:9.0f} {basis}")
