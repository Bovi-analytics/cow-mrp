"""One cow, or cow A against cow B -- expected value over two years, and where each cow is likely to be.

    python examples/cow_comparison_example.py

Two second-lactation cows 150 days in milk, one open and one already 60 days pregnant, are projected forward with the model's own
transition matrix and rewards. Change the scenarios (lactation, dim_days, dp_days, time_unit) to compare other cows.
"""
from cowmrp.domains.cow.projection import compare_cows

scenarios = [
    dict(label="open", time_unit="month", lactation=2, dim_days=150, dp_days=0),
    dict(label="pregnant", time_unit="month", lactation=2, dim_days=150, dp_days=60),
]
results = compare_cows(scenarios, years=2)          # {scenario index: result}; n_trajectories=100 adds Monte Carlo bands

print("Expected value over the next 2 years (USD, discounted):")
for i, sc in enumerate(scenarios):
    print(f"  {sc['label']:9s} {results[i]['finite_horizon_value']:8.0f}")
