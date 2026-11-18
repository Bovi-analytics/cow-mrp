"""cow/evidence -- literature values transcribed for the lookup tables, each with its provenance.

One module per source. Nothing here computes a model quantity; the probability modules turn these
numbers into native-month tables (with a `basis` label per row) and the single time-discretization rule
turns those into per-step probabilities. Evidence classes used in the notes:
  A exact (a value printed in the paper)   B adapted (a documented transformation of printed values)
  C insight only (shape/direction, no number used)   D grouped/repeated (one value for several lactations)
"""
