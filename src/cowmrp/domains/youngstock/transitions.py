"""
youngstock/transitions.py
=========================
Builds the youngstock transition matrix Q from the states table and the
probabilities. Q is ROW-STOCHASTIC (absorbing outcomes self-loop with 1), so
it is validated by mrp_core.validate_row_stochasticity exactly like the cow
matrix -- births are NOT in Q; they enter as an external inflow
(see farm_population.py:  y_{t+1} = y_t Q + births x birth_mix).

Per-step branch order (matches cow/transitions.py):
  CALF/HEIFER (DAIRY_F): die | age +1 (or -> BREEDING at K_BREED)
  BREEDING            : removal(CULLED) | conceive -> PREGNANT dp=1 | stay
  PREGNANT dp         : removal(CULLED) | abort -> BREEDING | calve -> GRADUATED
                        | dp+1
  terminal CALF       : age +1, sold DURING age step K_SALE[line] -> SOLD
GRADUATED means "reached first calving and is available to enter the cow
herd as L1"; the coupling (closed vs open herd) is decided by the Runner,
never here.
"""


from cowmrp.domains.youngstock.parameters import K_BREED, K_SALE, DP_CALVE, BREED_CROSSING, TERMINAL_CROSSING, crossing_conditional


def compute_transitions(states, probs):
    """Returns (rows, cols, vals) lists for mrp_core.assemble_matrix."""
    line = states['line'].values
    stage = states['stage'].values
    age = states['age'].values
    dp = states['dp'].values
    sid = states['state_id'].values

    # (line, stage, age, dp) -> state_id, built once (states table is the
    # single source of truth for what exists; a missing key raises KeyError)
    index = {(l, s, a, d): int(i) for l, s, a, d, i in
             zip(line, stage, age, dp, sid)}
    dairy_f_stage_at_age = {int(a): s for l, s, a in zip(line, stage, age)
                            if l == 'DAIRY_F' and s in ('CALF', 'HEIFER')}

    def target(l, s, a=0, d=0):
        return index[(l, s, a, d)]

    rows, cols, vals = [], [], []

    def add(src, dst, p):
        if p > 0:
            rows.append(src); cols.append(dst); vals.append(p)

    died, culled = target('DAIRY_F', 'DIED'), target('DAIRY_F', 'CULLED')
    graduated, breeding = target('DAIRY_F', 'GRADUATED'), target('DAIRY_F', 'BREEDING', 0, 0)

    for i in range(len(states)):
        l, s, a, d, src = line[i], stage[i], age[i], dp[i], sid[i]

        if s in ('DIED', 'CULLED', 'GRADUATED', 'SOLD', 'STILLBORN'):
            add(src, src, 1.0)                                   # absorbing

        elif l == 'DAIRY_F' and s in ('CALF', 'HEIFER'):
            p_die = probs['p_die'][i]
            q = crossing_conditional(BREED_CROSSING, a + 1)          # reaches breeding age at boundary a+1
            add(src, died, p_die)
            add(src, breeding, (1 - p_die) * q)
            if q < 1.0:
                add(src, target('DAIRY_F', dairy_f_stage_at_age[a + 1], a + 1, 0), (1 - p_die) * (1 - q))

        elif l == 'DAIRY_F' and s == 'BREEDING':
            p_ex, p_c = probs['p_exit_breeding'][i], probs['p_conceive'][i]
            add(src, culled, p_ex)
            add(src, target('DAIRY_F', 'PREGNANT', 0, 1), (1 - p_ex) * p_c)
            add(src, src, (1 - p_ex) * (1 - p_c))

        elif l == 'DAIRY_F' and s == 'PREGNANT':
            p_ex, p_ab, p_cv = probs['p_exit_pregnant'][i], probs['p_abort'][i], probs['p_calve'][i]
            surv = 1 - p_ex
            add(src, culled, p_ex)
            add(src, breeding, surv * p_ab)                       # pregnancy loss -> open
            add(src, graduated, surv * (1 - p_ab) * p_cv)         # first calving
            if d < DP_CALVE:
                add(src, target('DAIRY_F', 'PREGNANT', 0, d + 1), surv * (1 - p_ab) * (1 - p_cv))
            elif surv * (1 - p_ab) * (1 - p_cv) > 1e-12:
                raise ValueError("PREGNANT at DP_CALVE must calve with probability 1")

        elif s == 'CALF':                                         # terminal lines
            q = crossing_conditional(TERMINAL_CROSSING[l], a + 1)    # sale boundary a+1
            add(src, target(l, 'SOLD'), q)
            if q < 1.0:
                add(src, target(l, 'CALF', a + 1, 0), 1.0 - q)

        else:
            raise ValueError(f"unhandled youngstock state ({l},{s},{a},{d})")

    return rows, cols, vals

