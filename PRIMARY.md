# PRIMARY.md — Comprehensive Project Design Record for the MRP Engine

> **Reformulated 2026-10-08 (CowMRP fork).** The state is now only (lactation, DIM, DP). Health, the 6-level milk bucket, the DNB label/probability and the stored `flag` were removed; the milk-floor reproductive cull (`cow/tables.py`, `culling_repro`) replaces health/milk/DNB culling. Wherever text below describes those removed concepts it describes the PRE-reformulation model and is superseded -- see `docs/MODEL_CHANGELOG.md` (2026-10-08) and `docs/ASSUMPTIONS_REGISTER.md` A-063..A-067.

> **Status:** Primary design/context document.
>
> This document consolidates the necessary architecture, requirements, refinements, examples, constraints, and unresolved decisions contained in the supplied project discussions and versions. It is intended to be the detailed source of project context. `CLAUDE.md` should remain the concise operational implementation rulebook.

---

# 1. Purpose and Scope

The project is intended to become a **generalizable Markov Reward Process (MRP) engine**.

The first implementation/domain is dairy cattle, referred to as the **Cow domain**. The Cow domain is an important use case and source of requirements, but:

> **THE COW DOMAIN IS THE FIRST DOMAIN USING THE MRP ENGINE. IT IS NOT THE DEFINITION OF THE MRP ENGINE.**

The long-term goal is:

```text
Generic MRP Core
+
Generic MRP Engine
+
Domain-specific implementation
=
Reusable, extensible MRP system
```

Possible future domains discussed include sheep, beef cattle, nutrition/CNCPS, manufacturing/CNC machines, and other state-transition systems. These illustrate intended generality; they are not immediate implementation requirements.

The current project is an **MRP**, not an MDP. It consists of:

```text
States
+
Transition probabilities
+
Rewards
```

It does not currently include actions, policies, action selection, control decisions, or optimization over actions. Introducing MDP or reinforcement-learning control logic requires an explicit architectural decision.

---

# 2. Role of PRIMARY.md and CLAUDE.md

## PRIMARY.md

`PRIMARY.md` is the detailed project/design record. It should preserve:

- project goals;
- historical decisions and their rationale;
- architecture;
- modeling discussions;
- scientific assumptions;
- requirements;
- examples and edge cases;
- unresolved questions;
- intended future capabilities;
- decisions that must not be silently contradicted.

## CLAUDE.md

`CLAUDE.md` is the operational rulebook. It should tell an implementation agent:

- what rules to follow;
- where functionality belongs;
- how dependencies are validated;
- how time is handled;
- how errors are raised;
- how models are added.

`CLAUDE.md` should not simply duplicate this document.

If `PRIMARY.md`, `CLAUDE.md`, and current code conflict:

1. identify the conflict;
2. do not silently redesign the architecture;
3. explain the alternatives or inconsistency;
4. ask before making a major architectural decision.

---

# 3. Core Architectural Model

The intended conceptual architecture is:

```text
Runner / Notebook / User Configuration
                |
                v
+------------------------------------------+
| DOMAIN LAYER                             |
|                                          |
| states                                   |
| parameters                               |
| derived variables                        |
| probabilities                            |
| rewards                                  |
| transitions                              |
| lookup tables / equations / models       |
| domain interpretation                    |
+-------------------+----------------------+
                    |
                    | definitions, data, contracts
                    v
+------------------------------------------+
| GENERIC MRP ENGINE                       |
|                                          |
| registration                             |
| dependency validation                    |
| model resolution                         |
| state-space orchestration                |
| transition construction                  |
| reward orchestration                     |
| model completeness validation            |
| simulation orchestration                 |
+-------------------+----------------------+
                    |
                    v
+------------------------------------------+
| MRP CORE                                 |
|                                          |
| MRP mathematics                          |
| matrices                                 |
| Bellman/value evaluation                 |
| discounting                              |
| state-space utilities                    |
| distribution evolution                   |
| generic stochastic simulation            |
+------------------------------------------+
```

The governing principle is:

> **Generic layers provide mechanisms. Domains provide meaning.**

## 3.1 Implementation Mapping (current)

The three layers above are implemented as:

```text
DOMAIN LAYER       -> cow/ (parameters.py, states.py, transitions.py,
                       probabilities/, rewards/)
GENERIC MRP ENGINE  -> mrp_engine.py
MRP CORE            -> mrp_core.py
```

The Engine layer is implemented as **plain functions operating on plain
dicts/DataFrames** -- not a class, not a registry object, not a plugin
framework. `mrp_engine.py` may import from `mrp_core.py`; `mrp_core.py`
must never import from `mrp_engine.py` or from any domain package. This
constraint is permanent, not a migration-phase simplification -- see §35.

---

# 4. Architectural Boundaries

## 4.1 MRP Core

MRP Core may contain:

- generic state data structures;
- state-space construction utilities;
- transition matrices;
- sparse/dense matrix operations;
- probability validation;
- reward vectors;
- finite-horizon evaluation;
- infinite-horizon evaluation;
- discounting;
- distribution evolution;
- stochastic sampling;
- generic simulation;
- generic dependency/model validation mechanisms where appropriate.

MRP Core must not contain Cow-specific knowledge such as:

- lactation;
- DIM;
- pregnancy;
- calving;
- milk;
- mastitis;
- lameness;
- culling;
- dairy/feed prices;
- biological constants;
- Cow-specific equations;
- Cow-specific lookup tables.

### 4.1.1 Current `mrp_core.py` contents

`lookup`, `assemble_matrix`, `validate_row_stochasticity`, `get_transitions`,
`annual_to_per_step_gamma`, `steps_per_year`, `evaluate_finite_horizon`,
`evaluate_infinite_horizon`, `find_state_id`, `state_values`,
`state_value_diff`, `make_initial_distribution`,
`evaluate_state_trajectory`, `evaluate_state_trajectory_to_convergence`,
`distribution_recursive`, `distribution_matrix_power`, `distribution_eigen`,
`export_transitions`, `export_rewards`, `precompute_transition_lists`,
`simulate_stochastic_trajectory`, `simulate_monte_carlo`.

Note: `build_states` and `resolve_method` live in `mrp_engine.py`, not here
-- despite sounding like Core, both orchestrate the spec-shaped
declare/validate/resolve workflow (§6.2, §11), which this document's own
§3 diagram already places in the Engine box.

## 4.2 Generic MRP Engine

The Engine may provide generic orchestration for:

- registering state definitions;
- validating state and model dependencies;
- registering probability specifications;
- registering reward specifications;
- resolving table/formula/model implementations;
- checking required inputs;
- constructing transitions;
- constructing rewards;
- validating model completeness;
- running MRP calculations;
- running generic simulations.

The Engine must not hard-code Cow behavior.

It should know:

> “This model requires X, Y, and Z.”

It should not know:

> “X means days in milk.”

The domain defines meaning; the Engine validates and connects generic representations.

### 4.2.1 Current `mrp_engine.py` contents

**Implemented:**
- `build_states(variable_domains, derived_variables)` -- state-space construction
- `resolve_method(spec)` -- method resolution, no silent fallback
- `run_specs(specs, states, context, has_sign)` -- the shared spec executor
  used identically by probability and reward orchestration (§10.1, §17).
  Also applies a post-computation NaN safety net (raises
  `LookupCoverageError`) as its lookup-coverage check.
- `validate_dependencies(spec, available)` / `MissingDependencyError` --
  checks a spec's declared `depends_on` against what's actually available.
  Accepts either the typed-dict shape or a flat list (a flat list is
  treated as an all-`state` shorthand, which is what probability specs
  currently use -- see §10.1). `parameter` availability is read directly
  from the domain's parameters module namespace (e.g.
  `{n for n in dir(cow.parameters) if n.isupper()}`) -- no separate
  parameter registry is built. `external_input` availability is an empty
  set for now (nothing in the repo declares one yet); a spec that did
  declare an `external_input` dependency today would correctly fail
  validation rather than silently pass.
- `validate_lookup_coverage(table, key_columns, spec_name)` /
  `LookupCoverageError` -- structural check only (required key columns
  present). Full combination coverage is enforced by `run_specs`' NaN
  check above, not by eager enumeration of every combination.
- `validate_model_completeness(states, probability_specs, reward_specs,
  available)` -- runs `validate_dependencies` + `resolve_method` for
  every given spec before numeric computation (the §18 validation
  report). Invoked as a pre-flight check at the start of
  `cow/probabilities/__init__.py::compute_probabilities()` (validating
  `probability_specs`) and `cow/rewards/__init__.py::compute_rewards()`
  (validating `reward_specs`, using the real, already-computed `probs`
  dict keys -- including the Domain-composed `p_cull` -- as the
  available `probability_output` set).

## 4.3 Domain Layer

The domain owns:

- state definitions;
- variable meanings;
- domain parameters;
- derived variables;
- biological/economic equations;
- lookup tables;
- probability models;
- reward models;
- domain transition rules;
- assumptions;
- scientific interpretation.

For Cow, possible organization discussed includes:

```text
cow/
    parameters.py
    states.py
    probabilities.py
    rewards.py
    transitions.py
```

This is a design direction, not a mandate to reorganize existing code before inspection.

**Current implementation:** this is now the actual structure, except
`probabilities.py`/`rewards.py` are packages (`probabilities/`, `rewards/`)
with one spec per file plus an `__init__.py` orchestrator, not single
files. Every culling-risk submodule -- including the DNB-past-cutoff rule
-- is a formal spec registered in `probability_specs` (`culling_dnb` was
previously a bare function called outside the spec system entirely).

---

# 5. Genericity Test

Before adding functionality to generic Core or Engine code, ask:

> **Could this mechanism work for a completely different domain without changing the generic code?**

Examples:

```text
Discount a reward stream                  -> Generic
Build a transition matrix                -> Generic
Validate required model inputs           -> Generic
Resolve a selected model                 -> Generic

Pregnancy duration                       -> Cow domain
Milk as a function of DIM                -> Cow domain
Abortion biology                         -> Cow domain
Cow-specific culling rule                -> Cow domain
```

When a domain requirement reveals a missing generic capability:

```text
DOMAIN REQUIREMENT
        |
        v
What generic capability is actually needed?
        |
        +-- Domain-specific -> keep in domain
        |
        +-- Generic -> add the smallest generic mechanism
                          |
                          v
                  Domain supplies data/rules
```

Do not solve a Cow requirement by copying Cow logic into generic code.

---

# 6. State System

States are a central extension point.

A domain should be able to declare variables without rewriting generic MRP mathematics.

Conceptually:

```python
variable_domains = {
    "lactation": ...,
    "dim": ...,
    "dp": ...,
    "health": ...,
}
```

The generic Engine should be able to:

1. receive declared state variables;
2. construct candidate state combinations;
3. apply domain constraints;
4. apply derived-variable rules;
5. validate dependencies;
6. validate state uniqueness;
7. assign stable IDs;
8. provide lookup by explicit variable values.

The generic Engine must not assume that any particular Cow variable exists.

## 6.1 Independent, derived, and produced variables

A variable may be:

1. directly supplied;
2. derived from other variables;
3. produced by another model/function.

Derived variables should declare dependencies when useful:

```python
{
    "name": "pregnancy_status",
    "depends_on": ["dp"],
    "function": derive_pregnancy_status,
}
```

The Engine validates that required inputs exist before evaluation.

If ordered rules are intended to assign one result, detect:

- zero matches;
- invalid dependencies;
- ambiguous matches;

unless multiple matches are explicitly permitted.

The domain defines biological interpretation.

For example, a domain may define how `dp` maps to Open, Pregnant, or Dry. The generic system must not infer that meaning.

## 6.2 State-space construction

Conceptually:

```text
declared independent variables
        |
        v
Cartesian / constrained combinations
        |
        v
derived variables
        |
        v
state validation
        |
        v
final state table
        |
        v
stable state IDs
```

State IDs should be generated rather than manually maintained.

The system should support:

```text
find_state_id(...)
```

based on explicit state-variable values.

For a given state-space construction, IDs should remain stable.

---

# 7. Seamless State Addition

Adding a state must not force unrelated models to change.

Example:

```text
Existing:
DIM
Lactation
Health

Models:
Mortality -> Health
Conception -> DIM, Lactation
```

Add:

```text
BodyWeight
```

Result:

```text
Mortality -> unchanged
Conception -> unchanged
BodyWeight -> available for future models
```

The presence of a variable does **not** mean every probability or reward must consume it.

The dependency direction is:

```text
Model declares what it needs
        |
        v
Engine checks whether it exists
```

not:

```text
Every available state -> force into every model
```

---

# 8. The Complete Model Input / Dependency System

This is a critical requirement.

Do **not** think only in terms of “state dependencies.” A model may need several kinds of inputs.

Dependencies may include:

```text
STATE
DERIVED STATE
PARAMETER
EXTERNAL INPUT
LOOKUP TABLE / DATA
MODEL OUTPUT
TIME VARIABLE / TIME METADATA
```

Example:

```text
ABORTION MODEL

State variables:
    DIM
    Lactation
    Pregnancy / DP

External inputs:
    HerdAbortionRate
    Season

Parameters:
    GestationLength

Lookup data:
    AbortionProbabilityTable
```

Therefore, the general concept is:

> **Model Inputs / Dependencies**

rather than only:

> **State Dependencies**

Conceptually:

```text
MODEL
  |
  v
DEPENDENCY CONTRACT
  |
  +--> states
  +--> derived states
  +--> external inputs
  +--> parameters
  +--> lookup data
  +--> outputs of other models
  +--> time information
  |
  v
VALIDATION
  |
  +--> complete -> execute
  |
  +--> missing -> clear actionable error
```

The same mechanism should support both probability and reward models.

## 8.1 Required versus optional inputs

Where useful:

```python
{
    "depends_on": ["dim", "lactation"],
    "optional_inputs": ["health"],
}
```

Required inputs must exist.

Optional inputs may be absent only if the model explicitly defines what happens when they are absent.

Do not make dependencies optional merely to suppress errors.

## 8.2 Required versus available

The fundamental rule is:

```text
AVAILABLE INFORMATION >= MODEL REQUIREMENTS
```

Extra available information is valid.

Example:

```text
Available:
DIM, Lactation, Pregnancy, Health, Milk,
BodyWeight, DMI, SCC, Activity

Abortion model requires:
DIM, Lactation, Pregnancy

Result: VALID
```

The model uses only what it declares.

## 8.3 Explicit dependency names

Do not silently perform approximate matching.

```text
pregnancy_days != dp
health != health_score
health != health_status
```

unless the domain explicitly registers equivalence or a transformation.

This prevents subtle scientific errors.

## 8.4 Actionable dependency errors

Errors should answer:

1. What was requested?
2. Which model?
3. What inputs are required?
4. What is available?
5. What is missing?
6. Which implementation was searched?
7. What can the user do?

Example:

```text
MissingDependencyError

Probability: abortion_probability

Required:
    dp
    dim
    lactation

Available:
    dim
    health
    age

Missing:
    dp
    lactation

Action:
    Add the missing inputs or select/provide a compatible model.
```

Avoid obscure errors such as:

```text
KeyError: dp
```

when a meaningful model error can be raised.

---

# 9. Dependency Graph Concept

The project may conceptually form dependency graphs:

```text
STATE / INPUT SPACE
        |
        +--> Probability Model
        |          |
        |          v
        |      Transition
        |
        +--> Reward Model
                   |
                   v
                 Reward

        \______________/
                |
                v
          MRP STRUCTURE
             P + R
                |
                v
          MRP evaluation
```

Another example:

```text
DIM + Lactation + DP
        |
        v
Abortion Probability
        |
        v
Abortion Event
        |
        v
Reproduction Reward
```

This graph is conceptual.

Do not build a complex graph framework unless the actual project requires one.

---

# 10. Probability Model Architecture

Every probability should have an explicit specification rather than scattered hard-coded logic.

Conceptually:

```python
probability_spec = {
    "name": "abortion_probability",
    "depends_on": ["dp", "dim", "lactation"],
    "optional_inputs": [],
    "method": "table",
    "methods": {
        "table": abortion_table_lookup,
        "formula": abortion_formula,
        "model": None,
    },
    "time_unit": "month",
    "source": "...",
    "description": "...",
}
```

Exact APIs may evolve. The important contract is:

- explicit identity;
- explicit inputs;
- explicit method;
- explicit available implementations;
- explicit metadata where relevant.

Possible probability approaches include:

- lookup tables;
- empirical probabilities;
- conditional probabilities;
- state-dependent probabilities;
- time-dependent probabilities;
- parametric functions;
- statistical/ML models;
- directly supplied probabilities;
- probability matrices;
- other explicitly supported methods.

Not every probability must implement every method.

Do not invent scientific models merely to make the system appear complete.

Scientific models must come from:

- user specification;
- documented project decisions;
- this design record;
- documented scientific sources;
- explicit requests;
- scientifically justified and explicitly approved assumptions.

---

# 11. Probability Method Resolution

A selected method must actually exist.

Example:

```text
Requested method: model
Registered implementation: None
```

This should raise an informative error, such as:

```text
ModelResolutionError:
Probability 'abortion_probability' is configured to use method 'model',
but no implementation is registered.

Available:
    table
    formula
```

Do not silently fall back:

```text
Requested: model
Missing: model
Automatically use: table
```

unless explicit fallback behavior was configured.

A missing implementation is a model-completeness/configuration error.

A simple resolver is acceptable; a plugin framework is not required.

## 10.1 Implemented Spec Shape

**Probability specs** (e.g. `abortion_spec`) carry `produces`, but
intentionally keep the **flat list** `depends_on` shape -- this is a
deliberate scope decision, not a pending step:

```python
probability_spec = {
    "name": "abortion",
    "produces": "p_abort",
    "depends_on": ["dp"],
    "method": "table",
    "methods": {"table": p_abortion_table, "formula": p_abortion_formula, "model": None},
}
```

For a multi-output spec (the health-transition spec is the one current
example), `produces` is a list of names rather than a single string:

```python
"produces": ["p_improve", "target_improve", "p_worsen", "target_worsen", "p_remain"],
```

**Reward specs** (e.g. `milk_income_spec`) use the **typed dict**
`depends_on` shape and do not carry `produces`:

```python
reward_spec = {
    "name": "milk_income",
    "sign": +1,
    "depends_on": {
        "state": ["dim", "lactation", "health", "flag"],
        "derived_state": [],
        "probability_output": [],
        "reward_output": [],
        "parameter": [],
        "external_input": [],
        "model_output": [],
        "time": [],
    },
    "method": "formula",
    "methods": {"table": None, "formula": milk_income_formula, "model": None},
}
```

`mrp_engine.validate_dependencies` accepts both shapes: a flat list is
treated as an all-`state` shorthand, so probability specs validate
correctly without the typed-dict conversion. Converting probability
specs to the typed-dict shape remains a possible future step, not a
currently pending one.

`produces` names the array (or, for multi-output specs, the list of
names) a probability spec's active method returns -- it lets the Engine
register the value as an available `probability_output` for later specs
to depend on. **Reward specs do not need `produces`**: `compute_rewards()`
already keys results by `spec['name']` directly, so no naming mismatch
exists there (see §16).

The eight `depends_on` kinds map onto this document's earlier conceptual
dependency categories (§8) as follows: `state`/`derived_state` -> states /
derived states; `probability_output`/`reward_output`/`model_output` ->
"outputs of other models"; `parameter`/`external_input` -> parameters /
external inputs; `time` -> time information. Lookup-table structure is
not itself a separate dependency kind -- it is part of a spec's chosen
`method` implementation.

---

# 12. Model Compatibility and Compatible Model Discovery

Compatibility should not be based only on model name.

Where applicable, compatibility should consider:

- required inputs;
- state variables;
- units;
- time unit;
- expected output;
- selected method;
- lookup keys;
- valid ranges/domain.

The generic system checks structural compatibility.

The domain is responsible for scientific meaning.

A future-friendly behavior is to identify compatible registered implementations.

Example:

```text
Requested:
    abortion_probability

Available inputs:
    DIM
    Lactation
    Pregnancy

Registered implementations:

Model A:
    DIM, Lactation, Pregnancy
    -> compatible

Model B:
    DIM, Lactation, Pregnancy
    -> compatible

Model C:
    DIM, Lactation, Pregnancy, BodyWeight
    -> incompatible
       Missing: BodyWeight
```

The user can then choose among compatible models.

This is desirable functionality, but does not require a complex discovery framework immediately.

---

# 13. Lookup Tables

Lookup tables are models/data with explicit contracts.

A table must declare the inputs/keys it requires.

Example:

```text
DIM | Lactation | Pregnancy | Abortion Probability
```

The table contract may require:

```text
DIM
Lactation
Pregnancy
```

The system validates that these dimensions exist and that the table actually covers the combinations the selected model requires.

## 13.1 Key-based matching

Scientific lookup must use explicit key-based matching.

Do not rely on positional row assumptions such as:

```text
iloc
```

as the scientific lookup mechanism.

Prefer explicit keys, merges, joins, or equivalent mechanisms.

## 13.2 Lookup validation

Validate where applicable:

- required columns;
- key names;
- units;
- duplicate keys;
- missing combinations;
- unsupported values;
- interpolation behavior;
- extrapolation behavior.

## 13.3 Dimension coverage

If a model declares:

```text
DIM
Lactation
Health
```

but the table only provides:

```text
DIM
Lactation
```

then the table is incomplete for that contract.

Raise an explicit coverage error.

## 13.4 Combination coverage

For discrete states:

```text
DIM:        0, 30, 60
Lactation:  1, 2
Health:     Healthy, Sick
```

There are:

```text
3 × 2 × 2 = 12
```

potential combinations.

If the lookup provides only 10 and no explicit interpolation/default/extrapolation behavior covers the missing two:

> **ERROR**

The missing combinations should be identified.

Never silently:

- use zero;
- convert NaN to zero;
- use an undocumented average;
- use a nearest value;
- extrapolate;

unless the domain/model explicitly declares that behavior.

Missing scientific information must remain visible.

---

# 14. Transition System

Transitions and probabilities are conceptually separate.

```text
Domain state definitions
        |
        v
Domain transition rules
        |
        v
Probability models
        |
        v
Transition probabilities
        |
        v
Generic transition-matrix assembly
        |
        v
MRP Core
```

Transition code should not duplicate the biological equations that produce probabilities.

The generic matrix assembler consumes transition probabilities and state information; it should not know what conception, abortion, replacement, or another event means.

---

# 15. Transition Matrix Validation

The transition matrix should represent a valid stochastic process.

Validate where appropriate:

- probability ranges;
- non-negativity;
- valid state IDs;
- existing transition targets;
- declared absorbing states;
- row stochasticity.

For every state:

```text
sum(P[i, :]) ≈ 1
```

If invalid, report the problematic state(s).

Do not silently normalize rows unless explicit normalization behavior was requested.

---

# 16. Reward Architecture

Rewards use the same dependency-contract philosophy as probabilities,
using the `depends_on` typed-dict shape described in §10.1 -- with one
difference: reward specs do not need a `produces` field, because
`compute_rewards()` already keys results by `spec['name']` directly.

Example:

```python
milk_income_spec = {
    "name": "milk_income",
    "depends_on": ["milk", "milk_price"],
    "method": "formula",
    "methods": {
        "table": None,
        "formula": milk_income_formula,
        "model": None,
    },
}
```

The Engine validates:

```text
milk exists
milk_price exists
selected implementation exists
```

Possible Cow reward components include:

- production/milk income;
- pregnancy/reproductive value;
- health cost;
- feed cost;
- treatment cost;
- replacement cost;
- culling/salvage value;
- mortality cost;
- reproduction cost.

These are domain concepts.

The generic Engine receives, validates, resolves, calculates, and aggregates them.

## 16.1 Modular reward components

Conceptually:

```text
Total Reward
├── Milk Income
├── Pregnancy / Reproduction Value
├── Cull / Salvage Value
├── Mortality Cost
├── Replacement Cost
├── Feed Cost
├── Health Cost
└── Reproduction Cost
```

Each component should identify, where relevant:

- name;
- sign;
- dependencies;
- selected method;
- implementation;
- formula/table/model;
- units;
- parameters;
- assumptions;
- source;
- applicable conditions;
- aggregation method.

## 16.2 Avoid double counting

When adding a component, explicitly check whether the same economic transaction is already represented elsewhere.

For example:

```text
Cull Value
Replacement Cost
Salvage Value
Culling Cost
```

must not accidentally count the same transaction twice.

The economic interpretation belongs to the domain.

If the accounting relationship is unclear, ask rather than silently deciding.

---

# 17. Model Registration and Adding Models

A simple generic registry/specification mechanism is preferred where sufficient.

Conceptually:

```text
Probability Registry
├── mortality_probability
├── culling_probability
├── conception_probability
├── abortion_probability
└── ...

Reward Registry
├── milk_income
├── feed_cost
├── health_cost
└── ...
```

The registry mechanism is generic. Entries are domain-specific.

**Implemented as:** `cow/probabilities/__init__.py` and `cow/rewards/__init__.py`
each hold a flat `probability_specs` / `reward_specs` list of spec dicts,
executed by the single shared `mrp_engine.run_specs(specs, states, context,
has_sign)` function -- no separate registry object, no class. Both
orchestrator files call the same function; the probability-vs-reward
distinction is just the `has_sign` flag and what `context` holds (§10.1).
(`compute_rewards()` calls `run_specs` more than once -- one call over
`BASE_REWARD_SPECS`, then one call per spec in `ENVIRONMENTAL_REWARD_SPECS`
-- to keep environmental components always-visible without folding them
into `total` unconditionally; see §44. `reward_specs` itself, the flat
concatenation of both lists, still exists for anything that wants the
full set.)

## Adding a probability

```text
1. Define probability
2. Declare dependencies
3. Provide table/formula/model
4. Register it
5. Validate dependencies
6. Resolve selected method
7. Validate coverage/compatibility
8. Make available
```

## Adding a reward

```text
1. Define reward
2. Declare dependencies
3. Provide table/formula/model
4. Register it
5. Validate dependencies
6. Resolve selected method
7. Make available
```

Adding one model should not require rewriting unrelated generic MRP mathematics or unrelated Engine logic.

Do not build an elaborate plugin ecosystem unless the project genuinely requires one.

---

# 18. Model Completeness and Validation

Model completeness is a first-class concept.

Before serious simulation, the system should be able to answer:

- Are states defined?
- Are derived-variable dependencies satisfied?
- Are all required inputs available?
- Are selected probability methods implemented?
- Are selected reward methods implemented?
- Are lookup tables complete?
- Are units compatible?
- Are time units compatible?
- Are transitions valid?
- Are reward outputs valid?
- Is the transition matrix stochastic?

A conceptual report:

```text
MODEL VALIDATION REPORT

States:
    ✓ dim
    ✓ lactation
    ✓ health
    ✓ dp

Probabilities:
    ✓ mortality
    ✓ conception
    ✗ abortion

Rewards:
    ✓ milk_income
    ✗ feed_cost

Missing:
    abortion -> missing dp
    feed_cost -> missing dmi

Action:
    Add the missing state/input/model/table/function,
    or select a compatible implementation.
```

Validation should be re-runnable. After missing components are added, the model should become valid without unrelated code changes.

## Preferred validation sequence

```text
1. Configuration validation
2. State definition validation
3. Derived-state dependency validation
4. State-space construction
5. Probability dependency validation
6. Probability method resolution
7. Lookup/model coverage validation
8. Reward dependency validation
9. Reward method resolution
10. Transition construction
11. Row-stochasticity validation
12. MRP evaluation / simulation
```

Fail as early as practical.

Validation should be centralized where practical rather than independently and inconsistently reimplemented in:

```text
probabilities.py
rewards.py
transitions.py
simulator.py
```

---

# 19. Time Architecture: One Shared System

Time is a cross-cutting system concept.

> **TIME MUST BE CONSISTENT THROUGHOUT THE ENTIRE SYSTEM.**

Years, months, weeks, days, and model time steps must not be independently interpreted by different files.

A change in model time must propagate consistently through:

- state construction;
- transitions;
- transition probabilities;
- rewards;
- simulations;
- discounting;
- time-dependent parameters;
- lookup tables;
- biological durations;
- age;
- DIM;
- DP;
- lactation timing;
- reproductive timing;
- economic calculations;
- farm tools;
- notebooks;
- benchmark calculations;
- reporting/output.

Changing:

```text
monthly -> weekly
```

or:

```text
monthly -> daily
```

must not require manually finding and fixing unrelated hard-coded assumptions.

---

# 20. Single Source of Truth for Time

There must be one authoritative model-time configuration.

Conceptually:

```python
TIME_UNIT = "month"
```

Possible units may include:

```text
day
week
month
year
```

The actual supported set must be documented by the implementation.

Derived quantities such as:

```text
UNIT_DAYS
steps_per_year
per_step_gamma
```

must derive from the authoritative time definition.

Do not scatter conflicting constants such as:

```python
DAYS_PER_MONTH = 30
```

in one file and:

```python
DAYS_PER_MONTH = 30.44
```

in another.

Time conversions should have one authoritative implementation.

Do not duplicate conversion logic across:

- probabilities;
- rewards;
- transitions;
- simulation;
- notebooks;
- farm tools.

---

# 21. Model Time versus Biological Time

> **APPROVED ARCHITECTURE DECISION (supersedes this section's earlier wording):**
> DIM and DP are represented in **model-step units**, not raw calendar days.

Model step and biological variable units are related but not identical --
but for a variable that is also a **discrete Markov state variable**, the
two are deliberately coupled, not independent.

The rule is:

- **Discrete state variables that advance once per Markov transition**
  (DIM, DP) are represented at **model-step resolution**. `transitions.py`
  advances them with `next_dim = dim + 1` once per transition; that is
  only correct biology if `dim` is already expressed in step-sized units.
  `TIME_UNIT` therefore determines the biological/calendar duration
  represented by one model step -- and, deliberately, the resolution
  (and size) of the discrete state space. Do not interpret DIM/DP as
  independent calendar-day counters when the model step is monthly or
  otherwise aggregated: a coarser `TIME_UNIT` intentionally buys a
  smaller state space by coarsening DIM/DP resolution along with it.
- **Free-standing biological/calendar duration constants** that are not
  themselves part of the discrete state (gestation length, dry-off
  period, voluntary waiting period, etc.) remain real-day constants,
  converted to the active model step only where a formula needs it
  (e.g. via `UNIT_DAYS`/`to_units()`).

Example:

```text
MRP step: month (TIME_UNIT = "month", UNIT_DAYS = 30)

dim, dp   -> stored in months (model-step units); dim=3 means "3 months in milk"
GESTATION_LENGTH_DAYS = 279   -> stays a real-day constant, always
```

The model time system defines:

> How often the MRP advances, and at what resolution DIM/DP are tracked.

The domain defines:

> What each biological variable means, and which constants are
> free-standing durations rather than part of the discrete state.

This distinction must remain explicit.

---

# 22. Time Metadata and Validation

Where useful, model specifications should include time metadata:

```python
{
    "name": "abortion_probability",
    "depends_on": ["dp", "dim", "lactation"],
    "time_unit": "month",
    "method": "table",
    ...
}
```

The Engine should detect obvious inconsistencies such as:

- a probability declared in one time unit but used as another;
- reward duration inconsistent with the model step;
- unsupported time units;
- missing required conversions;
- negative duration;
- zero/invalid step;
- incompatible model metadata.

The objective is to fail early rather than produce silently incorrect results.

---

# 23. Probability and Risk Scaling Across Time

A probability or rate defined for one period must not automatically be assumed to scale linearly to another.

Examples of possible conversions, depending on scientific interpretation, include:

```text
p_step = 1 - (1 - p_reference)^(step_duration / reference_duration)
```

or, for an appropriate hazard-rate interpretation:

```text
p_step = 1 - exp(-rate × duration)
```

Do not automatically use:

```text
p_step = rate × duration
```

unless scientifically appropriate.

The generic system may provide generic conversion utilities.

The domain/model must define the scientific interpretation that makes a particular conversion valid.

---

# 24. Discounting and Time

Discounting must use the same authoritative model-time system.

Conceptually:

```text
Annual discount specification
        |
        v
Centralized time conversion
        |
        v
Per-step discount factor
        |
        v
MRP evaluation
```

For example, a conversion utility may conceptually perform:

```text
annual_to_per_step_gamma(gamma_annual, unit_days)
```

Changing the model step must not silently alter the intended annual discounting behavior.

The domain should not hard-code generic discounting into biological formulas.

---

# 25. Time Responsibilities Across Files

The exact file names must follow the actual repository, but the intended responsibility pattern is:

```text
Authoritative time configuration
        |
        +--> centralized conversion utilities
        |
        +--> state construction / progression
        |
        +--> transition timing
        |
        +--> probability scaling
        |
        +--> reward duration
        |
        +--> discounting
        |
        +--> simulation step count
        |
        +--> farm tools
        |
        +--> notebooks / benchmarks / reporting
```

No downstream component should independently redefine the meaning of a model month/day/week.

Domain files may define biological durations such as gestation or dry-period durations, but their relationship to model steps must use the shared time system.

---

# 26. Configuration Philosophy

Normal configuration should be direct and easy to find.

Conceptually:

```python
# ============================================================
# USER CONFIGURATION
# ============================================================

TIME_UNIT = "month"
GAMMA_ANNUAL = ...
NUM_YEARS = ...
DOMAIN = ...
INITIAL_STATE = ...

PROBABILITY_MODEL = "..."
REWARD_MODEL = "..."
```

Each configurable item should document nearby:

- meaning;
- units;
- valid options;
- expected format;
- useful ranges where relevant;
- assumptions.

When a new selectable model is added, the available options should be discoverable.

Avoid interactive prompts for ordinary configuration unless specifically useful.

Avoid unnecessary configuration frameworks.

---

# 27. Parameter Categories

Keep conceptual categories distinct:

```text
Biological/domain parameters
Economic/reward parameters
Generic MRP/runner configuration
```

Examples:

```text
Gestation length   -> biological/domain
Milk price         -> economic/reward
Feed price         -> economic/reward
Discount factor    -> MRP/runner
```

Do not casually mix biology and economics unless the model explicitly defines a combined quantity.

Economic parameters belong with reward/economic modeling rather than generic MRP Core.

---

# 28. Scientific Integrity and Traceability

This is a research-oriented modeling system.

Never silently invent:

- probabilities;
- biological constants;
- transition probabilities;
- equations;
- reward values;
- economic values;
- physiological assumptions;
- thresholds;
- lookup values/tables;
- scientific references.

If scientific information is missing:

```text
Identify it
    |
Explain why it is required
    |
Ask for source or explicit assumption
    |
Do not silently invent it
```

Clearly distinguish:

- literature-based;
- user-provided;
- estimated;
- assumed;
- provisional;
- placeholder.

Do not fabricate citations or imply literature support that does not exist.

Every important scientific model should document where applicable:

- name;
- purpose;
- inputs;
- output;
- units;
- equation/table/model;
- parameters;
- assumptions;
- valid range;
- source;
- limitations.

Scientific assumptions should not be hidden inside implementation code.

---

# 29. Multiple Scientific Models

Different scientifically valid formulations may exist for the same process.

Do not silently declare one universally correct.

Where appropriate:

- preserve alternatives;
- document sources;
- document assumptions;
- expose required inputs;
- make selection explicit;
- permit comparison when useful.

This applies especially to:

- probabilities;
- biological relationships;
- reward/economic formulations.

---

# 30. MRP Mathematics

The generic Core owns MRP mathematics.

Finite horizon:

```text
V_(t+1) = R + γ P V_t
```

Infinite discounted horizon, where appropriate:

```text
V = R + γ P V
```

or:

```text
V = (I - γP)^(-1) R
```

The domain owns the scientific meaning and construction of:

```text
P
R
states
```

The numerical method should consider computational appropriateness, including sparse/dense representation and numerical stability.

---

# 31. Distribution Evolution and Simulation

Generic distribution evolution may support, where useful:

- recursive multiplication;
- matrix powers;
- eigen-based methods for suitable cases.

Do not force an elegant method onto a large state space if a simpler method is more appropriate.

Generic stochastic simulation may operate on:

- transition matrix;
- initial state IDs/distribution;
- number of steps;
- random seed.

Potential capabilities:

- single trajectories;
- multiple independent simulations;
- Monte Carlo;
- seeded reproducibility;
- transition sampling;
- trajectory output.

Generic simulation should not need to know whether states represent cows, sheep, machines, or another domain.

Domain-specific grouping/reporting belongs outside generic simulation mathematics.

---

# 32. Farm-Level Tools

The project discusses:

```text
farm_generator.py
farm_simulator.py
```

These introduce a level beyond an individual state-transition model.

Possible generator responsibilities:

- herd size;
- herd mix;
- initial state distribution;
- simple/proportion-based generation;
- exact state-count generation.

The generator should use domain-valid state definitions rather than maintaining a separate hard-coded list of biologically valid states.

Possible simulator modes:

```text
expected
stochastic
monte_carlo
```

Conceptually:

- **Expected:** distribution evolution for expected herd dynamics.
- **Stochastic:** individual random trajectories.
- **Monte Carlo:** repeated independent simulations to quantify variability.

However, file names do not determine architecture.

Inspect actual contents before moving these components.

Keep farm functionality Cow-specific unless inspection demonstrates a genuinely generic mechanism. If a farm requirement reveals a generic capability, extract only that mechanism and retain farm interpretation in the domain/tool layer.

---

# 33. Research Reproducibility

A result should be understandable in terms of:

- selected probability models;
- selected reward models;
- parameter values;
- time unit;
- state definitions;
- assumptions;
- scientific sources;
- random seed where stochastic simulation is used.

Model selections and time settings should be explicit and traceable.

---

# 34. Testing and Validation

Where practical, test independently:

- MRP Core;
- generic Engine;
- domain logic;
- domain/Engine integration.

Test:

- state-space construction;
- derived-variable dependencies;
- dependency validation;
- model selection;
- probability validity;
- lookup coverage;
- reward construction;
- reward aggregation;
- possible double counting;
- time conversions;
- probability/rate scaling;
- discounting consistency;
- transition-matrix validity;
- stochasticity;
- MRP dimensions and reward compatibility;
- numerical stability where relevant.

Generic tests should not require Cow assumptions unless explicitly testing Cow integration.

**Current implementation:** there is no pytest or other test-framework
dependency, by deliberate choice (§37 -- don't add a dependency without
genuine need). Verification instead uses plain scripts that compare
deterministic model outputs (state count, transition nonzero count,
finite/infinite-horizon value means, steady-state trajectory values)
against a saved baseline within tight numerical tolerance: `benchmark.py`
and a small parity-check script, `verify_migration_parity.py` -- both
implemented.

---

# 35. Engineering Philosophy

> **SIMPLE OVER CLEVER**

Avoid unnecessary:

- classes;
- factories;
- metaclasses;
- plugin frameworks;
- abstractions;
- files;
- modules;
- interfaces;
- architectural layers;
- configuration frameworks;
- dependencies;
- refactoring.

A dictionary, specification, or simple registry may be preferable to a large class hierarchy.

However, simplicity must not justify silent scientific errors or hard-coded domain behavior.

The target is:

> **The simplest architecture that preserves genericity, correctness, extensibility, validation, and scientific traceability.**

---

# 36. Preserve Existing Working Code

Before changing existing components:

1. inspect how they are used;
2. understand their responsibility;
3. identify their architectural layer;
4. inspect related dependencies;
5. check whether functionality already exists;
6. determine whether the requirement is generic or domain-specific;
7. make the smallest appropriate change.

Do not:

- rewrite working code merely because another implementation looks cleaner;
- refactor unrelated code;
- rename/move files without a clear architectural reason;
- reorganize the repository based only on filenames.

---

# 37. Before Creating Files or Dependencies

Before creating a file:

1. Is it genuinely necessary?
2. Can the responsibility live clearly in an existing file?
3. Does it have one clear responsibility?
4. Which architectural layer owns it?
5. Does it create unnecessary dependency?

Before adding a package:

1. Does the project already provide the functionality?
2. Is the standard library sufficient?
3. Is an existing dependency sufficient?
4. Is the new dependency genuinely necessary?

Known project discussions mention:

```text
numpy
pandas
scipy
matplotlib
jupyter
pyarrow
```

with Plotly discussed as potentially optional for interactive dashboards.

Do not add dependencies merely for convenience.

---

# 38. Repository Inspection Before Major Refactoring

Before restructuring, perform analysis first.

Inspect:

```text
mrp_core.py
config.py
cow/
farm_generator.py
farm_simulator.py
benchmark.py
benchmark_worker.py
mrp_engine.ipynb
```

and any actual equivalent files.

Create an architectural map showing:

1. what each file does;
2. whether it is Core, Engine, Domain, Configuration, Farm, Benchmark, or Notebook/Example;
3. imports/dependency direction;
4. where time is defined;
5. whether time is consistent;
6. where probability models live;
7. where reward models live;
8. how new probability models are added;
9. how new rewards are added;
10. where the current implementation conflicts with this design;
11. what information is missing before architecture changes.

Do not automatically fix problems during this inspection.

Establish:

```text
Current repository
        vs.
Intended architecture
        |
        v
Explicit gap analysis
        |
        v
Discuss major decisions
        |
        v
Smallest justified changes
```

---

# 39. Major Decisions Requiring Discussion

Ask before:

- changing MRP into MDP;
- moving major functionality between layers;
- changing state semantics;
- changing time representation;
- replacing a scientific model;
- introducing a major abstraction;
- creating a new architectural layer;
- major folder restructuring;
- changing the fundamental state/probability/reward architecture;
- making generic code domain-specific.

If architecture is ambiguous and the decision materially affects future design, explain the ambiguity rather than guessing.

---

# 40. What Must Never Happen

The following are explicitly undesirable:

1. Cow-specific logic hard-coded into generic MRP mathematics.
2. Generic code assuming Cow state names exist.
3. Different files interpreting time differently.
4. Changing `TIME_UNIT` requiring manual fixes throughout unrelated files.
5. A model silently using undeclared variables.
6. A model silently receiving an approximate substitute for a missing input.
7. Missing lookup values silently becoming zero.
8. Missing scientific information hidden by undocumented defaults.
9. Scientific values or references fabricated.
10. Adding a probability requiring unrelated Engine rewrites.
11. Adding a reward requiring unrelated Engine rewrites.
12. Introducing MDP/action logic without explicit decision.
13. Large architectural refactors without approval.
14. Over-engineering infrastructure.
15. Treating the Cow implementation as the definition of the generic engine.
16. Silently falling back from a missing requested model implementation to another method.
17. Silently normalizing an invalid transition row.
18. Forcing every model to use every available state.
19. Making all dependencies optional merely to avoid validation errors.

---

# 41. Final Pre-Commit / Pre-Completion Checklist

## Architecture

- Cow/domain knowledge remains in the domain.
- MRP Core remains domain-independent.
- MRP Engine remains domain-independent.
- No unnecessary architectural layer was added.

## States and dependencies

- New states can be added through domain definitions.
- Required dependencies are validated.
- Derived dependencies are validated.
- Optional dependencies have explicit absence behavior.
- Adding an unused state does not break unrelated models.
- State IDs are stable and unique.

## Probabilities and lookups

- Every selected method exists.
- Required inputs exist.
- Method resolution is explicit.
- No silent fallback occurred.
- Lookup dimensions match required dependencies.
- Lookup coverage is validated.
- Missing combinations are visible.
- Units/time metadata are compatible.
- No scientific defaults were invented.

## Rewards

- Every selected reward method exists.
- Dependencies exist.
- Components are modular.
- Signs and aggregation are explicit.
- Double counting has been considered.

## Time

- One authoritative model-time definition exists.
- Conversions are centralized.
- Biological units are not confused with model step.
- Probability/rate scaling is explicit.
- Missing/incompatible time conversion is detected.
- Discounting uses the correct model step.
- Time logic is not duplicated across unrelated files.

## Transitions and MRP

- Transition targets exist.
- Probabilities are valid.
- Rows are stochastic where required.
- Invalid rows are not silently normalized.
- Reward dimensions are compatible.
- Numerical stability is considered where relevant.

## Scientific integrity

- Sources are documented where applicable.
- Assumptions are clearly identified.
- No unsupported scientific values were invented.

## Engineering

- Existing architecture was inspected first.
- No unrelated code was rewritten.
- No unnecessary dependency was added.
- No unnecessary abstraction was introduced.

---

# 42. Ultimate Project Rules

Always preserve:

> **THE COW DOMAIN IS THE FIRST DOMAIN USING THE MRP ENGINE. IT IS NOT THE DEFINITION OF THE MRP ENGINE.**

> **GENERIC CODE PROVIDES MECHANISMS; DOMAINS PROVIDE MEANING.**

> **A STATE DECLARATION DOES NOT AUTOMATICALLY REQUIRE EVERY PROBABILITY OR REWARD TO USE THAT STATE.**

> **MODELS DECLARE WHAT THEY NEED; THE ENGINE VALIDATES WHETHER THOSE REQUIREMENTS CAN BE SATISFIED.**

> **MODEL INPUTS ARE NOT LIMITED TO STATES. THEY MAY INCLUDE STATES, DERIVED STATES, PARAMETERS, EXTERNAL INPUTS, LOOKUP DATA, MODEL OUTPUTS, AND TIME INFORMATION.**

> **LOOKUP TABLES MUST DECLARE THEIR KEYS AND COVERAGE REQUIREMENTS. MISSING SCIENTIFIC VALUES MUST NOT BE SILENTLY FILLED.**

> **TIME IS A SHARED SYSTEM CONCEPT. ONE AUTHORITATIVE TIME INTERPRETATION MUST PROPAGATE THROUGH STATES, TRANSITIONS, PROBABILITIES, REWARDS, SIMULATION, DISCOUNTING, FARM TOOLS, BENCHMARKS, AND OUTPUTS.**

> **BIOLOGICAL VARIABLE UNITS AND MRP MODEL TIME ARE DISTINCT CONCEPTS.**

> **PROBABILITY/RISK SCALING ACROSS TIME MUST BE EXPLICIT AND SCIENTIFICALLY JUSTIFIED.**

> **NEW PROBABILITY AND REWARD MODELS SHOULD BE ADDABLE WITHOUT REWRITING UNRELATED ENGINE CODE.**

> **WHEN SOMETHING IS MISSING, FAIL EARLY, FAIL CLEARLY, AND EXPLAIN WHAT IS REQUIRED.**

> **DO NOT INVENT SCIENTIFIC KNOWLEDGE, VALUES, MODELS, OR REFERENCES.**

> **EXTENSIBILITY IS IMPORTANT, BUT OVER-ENGINEERING IS NOT THE GOAL.**

The intended outcome is:

```text
Generic MRP Core
+
Generic MRP Engine
+
Domain-specific scientific models
+
Extensible state/input system
+
Explicit dependency contracts
+
Validated lookup coverage
+
Centralized time architecture
+
Selectable probability and reward models
+
Traceable assumptions and configuration
+
Clear failure when information is missing
=
Reusable, extensible, research-grade MRP framework
```

---

# 43. Source Reconciliation Note

The supplied source contains multiple overlapping versions, pasted conversations, and repeated drafts. This document consolidates the recurring requirements and later refinements rather than treating every earlier wording as an independent rule.

Where the source evolved, the stronger later refinements are preserved. Examples include:

- moving from “state dependencies” to the broader concept of **model inputs/dependencies**;
- explicitly allowing non-state inputs such as parameters, external variables, lookup data, and model outputs;
- distinguishing required and optional dependencies;
- expanding lookup validation from basic key presence to dimensions and full combination coverage;
- expanding time consistency into explicit centralized conversion, time metadata, validation, probability scaling, and discounting;
- clarifying that generic layers must not hard-code Cow knowledge, rather than imposing an overly literal rule that could prevent a generic engine from receiving domain implementations;
- treating model completeness and actionable pre-run validation as first-class requirements;
- resolving the Core/Engine boundary as a real functional split (`mrp_core.py` /
  `mrp_engine.py`), not a single fused file -- while explicitly rejecting a
  class/registry/plugin-framework implementation of the Engine layer;
- correcting §21's original worked example: DIM/DP are represented in
  model-step units, not raw calendar days, because they are discrete
  state variables that advance once per Markov transition -- only
  free-standing biological/calendar duration constants (e.g. gestation
  length) remain in real days.

Any future decision not clearly supported by this record should remain explicit rather than being silently invented.

---

# 44. Environmental Rewards Architecture

Added following a scientific audit of a candidate emissions specification
(see git history for the audit report). Satisfies §28/§29 above: multiple
methods per component where they exist, method metadata instead of
hidden constants, nothing silently invented.

**Layering** (`cow/rewards/environment/`): a biological-calculation layer
(`compute_environment(states, probs)`, mirroring `compute_probabilities`
-> `compute_rewards` threading) produces physical quantities -- never $
-- which reward specs then price. BW/DMI/milk yield are the repo's
*existing* pipeline functions (`cow.rewards.base.body_weight_kg`/
`dry_matter_intake_kg`, `cow.states.milk_yield_kg`), wrapped as stage-0
specs so environmental calculations use the same BW/DMI as feed cost --
never a second, disagreeing value for the same cow.

**Method selection**: every component (`nitrogen`, `enteric_methane`,
`manure_methane`, `n2o`) is an ordinary spec (`produces`/`depends_on`/
`method`/`methods`), reusing `mrp_engine.resolve_method` exactly as
probability/reward specs do -- no new Engine/Core code was needed;
`methods` dict keys are just named differently (`charmley_2016`,
`ipcc_tier2`, `po_valley_2026`, `ipcc_2019`) instead of always
table/formula/model. Selecting a method means editing the spec's
`'method'` field, same convention as every other spec in this repo. Each
method's module carries a `*_METHOD_INFO` dict (citation, scope,
required/optional inputs, equation, units, assumptions, limitations) --
see `cow/rewards/environment/*.py`.

**Not implemented**: CNCPS enteric methane (needs a feed-ingredient
library, ration formulation, and a rumen-kinetics submodel -- none exist
in this repo's herd-level diet representation). The `methods` dict
documents exactly what's missing rather than faking an equation.

**CO2 is tracked, not priced**: respiratory CO2 is computed
(`respiratory_co2_kg`) but has no `reward_co2` spec and is never in
`reward_total` -- IPCC/EPA convention excludes livestock respiratory CO2
from GHG-inventory accounting (biogenic, carbon-neutral over the
feed-growth cycle; only CH4 is counted). See
`cow/rewards/environment/co2.py`.

**Manure CH4/N2O require explicit configuration**: `MANURE_SYSTEM`/
`MANURE_CLIMATE` (`cow/parameters.py`, via `COW_MANURE_SYSTEM`/
`COW_MANURE_CLIMATE`) have no default -- IPCC MCF/EF values are climate-
and system-dependent, so no single default is defensible. Both
quantities are *skipped* (absent from the CSV), not silently defaulted,
until both are set.

**Independent selectability and no double counting**: `reward_nitrogen`/
`reward_enteric_methane`/`reward_manure_methane`/`reward_n2o` are always
computed and always present in the results dict/CSV (value `0.0` under
default price) -- visibility is never gated by a flag, per an explicit
later requirement that environmental components never be hidden.
Nitrogen and enteric methane are unconditionally in
`ENVIRONMENTAL_REWARD_SPECS`; manure methane and N2O are added to that
list only when `MANURE_SYSTEM`+`MANURE_CLIMATE` are both configured
(independent of `ENABLE_REWARD_*`). What `ENABLE_REWARD_*` (`cow/
parameters.py`) actually gates is narrower than the name once implied:
whether each component's value is folded into `results['total']` --
all default `False`, so `total` is byte-identical to before this
feature existed unless a user opts in. `compute_rewards()` also exposes
`reward_environment_total`, a post-hoc sum of `sign * value` over
whichever environmental components are currently `total`-counted --
it is a derived aggregation, never its own priced spec, so it can never
double-count against the components it sums. N2O's three pathways
(`n2o_direct_kg`/`n2o_indirect_volatilization_kg`/
`n2o_indirect_leaching_kg`) are each independently observable in the
CSV, but only their sum (`n2o_total_kg`) is priced -- pricing each
pathway separately would triple-price the same manure-N source.

**Pricing**: `PRICE_PER_KG_N_MANURE`/`PRICE_PER_KG_CH4`/
`PRICE_PER_KG_N2O` default to `0.0` -- explicit economic/policy
assumptions, not biological constants; a reward is exactly zero until a
price is set intentionally.

**Units and time convention**: every biological quantity is computed as
a daily rate internally, then multiplied by `UNIT_DAYS` once (matching
`milk_produced_kg`'s existing convention) to report the per-model-step
amount in the CSV -- consistent regardless of `TIME_UNIT`. BW is a
stock, not a flow, and is deliberately never `UNIT_DAYS`-scaled.

**Verification pass (follow-up)**: every numeric constant flagged
UNCONFIRMED above was subsequently checked directly against the primary
IPCC (2019) PDFs (Chapter 10 and Chapter 11) and Pedersen et al. (2008),
not secondary summaries. Corrected: `manure_methane.py`'s `MCF_TABLE`
(liquid/slurry and uncovered-lagoon rows; dry_lot/daily_spread were
already exact); `n2o.py`'s `EF_DIRECT_TABLE` (daily_spread, liquid_slurry,
anaerobic_digester) and `FRAC_GAS_TABLE` (daily_spread, solid_storage,
liquid_slurry, anaerobic_lagoon); the single global `MANURE_FRAC_LEACH`
scalar (removed -- it matched no real per-system value) was replaced by
`n2o.py`'s per-system `FRAC_LEACH_TABLE`, verified against Table 10.22.
Confirmed exact and unchanged: `B0=0.24`, the `0.67` m3-to-kg conversion,
`Ym=6.0%`, the `55.65` MJ/kg CH4 constant, `EF4=0.010`, `EF5=0.011`.
CO2's `209 L/Mcal` constant was corrected to `207`, derived transparently
from Pedersen et al. (2008)'s own cow-specific measurements (not the
paper's general cross-species figure) rather than left unconfirmed.
`ME_OVER_DE=0.82` and `NE_MAINT_COEFFICIENT=0.08` are both real,
sourced, long-standing values (ARC 1965/NRC 1976-2000/NASEM 2016; NRC
2001, respectively) but each has a documented newer-standard caveat
(NASEM discourages 0.82 for high-concentrate diets; NASEM 2021 revised
maintenance to 0.10) -- kept at the older, still-valid values rather
than silently switched, per that pass's own scope (verify/correct/label,
not redesign). See each module's `*_METHOD_INFO` dict and inline
comments for the full per-constant citation and derivation.

**Adding a new method**: add a function to the relevant module's
`methods` dict plus an entry in its `*_METHOD_INFO` dict. Adding a new
environmental component entirely follows the same three steps as any
other spec in this repo (create it, add it to the right `_STAGE_*` list
in `cow/rewards/environment/__init__.py`, done) -- see that file's
module docstring.

---

# 45. Scientific/Thesis Documentation

Thesis-grade scientific documentation (mathematical formulation with
every variable/unit/citation, biological/environmental method templates,
assumptions register, limitations register, validation log, reference
bibliography, model changelog) lives in **`docs/`**, not here.
`PRIMARY.md` stays the engineering/architecture design record (why the
code is shaped this way); `docs/` is the science-reader-facing record
(what the biology/math means and what supports it). Do not duplicate
`docs/`'s content into `PRIMARY.md`, or vice versa -- cross-reference.

`docs/` files: `THESIS_MODEL.md` (master narrative), `ARCHITECTURE.md`,
`BIOLOGICAL_ASSUMPTIONS.md`, `ENVIRONMENTAL_EMISSIONS.md`,
`REWARD_SYSTEM.md`, `REFERENCES.md`, `ASSUMPTIONS_REGISTER.md`,
`LIMITATIONS_REGISTER.md`, `VALIDATION.md`, `MODEL_CHANGELOG.md`,
`SYSTEM_WORKFLOW.md` (code-grounded architecture/workflow map -- system
tree, execution order, reward/environmental/lactation/benchmark
workflows, "where do I change X" table; every reference in it must be
verified against current code, not assumed). Structural (presence-only,
never correctness) consistency check: `python
validate_model_documentation.py`.

**Documentation-maintenance trigger table** (also in `CLAUDE.md` -- kept
in both since one is the architecture record and the other is the
operational instruction future agents actually read before editing):

| Change | Update |
|---|---|
| New/changed spec in `cow/probabilities\|rewards/*.py` | `docs/BIOLOGICAL_ASSUMPTIONS.md` or `docs/ENVIRONMENTAL_EMISSIONS.md` |
| New/changed constant in `cow/parameters.py` | `docs/ASSUMPTIONS_REGISTER.md` |
| New file, dependency kind, or architectural pattern | `PRIMARY.md` (this file) |
| Dev-workflow/instruction change | `CLAUDE.md` |
| Any of the above | `docs/MODEL_CHANGELOG.md` (always, append-only) |

What's auto-checkable (`validate_model_documentation.py`): presence of
required fields in `*_METHOD_INFO` dicts, presence of reward-spec/flag
names in `docs/REWARD_SYSTEM.md`, presence of bio-quantity names in
`docs/ENVIRONMENTAL_EMISSIONS.md`. What is NOT automatable and never
will be: whether an equation is scientifically correct, whether a
citation is accurately characterized, whether an assumption's impact
assessment is right -- that always requires the primary-source-fetching
audit process recorded in `docs/REFERENCES.md`.

---

# 46. State-Value Comparison Diagnostics

A generic capability for comparing the expected future value of two
existing states -- `V(state_A) - V(state_B)` -- layered on top of the
Core math that already existed (`evaluate_finite_horizon`/
`evaluate_infinite_horizon` already computed a full per-state value
vector; it was previously discarded after being collapsed to a mean).
Full design history and validation: `docs/superpowers/specs/2026-08-17-
state-value-difference-framework-design.md`.

**Core** (`mrp_core.py`): `state_values(states, V, query_A, query_B)`
returns the two raw values `(V[id_a], V[id_b])`; `state_value_diff(...)`
returns their difference. Both are domain-independent -- they know
nothing about pregnancy, cows, or any domain concept, only "state table +
value vector + attribute query," via the existing `find_state_id`.

**Cow Domain** (`cow/value_comparisons.py`): owns biological meaning only.
`pregnancy_value(states, V, lactation, dim, health)` = V(Pregnant,dp=1) -
V(Open,dp=0), matched dim/lactation/health. `replacement_value(states, V,
lactation, dim, dp, health, flag)` = V(current state) - V(replacement
state), where the replacement state is the exact fixed state
`cow/transitions.py` already routes every culled/dead cow to. Both return
`{"value", "V_state", "V_comparison"}`. `VALUE_COMPARISON_FUNCTIONS` (a
plain `{name: function}` dict in that same file) is the add/remove point
for new comparisons -- adding one means writing a small function shaped
like these two plus one registry line; no Core/Engine change is ever
required.

**Selection/reporting** (`config.py`/`benchmark_worker.py`): `config.py`'s
`VALUE_COMPARISONS` list (mirrors the existing `TRACK_STATES` convention --
real-calendar-day parameters, converted to model steps automatically) names
which registered comparisons run and at which states, per model.
`benchmark_worker.py` looks each one up in `VALUE_COMPARISON_FUNCTIONS` by
name and writes the results to `value_comparisons.csv` /
`worker_result.json['value_comparisons']`, alongside (never inside) the
existing reward breakdown.

**Not reward terms**: never registered in `cow/rewards/`, never appended
to `reward_specs`, never affect `rewards['total']`/
`reward_environment_total` -- verified by an explicit bit-identical check
in `verify_value_comparisons.py`. Pregnancy and replacement are two
concrete examples of one generic mechanism, not two separate
implementations -- see `cow/value_comparisons.py`'s module docstring for
the full counterfactual-validity and biological-validity discussion
(including `replacement_value`'s known unconstrained-heifer-supply
assumption, inherited unchanged from `cow/transitions.py`).

Architecture unchanged: Cow Domain (comparison definitions) -> MRP Engine
(uninvolved -- nothing here needed orchestration) -> MRP Core (`state_values`/
`state_value_diff`, both generic). No MDP/action/policy layer exists or is
implied by this capability.

---

# 47. Local Research Dashboard

A local, single-user Flask UI (`dashboard/`) sits **outside and above**
the layered architecture this document otherwise governs -- alongside
`benchmark.py`/`farm_generator.py`/`farm_simulator.py` in the Runner
layer, not inside Core/Engine/Domain. Full operational detail (routes,
run-registry mechanics, reload guarantees) lives in `dashboard/README.md`;
this section records the one architectural invariant that matters here
and is not duplicated there.

**The invariant**: `dashboard/` never imports `cow/`, `mrp_core.py`, or
`mrp_engine.py` to compute anything. Every actual model computation is
delegated to the repo's existing entry points -- `benchmark.py`'s
`run_one_model()`/`main()` (called directly, reused not re-implemented)
for model runs, `verify_migration_parity.py`/`verify_value_comparisons.py`
launched as real subprocesses for validation. `benchmark_worker.py`
itself always runs in a fresh subprocess per model, exactly as it did
before the dashboard existed, so a dashboard-triggered run and a
command-line run are the same computation -- this was verified directly
(dashboard-triggered "Run Model A" reproduced the pre-existing
`benchmark_outputs/Model_A/worker_result.json`'s `num_states`,
`nnz_transitions`, `V_finite_mean`, `V_infinite_mean` exactly).

**Farm data boundary** (`dashboard/data_layer.py`): an uploaded CSV's
columns are mapped, by the user, onto a small canonical schema
(`cow_id`, `lactation`, `dim`, `dp`, `health`, `milk_kg`,
`body_weight_kg`, `dmi_kg`, `scc`). Only `lactation`/`dim`/`dp`/`health`
have any current Cow Domain consumer (`cow/states.py`'s
`variable_domains`); everything else is displayed as observed-only data
with no model consumer, never silently absorbed into a state. Matching
mapped farm rows to model output (`dashboard/runs_layer.py::
match_farm_to_run`) is a plain dataframe join against an
already-computed run's `reward_vector.csv` -- never a recomputation,
never a fabricated state. One real modeling subtlety this surfaced:
`cow/states.py`'s derived `flag` rule allows **two** valid states
(`Open` and `DNB`) at the same `(lactation, dim, dp, health)` whenever
`dp == 0` and `dim` is past `DNB_EARLIEST_POSSIBLE_DIM_DAYS` (see
config.py's docstring on FLAG's constraint by dp_days) -- farm data with
no reproductive-status field cannot disambiguate which one a real cow is
in, so such rows are reported `ambiguous` rather than matched to either
one (an earlier version of this join silently fanned out to both, which
was caught and fixed before this section was written).

**Value comparisons**: rendered generically from
`config.VALUE_COMPARISONS`/`worker_result.json['value_comparisons']` --
see §46. A new registry entry in `cow/value_comparisons.py` plus a
`config.py` entry appears in the dashboard automatically; no dashboard
code change is the expected outcome, not a lucky accident, because the
dashboard never enumerates comparison names itself.

No MDP/action/policy layer is introduced or implied by the dashboard.


---

# 48. Population Layers: Youngstock, Farm Population and Entrant Economics

*(Added 2026-09-19 at the explicit request of the user, after the architecture was finalized. Science-facing detail is in `docs/`; this section is the engineering record.)*

## 48.1 Four layers

1. **Cow MRP** — individual-cow stochastic valuation (`cow/`, `mrp_engine.py`, `mrp_core.py`). Unchanged contract: row-stochastic `P`, per-state rewards, `V = (I - gamma P)^-1 R`. It values one cow in a constant herd whose exits are replaced instantly.
2. **Youngstock population layer** — `youngstock/`: an explicit, row-stochastic chain `Q` for calf -> heifer -> breeding -> pregnant -> graduate, with real absorbing states (DIED, CULLED, GRADUATED, SOLD, STILLBORN) and lines DAIRY_F / DAIRY_M / BEEF_F / BEEF_M / ALL. Births are an *external inflow*, not a transition.
3. **Farm population layer** — `farm_population.py`, `cow/flows.py`, `population_economics.py`, `farm_population_scenarios.py`: population accounting and herd evolution. Runner-level; imports Domain, never modifies Core/Engine.
4. **Future MDP/OR layer** — management actions and optimization (culling/keep-replace, raise vs purchase, breeding program). **Not built.** Replacement optimization is NOT solved anywhere in this repository.

Dependency direction is unchanged (Domain -> Engine -> Core); layers 2-3 sit at the Domain/Runner level.

## 48.2 Equations

- Youngstock: `y_{t+1} = y_t Q + sum_p E_p(t) * mix_p`, with dam-parity events `E_2 = n_t . (c * 1[L=1])`, `E_3+ = n_t . (c * 1[L>=2])`, `E_1 = graduates` (heifers calving for the first time), `c` = per-cow calving probability. `mix_p` is the per-parity fate tree (stillbirth, twinning, semen program, sex ratio, retention, pre-sale mortality, terminal sale).
- OPEN (primary): `n_{t+1} = n_t P` with the replacement column replaced by `graduates + purchased`; culls/deaths leave the herd. CLOSED (compatibility/counterfactual): `n_{t+1} = n_t P` with the implicit replacement (cow outputs identical to the cow MRP).
- Conservation is asserted every step (max error ~1e-12).
- **Retention** = share of live female dairy calves kept; a scenario input. `farm_population.balancing_retention` finds the retention at which the cow inventory ends where it started (0.677 at MONTH, 5 y, conventional program). It is an *inventory-balancing retention scenario*, never an optimum or a policy.

## 48.3 Time convention (uniform entry)

An event that sets a state occurs at a uniform time inside its step: true age at boundary k is `k*U + u` (DIM) or `(k-1)*U + u` (gestation). Per-step probability `= 1 - Sbar(k+1)/Sbar(k)` with `Sbar` the u-averaged survival (`cow/probabilities/base.py::mean_step_probability`); threshold events cross over a distribution of boundaries (`threshold_crossing_mass`); the first pregnant step carries the pre-boundary loss. This is resolution-aware, not resolution-specific. Cumulative pregnancy loss to calving equals the source hazard at MONTH/WEEK/DAY. The conception table is read as pregnancy establishment (`cow/parameters.py::CONCEPTION_INTERPRETATION_2026_09_19`); values unchanged. Known residual: MONTH vs WEEK cow chain differ (~10% in replacement demand); see `docs/LIMITATIONS_REGISTER.md` L-037.

## 48.4 Economics

Biological flow and economic reward are separate: no headcount, sale or graduation ever enters `rewards['total']`. `population_economics.py` prices flows: OPEN cow operating vector = `total - replacement_cost - pregnancy_reward` (those two price the implicit constant-herd replacement/calving), plus `CALF_VALUE x live calves sold`, minus `REARING_COST x graduates`, minus `PURCHASE_COST x purchased entrants`. One biological event = one economic event; a sale is not mortality; a purchased heifer is not also charged a rearing cost. Calf prices are a single placeholder (L-038). The state-value diagnostics of §46 are unaffected.

## 48.4b Reward-unit consistency (2026-09-20)

Every per-step reward must be either a RATE (multiplied by `UNIT_DAYS`, or by `UNIT_DAYS / RATE_PERIOD_DAYS` when the constant is documented per native 30-day period) or an EVENT cost (never scaled; its expected occurrences per step come from an explicit event rate or probability). `health_cost` (treatment/disease per period, milk loss per day) and `reproduction_cost` (services per step) were brought under this rule; no price or meaning changed, and MONTH values are unchanged except the health milk-loss term. Even with consistent units, MONTH and WEEK still differ (equilibrium reward 520 vs 688 $/cow-year, V_eq ratio 0.757) because of discrete-time resolution and the existing health-chain step dependence (`docs/LIMITATIONS_REGISTER.md` L-037, L-041, L-042); no parameter was tuned to force agreement and there is no external validation claim. The sequential exit -> DNB -> conception factorization in `cow/transitions.py` is retained and its resolution dependence is documented, not corrected.

## 48.5 Verification and honesty clause

`verify_youngstock.py` (optionally `--with-week-cow`), `verify_population_economics.py`, `verify_farm_simulation.py`, `verify_value_comparisons.py`, `verify_lactationcurve_milk_income_reward.py`, `verify_migration_parity.py` (baseline reconstructed 2026-09-19, self-consistency only), `Victor/tests/run_all.py` (86/86), `validate_model_documentation.py`. These establish software consistency, not biological validity. Placeholders are registered (`youngstock/parameters.py::register`, `docs/ASSUMPTIONS_REGISTER.md`); nothing here claims the model is validated or optimal.


# 49. Lookup-table layer and the time-discretization convention (CowMRP, 2026-10-08)

`cow/lookup_tables.py` is the single place that (a) defines FULL lookup tables (lactation 1..MAX_LACTATION x every native month x status), (b) validates their coverage and loads optional overrides from `cow/data/`, and (c) converts native monthly values to per-step values. The convention: a cow entering a phase does so at a uniform time inside the previous step, so her true age in step k is k*U + u and her mean age during the step is (k+1)*U. Per-step probability = 1 - Sbar(k+1)/Sbar(k) with Sbar the entry-averaged survival; per-step quantities (milk, ages) are averaged over the same offset; thresholds in days map to steps by the mid-step rule (`cow/parameters.py::last_step_within`, `first_step_past`). `cow/tables.py` precomputes all (lactation, DIM)/DP quantities before states are generated, including the reachable DIM bound. Engine and Core are unchanged. See `docs/ARCHITECTURE.md`, `docs/ASSUMPTIONS_REGISTER.md` A-068..A-070.


# 50. Layout, reachable states and the audit (CowMRP, 2026-10-08)

**Layout.** `mrp/` is the domain-free engine (`core.py`, `engine.py`, `simulator.py`); `cow/` is the cow domain; `runners/` (benchmark, farm) and `verify/` (checks, audit, plots) use both and hold no model logic; generated files live in `outputs/`. Older sections of this file refer to the pre-restructure names (`mrp_core.py` = `mrp/core.py`, `mrp_engine.py` = `mrp/engine.py`, `benchmark.py` + `benchmark_worker.py` = `runners/benchmark.py`, `config.py` = `runners/config.py`, `farm_*.py` = `runners/farm/*`, `verify_*.py` = `verify/`).

**Reachable states.** The Engine's `build_states` is unchanged (generic Cartesian product). `cow.states.build_cow_states()` applies a Domain-level filter: all open states plus the pregnant states `(dim, dp)` whose conception step `dim - dp` has positive conception probability, within the calving window (`dp_max`) and the cull / re-opening window (`dim_max`) computed in `cow/tables.py` before any state exists. Lossless and complete (every state reachable from the replacement state; V identical to the unpruned space), checked by `verify/verify_state_space.py`.

**Audit.** `docs/COW_MODEL_AUDIT.md` records the state-sufficiency argument, the single time-discretization convention, every probability and reward audited, the experiments kept and rejected, the resolution results and the unresolved issues. Event-level resolution consistency (cycle length, calvings, calving interval) is a standing check in `verify/verify_time_discretization.py`; V consistency is reported, not enforced.


# 51. Evidence layer, two-clock hazards and the method registry (CowMRP, 2026-10-08)

**`cow/evidence/`** holds literature values with provenance (one module per source: `pinedo2010.py`, `miller2008.py`), never model logic. Evidence classes: A exact printed value, B adapted (documented transformation, including values read from a graph), C insight only, D grouped or repeated across lactations.

**Two-clock hazard tables.** Mortality and the "other-reason" culling hazard are full tables (lactation 1..MAX_LACTATION x month x status Open / Pregnant / Dry) whose `month` is the DIM clock for Open rows and the DP clock (months of pregnancy) for Pregnant and Dry rows (`cow/probabilities/hazard_tables.py`). The state `(lactation, dim, dp)` carries both clocks, so no state is added. Per-step conversion is still the single entry-offset rule of `cow/lookup_tables.py`.

**Method registry** (`cow/registry.py`): one record per probability spec with table / formula / model availability and evidence, units, time basis, dependencies, lactation coverage and validation status; `check_registry()` (run by `verify/audit_registry.py`) fails if the registry and the specs disagree. Every probability keeps all three slots; an unavailable slot is an explicit error (`resolve_method`), never a silent fallback. A method can be switched for one run with `COW_METHOD_OVERRIDES="spec=method,..."` in `verify/audit_metrics.py` (sensitivity only; defaults are never edited by it).

**Reward economics.** The days-open penalty is off by default (`COW_DAYS_OPEN_PENALTY`): a per-open-day cost derived from milk and replacement flows is an output of the explicit flows in an MRP, not an additional input. See `docs/COW_MODEL_AUDIT.md` sections 15-26 for the audit, experiments E9-E15 and the unresolved list.


# 52. Projection runner and Excel bridge (CowMRP, 2026-10-08)

`projection/run_projection.py` is a Runner (it sits above Domain, Engine and Core and contains no model logic): scenario list at the top of the file, one worker process per `TIME_UNIT` (the model reads it at import), `cow.*` builds states / probabilities / transitions / rewards, `mrp.core` provides `make_initial_distribution`, `distribution_recursive_origin_tracked`, `annual_to_per_step_gamma`, `evaluate_finite_horizon` (consistency check) and `simulate_monte_carlo`. `cow/lookup_workbook.py` is a Domain-level tool, not part of the model path: Excel is an editing format; `get_full_table` caches one validated DataFrame per table in memory (key: override file mtime and builder), `clear_table_cache()` exists for in-process sensitivity tests.
