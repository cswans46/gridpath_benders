# GridPath Benders Decomposition: Mathematical and Algorithmic Documentation

Nov 17, 2026 · @Carl

## 1. Overview and scope

This implementation solves GridPath's capacity-expansion (investment) problem by Benders decomposition: a **planning (master) problem** chooses investments, and one **operational subproblem** per weather/hydro/availability/horizon key prices the operating cost of those investments and returns cuts. When GridPath is linked with NextGrid, we maximise the net present value (NPV) of total system profits, so the master objective is an upper bound (UB) and the best complete plan (the investment problem for which all operational subproblems have been evaluated) is a lower bound (LB).

The document covers the Benders code in `gridpath/benders/`, its entry points in `gridpath/run_scenario.py` and command-line flags passed in `gridpath/common_functions.py`.

| File | Role |
| --- | --- |
| `run_scenario.py` | Builds the Benders setup dict, calls `benders()`, saves root results, writes `benders_convergence.csv` and `benders_summary.json` |
| `common_functions.py` | `--benders*` command-line flags |
| `benders/algorithms.py` | `benders()` main loop, cut generation, bounds, big-M, stabilisation update, `BendersResult` |
| `benders/planning.py` | Master construction (`vTHETA`, objective, cut containers), master solve, solution extraction |
| `benders/subproblems.py` | Persistent worker process, linking constraints, dual extraction, slacks, discrete handling, final export |
| `benders/restoration.py` | Phase-I feasibility restoration with a dual certificate |
| `benders/regularization.py` | Level-set (stabilisation) problem, L1 and L2 |
| `benders/operational_model_creation.py` | Operational model and linking-variable identification |
| `benders/soc_allocation.py` | Shared SOC allocation targets and variables used by master and operational subproblems |\n| `system/policy/benders_policy_constraint.py` | Master-side policy allocation constraints, including cyclic storage SOC chaining and capacity limits |
| `benders/{gurobi,mosek,gams}_*.py` | Solver wrappers (MOSEK adds native L2 cones and disjunctive constraints) |

## 2. Problem structure and notation

The full problem maximises planning NPV minus the operating cost summed across all subproblems. The decomposition replaces each operating cost by a cost function Q\_w(x) of the investment decisions, which is convex and piecewise linear when the subproblem is a linear program.

$$
\begin{aligned}
\max_{x \in X}\quad & \mathrm{NPV}_{\mathrm{plan}}(x) - \sum_{w \in W} Q_w(x_L) \\
Q_w(\bar x)\quad &= -\max_{\substack{z_w \in Z_w \\ x_L = \bar x}}\ \mathrm{OP\_NPV}_w(z_w)
\end{aligned}
$$

| Symbol | Meaning in the code |
| --- | --- |
| x | All planning (master) variables, `all_variables(planning_problem)` |
| x\_L ⊆ x | **Linking variables**: the planning variables that appear in subproblems (section below) |
| w ∈ W | A subproblem key `ScenarioKey(weather, hydro, availability, subproblem, stage)` |
| x̄ | Trial point: the master values of x\_L sent to the workers (`continuous_point`) |
| Q\_w(x̄) | Recourse cost, `op_cost = −OP_NPV` in dollars |
| ϑ\_w | `vTHETA[w]`, the master's approximation of Q\_w, in units of S = 10⁹ dollars |
| S | `BENDERS_THETA_SCALE = 1e9` (`planning.py:19`) |
| λ\_w | Sensitivity of Q\_w to x\_L, from the dual of the linking equalities |
| L\_w | Global lower bound on Q\_w, from the LP-relaxation upper bound of operating NPV |
| UB, LB | Master optimum (non-increasing) and best complete plan's NPV (non-decreasing) |

**Linking variables.** `define_linking_vars` (`operational_model_creation.py:15-86`) collects two groups:

- the variables inside the investment expressions `Capacity_MW`, `Tx_Min_Capacity_MW`, `Tx_Max_Capacity_MW`, `TxNewLin_Capacity_MW` and `Energy_Storage_Capacity_MWh`, which are the build and capacity decisions;
- the **policy allocation variables**: the master splits each system-wide target into budgets for individual subproblems. The components explicitly treated as policy-budget/linking variables by `define_linking_vars` in `operational_model_creation.py` are listed below. A component is included only when it exists on the model:
  - **Emissions and energy targets**
    - `CARBON_CAP_ALLOCATION`
    - `PERIOD_ENERGY_TARGET_ALLOCATION`
  - **Performance standards**
    - `PERFORMANCE_STANDARD_MWH_ALLOCATION`
    - `PERFORMANCE_STANDARD_MW_ALLOCATION`
  - **Carbon-credit budgets and limits**
    - `BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION`
    - `BENDERS_CARBON_CREDIT_SUPPLY_ALLOCATION`
    - `BENDERS_CARBON_CREDIT_GENERATED_TOTAL`
    - `BENDERS_PROJECT_CARBON_CREDIT_ABSOLUTE_THRESHOLD_ALLOCATION`
  - **Transmission targets**
    - `TRANSMISSION_TARGET_POS_DIR_MIN_ALLOCATION`
    - `TRANSMISSION_TARGET_POS_DIR_MAX_ALLOCATION`
    - `TRANSMISSION_TARGET_NEG_DIR_MIN_ALLOCATION`
    - `TRANSMISSION_TARGET_NEG_DIR_MAX_ALLOCATION`
  - **Fuel-burn limits**
    - `FUEL_BURN_MIN_ABS_ALLOCATION`
    - `FUEL_BURN_MAX_ABS_ALLOCATION`
  - **Capacity-factor limits**
    - `CAP_FACTOR_MIN_ALLOCATION`
    - `CAP_FACTOR_MAX_ALLOCATION`
  - **Yearly storage state-of-charge allocations**
    - `STOR_START_SOC_ALLOCATION`
    - `STOR_END_SOC_ALLOCATION`

  The operational subproblem receives the corresponding allocation variables and uses its own subproblem's budget in place of the system-wide target where the policy module supports this substitution.

For yearly-balanced storage, `soc_allocation.py` defines the shared SOC allocation targets and variables used by both the master and operational subproblems. It applies to projects with `operational_type == "stor"` and `balancing_type_project == "year"`, and creates nonnegative start- and end-SOC allocation variables for each project, period and subproblem. The allocations are expressed in units of 1000 MWh (`STOR_SOC_ALLOCATION_UNIT_MWH = 1000.0`).

The master-side constraints are implemented in `gridpath/system/policy/benders_policy_constraint.py`. For each project, balancing type, period and subproblem, `Stor_SOC_Chain_Constraint` equates the end-SOC allocation of the current subproblem to the start-SOC allocation of the next subproblem. Subproblems are sorted numerically, and the final subproblem is connected back to the first, so the sequence is cyclic. `Stor_SOC_Capacity_Constraint` also limits each allocated start SOC, converted to MWh, to the project's energy-storage capacity. These two constraints are added to the master only; the operational models receive the corresponding allocation variables through `add_stor_soc_allocation_components` in `soc_allocation.py`.

Each worker keeps only the linking variables that exist in its model (`classify_linking_vars`) and splits them into continuous, binary and integer sets.

**Sign convention.** GridPath maximises NPV. The master maximises `NPV_plan − S·Σϑ`, so ϑ is a *cost* with a negative sign, and an optimality cut is a *lower* bound on ϑ.

## 3. Master (planning) problem

The master is the GridPath planning model with its NPV objective replaced by a version that carries one cost variable per subproblem. `add_approximate_variable_cost` (`planning.py:189-213`) deactivates `NPV`, stores its expression as `ePlanningNPV`, and defines `MasterObjective`:

$$
\begin{aligned}
\max_{x,\vartheta}\quad
& \mathrm{NPV}_{\mathrm{plan}}(x)
- S\sum_{w\in W}\vartheta_w \\
\text{s.t.}\quad
& x\in X_{\mathrm{master}} \\
& \vartheta_w\geq \frac{L_w}{S}
&& \forall w\in W \\
& \vartheta_w\geq
\frac{Q_w(\bar{x}^{k})}{S}
+\sum_{i\in I_w^{k}}
\frac{\lambda_{w,i}^{k}}{S}
(x_i-\bar{x}_i^{k})
&& \forall w\in W,\ k\in K_w^{\mathrm{opt}} \\
& c_w^{k}\bar{s}_w^{k}
+\sum_{i\in I_w^{k}}
\mu_{w,i}^{k}(x_i-\bar{x}_i^{k})
\leq 0
&& \forall w\in W,\ k\in K_w^{\mathrm{feas}}
\end{aligned}
$$

Discounting and period weights are whatever the original GridPath `NPV` expression contains; `planning.py` adds none. The helper refuses to run without an `NPV` objective and does nothing if `vTHETA` already exists.

**Variables**

- x: every variable of the planning model, with GridPath's own bounds and domains. The Benders code changes them in two places only: the Pecci–Jenkins relaxation of integer and binary variables (section 6) and an emergency cap (below).
- ϑ\_w (`vTHETA[w]`): real, with no upper bound. Its scale S = 10⁹ keeps cut coefficients near one.
- Three empty `ConstraintList`s filled during the run: `BendersOptimalityCuts`, `BendersFeasibilityCuts`, `BendersDiscreteOptimalityCuts`. A fourth, `ThetaLowerBound`, holds the bounds on ϑ.

**Lower bound on ϑ.** Each worker solves the LP relaxation of its operational model before it reports READY and returns `recourse_upper_bound` U\_w, an upper bound on its operational NPV. Because Q\_w = −NPV, the master receives

$$
L_w = -U_w, \qquad \vartheta_w \ge L_w / S \quad (\text{replaces } \vartheta_w \ge 0)
$$

The LP that produces U\_w is solved over the whole operational model with the linking variables free, so it needs finite bounds on those variables; see "Linking-variable bounds in the recourse LP" in section 8.

`set_recourse_cost_lower_bounds` (`planning.py:216-223`) clears and rebuilds these constraints. The bound is valid for every x but loose, and it can be negative. It matters again for the discrete cuts in section 6, where it sets the big-M. There is no upper bound on ϑ: the maximisation pushes ϑ down, and the cuts and L\_w stop it.

**Solving.** `solve_planning_problem` (`planning.py:39-125`):

1. solve; if the result is not optimal, retry once with a fresh solver object;
2. if the result is unbounded, set an upper bound of `BENDERS_MASTER_VARIABLE_CAP = 1e8` on every non-integer planning variable that has none, and re-solve (a heuristic for build variables that only appear in cuts);
3. if it still fails, raise `RuntimeError`, writing the LP to `$GRIDPATH_BENDERS_FAILURE_LP` first if that variable is set.

`process_planning_sol` returns a `PlanningSolution` with `values` (every planning variable, not rounded) and `planning_cost = value(ePlanningNPV)`, which excludes the ϑ term. The upper bound is `UB = value(MasterObjective)`; `benders()` keeps `UB = min(UB, UB_new)`, valid because cuts only tighten a maximisation master.

Cuts are appended to the model and the whole instance is re-passed to the solver each iteration; `planning.py` has no warm-start or incremental-cut logic (inferred for the native wrappers, which were not checked for warm starts).

## 4. Operational subproblems

Each subproblem is a GridPath operational model built once by `create_operational_problem_benders` and kept alive in its own worker process. It loads only the Operational modules and is flagged `_is_benders_subproblem = True`, so capacity becomes a linear expression in the linking variables. Constraints whose names also exist in the master (`get_shared_constraints`, matched by name) are deactivated so master-only rows are not duplicated.

**Linking equalities.** Every iteration `fix_linking_variables` rebuilds the `ConstraintList` `BendersLinkingConstraints` with one row per active linking variable:

$$
x_i = \bar x_i \qquad (i \in \text{active linking variables})
$$

The variables are constrained rather than fixed, so the solver returns a dual for each row. Targets are clipped to the variable's own bounds to remove round-off. A name missing from the received `planning_sol` raises a `KeyError`.

**Recourse value and dual scaling.** The LP is solved with the objective divided by `BENDERS_RECOURSE_OBJECTIVE_SCALE = 1e6` (restored afterwards), so every dual is multiplied back by that scale:

$$
Q_w(\bar x) = -\mathrm{OP\_NPV}^{*}, \qquad \lambda_i = -\pi_i \cdot 10^{6} = \partial Q_w / \partial \bar x_i
$$

Here π\_i is the solver's dual on row i (`process_subproblem_solution`, `subproblems.py:1458`). The sign flip converts a maximised NPV sensitivity into a cost sensitivity. Before a solution is accepted, `primal_violations` checks rows, bounds, linking equalities and integrality at 1e-7. On a violation `solve_original` retries with primal simplex, then interior point, on row-scaled clones.

**Which duals are used.** In the Pecci–Jenkins relaxed phase (section 6) all present linking variables are active. Afterwards only the *continuous* linking variables are, and binary or integer ones are handled by state-conditioned cuts.

**Result returned to the master** (one dict per subproblem per iteration):

| Key | Meaning |
| --- | --- |
| `feasible` | True if the LP solved to optimality (or the restoration retry showed feasibility) |
| `op_cost` | Q\_w(x̄) in dollars |
| `benders_lambda`, `continuous_lambda` | λ for the active linking variables (identical as written) |
| `continuous_point` | x̄ for those variables |
| `theta_coeff` | 1.0 for an optimality cut, 0.0 for a feasibility cut |
| `master_discrete_state` | Master binary and integer values, in the discrete phases only |
| `operational_discrete_state` | Operational integer values z\*, with `fixed_z` only |
| `is_mip`, `operational_mip_variables` | Whether non-linking discrete variables existed, and their names |

**Infeasible subproblems and restoration.** If the LP is not optimal, `restoration.restore` solves a Phase-I problem on a clone. `add_slacks_to_subproblem` (`subproblems.py:151-292`) relaxes every active row with a finite bound, except the linking equalities, by a multiple of one shared variable s\_max ≥ 0, and moves variable bounds into rows so they can be relaxed too. Each row's multiple is its own scale `_constraint_scale`, so a unit of s\_max is comparable across rows:

$$
\min 10^{6} s_{\max}
\quad \text{s.t.}\quad g_r(z) \le b_r + \sigma_r s_{\max} \forall r,\qquad x_L = \bar x, s_{\max} \ge 0
$$

where σ\_r is the row scale. The solver runs primal simplex, then interior point if that fails, and `_certificate` recomputes the dual objective and checks primal residuals, dual signs, stationarity and complementarity at 1e-8. If both algorithms fail the certificate, a `RuntimeError` is raised. The result is `feasible = False` with `theta_coeff = 0`; section 5 gives the cut. If the minimum slack is at most `ZERO_SLACK_TOLERANCE = 1e-9`, the original LP is re-solved and, if that succeeds, treated as feasible after all.

Restoration raises `NotImplementedError` if a non-linking integer variable is still unfixed, because LP duals would then give an invalid feasibility cut.

## 5. Cuts

All cuts are lower bounds on ϑ\_w, written in units of S. They share one routine, `add_relaxed_benders_cut` (`algorithms.py:1788-1860`), which appends the cut to the global `BendersOptimalityCuts` list; the subproblem key enters only through `vTHETA[w]`.

**Optimality cut.** Q\_w is convex in x\_L, so its supporting hyperplane at x̄ under-estimates it everywhere:

$$
Q_w(x) \ge Q_w(\bar x) + \lambda_w^{\top}(x_L - \bar x)
\Longrightarrow
\vartheta_w \ge \frac{Q_w(\bar x)}{S} + \sum_i \frac{\lambda_{w,i}}{S}(x_i - \bar x_i)
$$

The code adds `κ·ϑ_w − op_cost/S − Σ(λ_i/S)(x_i − x̄_i) ≥ 0` with κ = `theta_coeff`. The intercept is the solved `op_cost`, not a recomputed constant. Cuts with κ = 1 are added for every feasible result, in the relaxed phase and in the all-continuous case.

**Feasibility cut.** Let s̄ be the optimal Phase-I slack at x̄ and μ the duals of the linking rows, each divided by its row scale and by `RESTORATION_OBJECTIVE_SCALE = 1e6`. The minimum slack s(x) is convex and zero exactly when the subproblem is feasible, so

$$
s(x) \ge \bar s + \mu^{\top}(x_L - \bar x)
\Longrightarrow
\bar s + \mu^{\top}(x_L - \bar x) \le 0 \quad \text{is necessary for feasibility}
$$

The worker multiplies the intercept and duals by c = max(1, 1/s̄) so a small violation is not ignored, and returns `theta_coeff = 0`. With κ = 0 the routine uses a scale of 1 and writes `c·s̄ + Σ λ_i (x_i − x̄_i) ≤ 0` with no ϑ term. It goes into the same `BendersOptimalityCuts` list. In the relaxed phase, `benders()` also checks that a new restoration cut does not exclude an earlier LP-feasible sample and raises `RuntimeError` if it does.

**State-conditioned cuts for discrete linking variables.** Once the master's binary or integer variables are restored to their true domains, a dual cut computed at one integer pattern d is valid only at that pattern. `register_benders_state` stores per subproblem and iteration: `op_cost`, `theta_coeff`, `continuous_lambda`, `continuous_point`, `master_discrete_state` and the recourse lower bound. `add_state_recourse_cut` (`algorithms.py:2017-2404`) then adds the cut only when the master's discrete variables equal d, using a binary z that is 0 exactly on a match:

$$
\vartheta_w \ge \frac{Q_w(\bar x) + \lambda^{\top}(x_{cont} - \bar x_{cont}) - Mz}{S},
\qquad
M = \max\bigl(0, Q_w(\bar x) + \max_{x \in [lb,ub]} \lambda^{\top}(x - \bar x) - L_w\bigr)
$$

When z = 1 the right-hand side is at most L\_w/S, so the cut is slack there. Variable bounds for M come from `get_continuous_variable_bounds` (section 8). Mismatch indicators are built per variable:

- **Binary** x\_j with stored value d\_j: m\_j = 1 − x\_j if d\_j = 1, else m\_j = x\_j. Then z ≥ m\_j for every j and z ≤ Σ m\_j, so z = 0 only if every variable matches.
- **Integer** x\_j with stored value t: two binaries `below` and `above` with big-M constraints, and a mismatch variable bounded by their sum.

Each cut lives in a Pyomo `Block` named `BendersDiscretePattern_{key}_{id}`.

**No-good cut for infeasible integer patterns.** `add_combinatorial_feasibility_cut` (`algorithms.py:1416-1707`) excludes the visited discrete pattern entirely:

$$
\sum_j \text{mismatch}_j \ge 1
$$

It is used after the master has been made integer when a subproblem is infeasible and native disjunctions are not in use. For integers it builds two-sided big-M constraints that force `below` and `above` to 1 on the corresponding side, so it is exact. It assumes the discrete pattern alone makes the subproblem infeasible, so it can be stronger than needed (inferred; the code's own comment for the native path says a different continuous allocation might make the same pattern feasible).

**Native alternatives.** With native MOSEK, the state-conditioned cut is not built as big-M rows. States are passed in `planning_problem._benders_native_djc_data` and `mosek_benders.add_benders_djc` adds one disjunction per cut:

$$
\bigl[ m(d) = 0 \wedge \kappa\vartheta - \tfrac{Q}{S} - \tfrac{1}{S}\lambda^{\top}(x - \bar x) \ge 0 \bigr] \vee \bigl[ m(d) \ge 1 \bigr]
$$

with m(d) the Hamming distance to d. This needs no big-M. Feasibility states (κ = 0) use the same form.

**Cuts added per phase.**

| Phase | Condition | Cuts |
| --- | --- | --- |
| Continuous | no discrete linking variables | Optimality (and feasibility) cuts, every result |
| P&J relaxed | `pj_relax`, integrality not yet restored | Optimality and feasibility cuts on all linking variables, every result |
| P&J discrete | `pj_relax`, integrality restored | Register feasible states; state-conditioned cuts (big-M or native); no-good or native feasibility state if infeasible; plain cut for subproblems with no discrete linking variables |
| DJC | `djc` | Same as the discrete phase, from the first iteration |

## 6. Discrete variables

Two kinds of integer variable exist, and the code treats them separately: **master (linking) discrete variables**, such as build decisions on lumpy units, and **operational discrete variables**, such as unit commitment inside a subproblem. Dual cuts are valid only for linear subproblems, so both need a rule.

### Master discrete methods

`--benders_master_discrete_method` selects one of four values (default `pj_relax`):

| Method | Behaviour |
| --- | --- |
| `pj_relax` | **Two phases.** (1) Relax integers to the reals and binaries to $[0, 1]$ (`algorithms.py:342-351`), run ordinary Benders until the gap closes. (2) Restore the domains, reset LB to −∞, clear the incumbent, re-solve the master as a MIP, and continue with state-conditioned cuts (section 5). If the relaxed optimum is already integral the run stops as optimal. |
| `linearized` | Rewritten to `pj_relax` at `algorithms.py:280`, with `use_native_djc = False`, so state cuts use big-M rows. The separate `linearized_phase` code path is therefore never reached (inferred). |
| `djc` | The master is a MIP from the first iteration and state cuts are native disjunctions. Requires native MOSEK; any other solver raises `ValueError` (`algorithms.py:429-434`). Optional seeding below. |
| `none` | Plain Benders for continuous linking variables only. If discrete linking variables exist the code warns and falls back to `pj_relax`. |

The `pj_relax` label is the code's own; the document uses it without claiming the source reference. In the relaxed phase the dual cuts include the (relaxed) discrete linking variables, so they are valid for fractional x and therefore for every integer x. That is why the first phase can use plain cuts.

`use_native_djc` is true when the requested method is not `linearized` and the solver is native MOSEK. It sends states to `mosek_benders.add_benders_djc` instead of building big-M blocks.

**DJC seeding.** Cuts are valid only in the state where they are generated, so an unvisited state is bounded only by the loose L\_w. With `DjcSeedStates` true and 2^n ≤ `DjcSeedMaxStates` (default 1024) binary linking variables, `benders()` builds one seed per joint binary state, with continuous linking variables set to zero, and evaluates each before the first master solve. Seeding is off by default and has no CLI flag; set it through `GRIDPATH_BENDERS_SETUP`.

### Operational discrete methods

`--benders_operational_discrete_method` selects how non-linking integers inside a subproblem are handled (CLI default `fixed_z`):

- **`pj_relax`**: `prepare_lp_for_duals` relaxes non-linking binaries to $[0, 1]$ and integers to the reals, and the dual comes from the LP relaxation. The relaxed value is at most the MIP value (in NPV terms), so the cut is valid as an LP-relaxation cut but looser.
- **`fixed_z`**: `solve_operational_mip_and_fix_z` solves the actual MIP with the linking equalities in place, checks that each integer is within 1e-8 of integral, fixes them at z\*, and re-solves the LP to get duals. The cut is then conditional on z\*, which is why `operational_discrete_state` is returned. `restore_lp_after_duals` and `restore_operational_discrete_variables` undo the changes because the instance is persistent. If the subproblem has no operational integers, it is solved directly.

During the master's relaxed phase the worker overrides the method to `pj_relax` (`subproblems.py:851-855`), because only the LP relaxation gives a globally valid cut there.

$$
\text{fixed\_z cut:}\quad Q_w(x) \ge Q_w(\bar x \mid z^{*}) + \lambda^{\top}(x_L - \bar x) \quad \text{valid only for fixed } z^{*}
$$

Inferred from the formulas, not stated in the code: at other x the MIP may choose a different z with lower cost, so a fixed-z cut can overestimate the true MIP cost. The resulting UB should then be treated as heuristic for subproblems that contain operational integers, while `pj_relax` keeps UB valid at the price of a looser bound.

## 7. Stabilisation (level sets)

The raw master optimum often jumps between extreme investment plans. Stabilisation chooses the *next trial point* by solving a second problem restricted to plans whose master objective is close enough to UB, and picking the one nearest to the last evaluated point. It does not change the master's cuts or its UB; the master is still solved first.

**Level set.** With a maximisation master, the level is

$$
\ell = UB - \gamma(UB - LB), \qquad \gamma \in (0,1),
\qquad
\text{stabilised point: } \arg\min_{x,\vartheta} \Phi(x) \text{s.t.} \mathrm{MasterObjective}(x,\vartheta) \ge \ell
$$

A small γ keeps the level near UB (tight); a large γ moves it toward LB (loose). The constraint is `cLevelSet`, and `LevelObjective` replaces `MasterObjective` for the solve and is removed afterwards (`regularization.py:268-510`). The default `--benders_stab_param` is 0.025; γ = 0 with L1 or L2 raises an error.

| `--benders_stabilization_method` | Φ(x) | Notes |
| --- | --- | --- |
| `none` | MasterObjective itself, minimised over the level set | Drives the point to the level boundary. If MOSEK fails, retries once with dual simplex |
| `l1` | Σ w\_i (p\_i + n\_i) with x\_i − x̄\_i = p\_i − n\_i, p, n ≥ 0 | Linear. Auxiliary variables and rows are built once and only right-hand sides change |
| `l2` | Σ w\_i (x\_i − x̄\_i)² | Quadratic; with MOSEK (or `djc`) a rotated quadratic cone on an epigraph variable t is used |

**Centre and weights.** The centre x̄ is the point the workers *just evaluated*, not the new master optimum (centring on the optimum would make the penalty zero there). Weights are all 1 in practice (`stabilization_weights=None`).

**What is penalised.** Only the *continuous* linking variables. Binary and integer variables are free in the level-set problem, so after integrality is restored the level-set problem is a MIP. If there are no continuous linking variables, Φ is zero and the solve only finds a feasible point.

**Fallbacks.** A failed level-set solve prints a warning and returns the plain master optimum, the unstabilised point. The stabilisation is also skipped on alternate iterations when the run is near the noise floor: `len(gap_hist) > 6`, `gap < 1e-3`, `gap_hist[-1] > 0.5·gap_hist[-6]` and an even history length (`algorithms.py:1120-1123`).

**Dynamic γ.** With `--benders_stab_dynamic` true (default) and k ≥ 1, `update_stab_param` (`algorithms.py:2968-2983`) compares the actual drop in UB with the drop the previous level predicted:

$$
ap = UB_{old} - UB, \qquad
pp = UB_{old} - \bigl(LB + \gamma(UB_{old} - LB)\bigr), \qquad r = ap/pp
$$

$$
\gamma \leftarrow
\begin{cases}
0.45 + 0.5\gamma & r \le 0.2 \quad (\text{poor progress: looser level, towards } 0.9)\\
0.5\gamma & r \ge 0.8 \quad (\text{good progress: tighter level})\\
\gamma & \text{otherwise}
\end{cases}
$$

The update applies only when ap ≥ 0 and pp ≥ 0. The meaning of tight and loose is inferred from the level formula. Two sharp edges are listed in section 13: r is computed before the guard, so pp = 0 raises `ZeroDivisionError`, and γ is not clipped.

## 8. Bounds, big-M and numerical scaling

Bounds matter in three places: the lower bound L\_w on each subproblem value (which keeps the first master solves bounded), the big-M in state-conditioned cuts (section 6), and variable bounds used by the level-set problem and the cut generators.

**Scaling constants.**

| Constant | Value | Where it acts |
| --- | --- | --- |
| `BENDERS_THETA_SCALE` (S) | 1e9 | Master carries ϑ\_w = (value)/S, so ϑ is O(1) |
| `BENDERS_RECOURSE_OBJECTIVE_SCALE` | 1e6 | Operational objective is divided by this; duals are multiplied back by it (λ = −dual · scale) |
| Phase-I elastic weight | 1e6 | Objective min 1e6 · s\_max in the restoration LP |
| Certificate tolerance | 1e-8 | Dual-ray entries below it are treated as zero in feasibility cuts |
| Variable-bound cap | 1e8 | Last-resort bound for an otherwise unbounded continuous linking variable; when it is used it is also enforced in the master and applied in each worker's recourse LP |

**Subproblem lower bound L\_w.** At worker startup the LP relaxation of the operational model is solved for an upper bound on the operational NPV; its negation, `−recourse_upper_bound`, is L\_w. The master then imposes ϑ\_w ≥ L\_w/S, which is valid because every real operational NPV is at most the relaxation's. The linking variables in that LP are bounded using the master-derived bounds described next.

**Linking-variable bounds in the recourse LP.** The worker's recourse LP is a clone of the operational model taken before the `x = x̄` linking constraints exist. The master's constraints that are shared by name are deactivated in it (section 4), and it has no investment cost. A linking variable with no explicit upper bound can therefore grow without limit, and because `OP_NPV` is maximised and includes revenue terms (for example sold carbon credits) the LP can be unbounded. In a failing run every linking variable reported `lb = 0, ub = None`: `GenNewLin_Build_MW`, `StorNewLin_Build_MW` and `_MWh`, `TxNewLin_Build_MW`, `BENDERS_CARBON_CREDIT_GENERATED_TOTAL` and `BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION`. MOSEK reported the LP as unbounded and the worker raised `RuntimeError`. Which family drives the ray was not isolated; the carbon-credit variables are the most likely cause because they map directly to revenue.

The fix takes the bounds from the master and applies them to the recourse LP only:

1. `benders()` calls `compute_linking_variable_bounds` after the solver is created and before any worker is started. For every continuous linking variable present in the master it calls `get_continuous_variable_bounds` (sources 1 to 4 above) and collects `{name: (lb, ub)}`. Binary and integer linking variables are skipped, as are variables for which no lower bound can be established (those raise `ValueError` inside the helper, which is logged as a warning and skipped).
2. Any variable that ends up on the 1e8 cap (recorded in `planning_problem._benders_assumed_bounds`) also gets that upper bound set on the master variable. Without this the bound used in the workers would not be valid at every master point.
3. The dictionary is passed to each worker as the last element of the `Process` arguments (`linking_var_bounds`).
4. `calculate_recourse_upper_bound(instance, optimiser, linking_var_names, linking_var_bounds)` clones the model, relaxes operational integrality, and calls `apply_linking_variable_bounds`. That function sets a lower or upper bound on the clone only where the variable has none, so an existing explicit bound is never loosened. It returns the names that still lack a bound.
5. If any variable still lacks a bound, a warning lists the count and the first ten names. If the LP is then not optimal, the `RuntimeError` message also gives the count and the first ten names.

The original operational instance is not changed, so the `x = x̄` solves in the worker loop are unaffected.

Setup key `RecourseBoundOBBT` (default `True`) controls whether step 3 of the bound sources (OBBT on the master) is tried before falling back to the 1e8 cap. OBBT costs one master solve per linking variable that has neither an explicit bound nor a capacity-constraint bound. Set it to `False` to skip those solves; those variables then take the 1e8 cap directly.

Consequences to keep in mind. A variable on the 1e8 cap gives a valid but very loose bound, so U\_w is large, L\_w is very negative, and M\_w in the big-M formula below grows with it. The log line `Using assumed Benders upper bound ... for <name>` marks each such variable, and the INFO line `Linking-variable bounds: N variables bounded (M with an assumed cap ...)` summarises them. Giving those variables real upper bounds in the master tightens everything downstream.

**Variable bounds** (`get_continuous_variable_bounds`). For a continuous linking variable the bound is taken from the first source that gives one:

1. an explicit bound on the Pyomo variable;
2. a capacity-based bound from `CAPACITY_BOUND_CONSTRAINTS` (`find_gridpath_capacity_upper_bound`), derived from the model's own capacity constraints;
3. optimisation-based bound tightening (`obbt_upper_bound`), i.e. maximising the variable over the master feasible set;
4. the 1e8 cap.

The exact capacity-bound formula is not reproduced here; read `find_gridpath_capacity_upper_bound` in the code for it.

**Dual-term bound.** `continuous_dual_term_upper_bound` bounds λᵀ(x − x̄) over the box of variable bounds, i.e. Σ\_i max(λ\_i(l\_i − x̄\_i), λ\_i(u\_i − x̄\_i)). This is the term that enters the big-M below.

**Big-M for state-conditioned cuts.**

$$
M_w = \max\Bigl(0, \mathrm{op\_cost}_w + \max_{x}\lambda^{\top}(x-\bar x) - L_w\Bigr)
$$

where op\_cost\_w = −OP\_NPV at x̄. M\_w is the largest amount by which the cut's right-hand side can exceed the global lower bound, so deactivating the cut (z = 0) leaves ϑ\_w ≥ L\_w/S effective. A loose L\_w or loose variable bounds inflate M\_w and weaken the LP relaxation of the master.

**Numerical notes (inferred).** Dividing by S and 1e6 keeps coefficients near O(1) for MOSEK/Gurobi, but a cut whose λ entries are all below tolerance can look numerically empty; the feasibility cut rescaling c = max(1, 1/s̄) guards the analogous case there.

## 9. Algorithm, convergence and termination

**Main loop** (`algorithms.py`, the Benders driver). Notation: UB is the best master objective (an upper bound, since the master is a relaxation of a maximisation), LB is the best evaluated true objective (the incumbent).

$$
\begin{aligned}
&\textbf{Init: } UB=+\infty, LB=-\infty, \gamma=\gamma_0, \text{derive linking-variable bounds from the master, start one worker per scenario key, collect } L_w \\
&\textbf{Solve master once} \to \bar x^{0} \\
&\textbf{for } k = 0,1,\dots,\mathrm{MaxIter}: \\
&\quad 1.\ \text{send } \bar x^{k} \text{ to all workers (sliding window)} \\
&\quad 2.\ \text{collect } (Q_w, \lambda_w) \text{ or feasibility certificates} \\
&\quad 3.\ \text{true objective } f(\bar x^k) = NPV_{plan}(\bar x^k) - \textstyle\sum_w Q_w \\
&\quad 4.\ \text{if all feasible and } f(\bar x^k) > LB:\ LB \leftarrow f(\bar x^k),\ \text{incumbent} \leftarrow \bar x^k \\
&\quad 5.\ \text{add optimality / feasibility cuts to master} \\
&\quad 6.\ \text{re-solve master} \to UB,\ x^{*} \\
&\quad 7.\ \text{update } \gamma \text{ (if dynamic)} \\
&\quad 8.\ \text{if stabilised: solve level-set problem around } \bar x^{k} \to \bar x^{k+1}; \text{ else } \bar x^{k+1}=x^{*} \\
&\quad 9.\ \text{test termination}
\end{aligned}
$$

The step numbering above is a condensed reading of the loop; the code splits some steps (for example discrete-state cut bookkeeping and the history appends) that are omitted here.

**Discrete two-phase structure.** With `pj_relax` (the default master method) the loop first runs with the master integrality relaxed. When that phase ends, LB and the incumbent are reset, the master is re-solved as a MIP, and the loop continues with state-conditioned cuts (section 6). In the second phase the master is a MIP at every iteration.

**Gap.**

$$
\text{gap} = \frac{|UB - LB|}{\max(|UB|,10^{-10})}
$$

**Termination.** The first condition met ends the loop:

| Status | Condition | Default |
| --- | --- | --- |
| Converged | gap ≤ ConvTol and LB ≤ UB + ConvTol · \|UB\| | ConvTol = 1e-10 |
| Stalled | gap improves by less than StallGap over StallIter iterations | 20 iterations, 1e-6 |
| Time limit | wall-clock time exceeds MaxCpuTime | 3600 s |
| Max iterations | k reaches MaxIter | 200 |

`AbsGapTol` (default 1e-4) is used only for display. `MaxCpuTime` is wall-clock time despite its name. The exact stall test is paraphrased from the research notes; check the code before relying on the wording.

**After the loop.** Unless `--benders_skip_final_export` is set, the driver sends EXPORT\_FINAL so each worker writes its own results at the incumbent. The incumbent investment decisions are written back into the planning instance, and the result status is mapped to OPTIMAL, TIMELIMIT, MAXITER or NONE. A STOP message then shuts the workers down.

**Result object.** `BendersResult` holds the planning results, the status, histories of UB, LB and gap, and `final_export_status` (added in the patch described in section 11).

## 10. Parallel architecture

**Processes.** The master lives in the driver process. Each operational subproblem, keyed by ScenarioKey(weather, hydro, availability, subproblem, stage), runs in its own persistent worker process created with the `spawn` start method (`subproblems.py`, `benders_subproblem_worker`). A worker builds its Pyomo model once, keeps it in memory, and on each iteration only updates the fixed linking values and re-solves.

**Messages.** Each worker has an input and an output queue. The driver sends the planning solution and receives the objective, duals or a feasibility certificate. The main message types are:

| Message | Direction | Effect |
| --- | --- | --- |
| READY | worker → driver | Model built; carries L\_w (the LP-relaxation bound) |
| solve request | driver → worker | Contains the light planning solution (`_light_planning_sol`, only linking-variable values) |
| result | worker → driver | Q\_w, λ, or feasibility data |
| EXPORT\_FINAL | driver → worker | Writes the worker's results for the incumbent (`_export_final_subproblem`) |
| STOP | driver → worker | Worker exits |

**Startup.** All workers are started first, then READY messages are collected. A semaphore, `ctx.Semaphore(worker_batch_size)`, limits how many workers build their model and solve the startup LP at once; each worker releases its permit just before sending READY (`_StartupGuard`).

**Dispatch window.** `_dispatch_window` keeps a bounded number of requests in flight, collects results as they arrive, returns them in the original order, and detects dead workers. A test script exercised order preservation and dead-worker detection.

**Memory.** All N operational models stay resident for the entire run, so memory scales with the number of scenario keys. This is the main cost of the persistent-worker design.

**Solvers.** Per-worker solver is chosen by `--benders_operational_solver` (default `automatic`). Wrappers exist for Gurobi, MOSEK and a GAMS adapter that contains no cut logic.

## 11. Results and reporting

**Where results go.** The planning (investment) results are written at the scenario root, `<scenario>/results/`, through `save_results` with empty weather, hydro, availability, subproblem and stage. Per-subproblem results are written by each worker on EXPORT\_FINAL into `weather/hydro/availability/subproblem/stage/results`.

**Convergence files.** `write_benders_convergence_files` writes `results/benders_convergence.csv` (iteration histories) and `results/benders_summary.json` (final status, bounds, gap) before `save_results` runs.

**Limits.** EXPORT\_FINAL skips duals when the subproblem is a MIP. `summarize_results` is still commented out in the code.

**Relation to the non-Benders run.** The same `result*.csv` files are produced at the scenario root, but the operational-level files come from the subproblem directories rather than a single combined model.

**Logging and incomplete runs.** With `--log` a log is opened for the planning-level run, and `--incomplete_only` skips the run if the root results export is complete. Both are described in the reporting patch; stdout and stderr are restored in a `finally` block.

## 12. Configuration reference

| CLI flag | Setup key | Default | Meaning |
| --- | --- | --- | --- |
| `--max_benders_iter` | MaxIter | 200 | Iteration cap |
| `--benders_conv_tol` | ConvTol | 1e-10 | Relative gap tolerance |
| `--benders_abs_gap_tol` | AbsGapTol | 1e-4 | Display only |
| `--max_benders_cpu_time` | MaxCpuTime | 3600 | Wall-clock limit (s) |
| `--benders_stab_param` | StabParam | 0.025 | Initial γ |
| `--benders_stab_dynamic` | StabDynamic | True | Adapt γ each iteration |
| `--benders_stabilization_method` | StabilizationMethod | none | none, l1 or l2 |
| `--benders_master_discrete_method` | MasterDiscreteMethod | pj\_relax | none, pj\_relax, linearized (rewritten to pj\_relax), djc |
| `--benders_operational_discrete_method` | OperationalDiscreteMethod | fixed\_z | pj\_relax or fixed\_z |
| `--benders_operational_solver` | n/a | automatic | Solver for workers |
| `--benders_skip_final_export` | n/a | False | Skip EXPORT\_FINAL (patch) |
| none | DjcSeedStates | False | Seed DJC states |
| none | DjcSeedMaxStates | 1024 | Cap on seeded states |
| none | StallIter | 20 | Stall window |
| none | StallGap | 1e-6 | Stall improvement threshold |
| none | RecourseBoundOBBT | True | Try OBBT on the master when bounding linking variables for the recourse LP (section 8); False goes straight to the 1e8 cap |

The environment variable `GRIDPATH_BENDERS_SETUP` holds a JSON object that is merged over the setup dictionary at run time (`run_scenario.py`, around line 2575), so it can set the keys that have no CLI flag.

## 13. Assumptions, limitations and open issues

Everything below comes from reading the code. **None of it was run**, and items marked *inferred* are my reading, not confirmed behaviour.

**Correctness questions**

- *Fixed-z cuts (inferred).* At a different x the operational MIP may choose a different integer state with lower cost, so a fixed-z cut can overestimate the true MIP cost.
- *Integer cuts (inferred).* The integer below/above constraints in `add_state_recourse_cut` look one-directional, which could leave a validity gap.
- *Infeasible subproblem, no discrete linking variables.* In the discrete phase it is silently skipped.
- *`benders_states`* is appended every iteration with no visible de-duplication.
- *Level-set `continue`* skips `planning_sol_hist.append`.
- *DJC* sets `master_integrality_restored` to False (around line 373), which may be inconsistent with the later phase logic.

**Recourse-LP bounds (section 8).** The change that bounds linking variables in the recourse LP was compile-checked only; it has not been run on a scenario.

- OBBT runs on the master before the workers start, once per linking variable with no explicit or capacity-constraint bound. For a large master this adds startup time. If OBBT returns nothing (for example the master is unbounded in that variable), the variable takes the 1e8 cap, and the only trace is the warning in the log.
- `compute_linking_variable_bounds` bounds only variables that exist in the master. A linking variable that exists only in a subproblem is left to whatever bounds it already has, and is reported as still unbounded.
- `relax_operational_integrality` replaces a general-integer variable's domain with `Reals`, which drops bounds that came from the domain (for example the lower bound of `NonNegativeIntegers`) even though its docstring says bounds are preserved. A strict LP has no such variables. A model with operational integers should read `var.bounds` first and set them again after the domain change.
- `subproblems_temp.py` and `benders.py` appear to be older copies of the worker and driver code. They were not changed, and `benders.py` still starts workers with the old argument list; the new `linking_var_bounds` argument defaults to `None`, so they still run but do not get the bounds.

**Numerical edge cases**

- `update_stab_param` can raise `ZeroDivisionError` when pp = 0 and does not clip γ.
- `add_state_recourse_cut` (around line 2156) passes a string name to `get_continuous_variable_bounds`, which expects a Var, for integer linking variables.

**Dead or unused code**

- `add_discrete_pattern_indicator`, `add_binary_pattern_constraints`, `add_integer_pattern_constraints` have no callers; the `linearized` phase is dead.
- The Gurobi native indicator path is never enabled, since `_benders_native_indicator_data` is never set.
- DJC and native paths raise `NotImplementedError` for non-binary integer linking variables.
- Solver wrappers do no warm starts.

**Documentation mismatches**

- `AbsGapTol` is display-only; `MaxCpuTime` is wall time.
- The CLI help for the master discrete method omits `none` and `linearized`.
- `OperationalDiscreteMethod` defaults to pj\_relax in the `.get` call but fixed\_z on the CLI; the CLI wins.
- The L2 docstring mentions division by objective\_scale² that is not applied.
- The label "P&J" / pj\_relax is the code's own; I did not verify the attribution to a paper.

**Scope**

- Memory holds all N operational models for the whole run (section 10).
- This text describes the code at HEAD of `new_gridpath_w_BD_imp` plus my reporting and efficiency patches where stated.
