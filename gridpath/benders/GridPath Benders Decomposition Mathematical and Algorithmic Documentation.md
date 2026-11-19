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

The full problem maximises planning NPV minus the operating cost summed across all subproblems. The decomposition replaces each operating cost by a cost function \(Q_w(x_L)\) of the investment decisions, which is convex and piecewise linear when the subproblem is a linear program.

$$
\begin{aligned}
\max_{x \in X}\quad & \mathrm{NPV}_{\mathrm{plan}}(x) - \sum_{w \in W} Q_w(x_L) \\
Q_w(\bar{x})\quad &= -\max_{\substack{z_w \in Z_w \\ x_L = \bar{x}}}\ \mathrm{OP\_NPV}_w(z_w)
\end{aligned}
$$

| Symbol | Meaning in the code |
| --- | --- |
| \(x\) | All planning (master) variables, `all_variables(planning_problem)` |
| \(x_L \subseteq x\) | **Linking variables**: the planning variables that appear in subproblems (section below) |
| \(w \in W\) | A subproblem key `ScenarioKey(weather, hydro, availability, subproblem, stage)` |
| \(\bar{x}\) | Trial point: the master values of \(x_L\) sent to the workers (`continuous_point`) |
| \(Q_w(\bar{x})\) | Recourse cost, `op_cost = −OP_NPV` in dollars |
| \(\vartheta_w\) | `vTHETA[w]`, the master's approximation of \(Q_w\), in units of \(S=10^9\) dollars |
| \(S\) | `BENDERS_THETA_SCALE = 1e9` (`planning.py:19`) |
| \(\lambda_w\) | Sensitivity of \(Q_w\) to \(x_L\), from the dual of the linking equalities |
| \(L_w\) | Global lower bound on \(Q_w\), from the LP-relaxation upper bound of operating NPV |
| \(UB\), \(LB\) | Master optimum (non-increasing) and best complete plan's NPV (non-decreasing) |

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

**Sign convention.** GridPath maximises NPV. The master maximises \(\mathrm{NPV}_{\mathrm{plan}}-S\sum_w\vartheta_w\), so \(\vartheta\) is a *cost* with a negative sign, and an optimality cut is a *lower* bound on \(\vartheta\).

## 3. Master (planning) problem

The master is the GridPath planning model with its NPV objective replaced by a version that carries one cost variable per subproblem. `add_approximate_variable_cost` (`planning.py:189-213`) deactivates `NPV`, stores its expression as `ePlanningNPV`, and defines `MasterObjective`:

```math
\max_{x,\vartheta}
\quad \mathrm{NPV}_{\mathrm{plan}}(x)
- S\sum_{w\in W}\vartheta_w
```

subject to

$$
x \in X_{\mathrm{master}}
$$

$$
\vartheta_w \geq \frac{L_w}{S} \qquad \forall w\in W
$$

$$
\vartheta_w \geq \frac{Q_w(\bar{x}^{k})}{S} + \sum_{i\in I_w^{k}} \frac{\lambda_{w,i}^{k}}{S}(x_i-\bar{x}_i^{k}) \qquad \forall w\in W,\ k\in K_w^{\mathrm{opt}}
$$

$$
c_w^{k}\bar{s}_w^{k} + \sum_{i\in I_w^{k}} \mu_{w,i}^{k}(x_i-\bar{x}_i^{k}) \leq 0 \qquad \forall w\in W,\ k\in K_w^{\mathrm{feas}}
$$

Discounting and period weights are whatever the original GridPath `NPV` expression contains; `planning.py` adds none. The helper refuses to run without an `NPV` objective and does nothing if `vTHETA` already exists.

**Decision variables**

- \(x\): the complete vector of planning (master) variables, with GridPath's original bounds and domains. The Benders implementation changes these domains only for the Pecci–Jenkins relaxation of integer and binary variables (section 6); it can also impose the emergency upper cap described below.
- \(x_i\): component \(i\) of the planning vector. In the cuts, only the linking-variable components represented in the relevant worker's cut are included.
- \(\vartheta_w\) (`vTHETA[w]`): one real master variable for each subproblem \(w\), with no upper bound. It approximates the recourse cost in units of \(S\).

**Other symbols appearing in the master formulation**

- \(S\): the scaling constant `BENDERS_THETA_SCALE = 1e9`; monetary cut values are divided by \(S\) before entering the master.
- \(W\), \(w\): the set of operational subproblems and one subproblem key, respectively. A key identifies the scenario dimensions and subproblem/stage.
- \(X_{\mathrm{master}}\): the feasible region of the planning model, including its original constraints and variable bounds/domains.
- \(L_w\): a global lower bound on the recourse cost for subproblem \(w\), calculated as the negative of the worker's LP-relaxation upper bound on operating NPV.
- \(Q_w(\bar{x}^{k})\): recourse cost at the trial point for subproblem \(w\) in iteration \(k\), equal to minus the operational NPV returned by the worker.
- \(\bar{x}^{k}\), \(\bar{x}_i^{k}\): the trial values of the linking variables sent to the worker in iteration \(k\), and the value of component \(i\), respectively.
- \(I_w^{k}\): the indices of the linking-variable components included in the cut for subproblem \(w\) at iteration \(k\).
- \(\lambda_{w,i}^{k}\): optimality-cut coefficient for linking variable \(i\); it is the corresponding sensitivity of recourse cost, obtained from the duals of the linking equalities.
- \(K_w^{\mathrm{opt}}\), \(K_w^{\mathrm{feas}}\): the iterations at which an optimality cut or feasibility cut, respectively, has been generated for subproblem \(w\).
- \(\bar{s}_w^{k}\): the optimal Phase-I restoration slack at the trial point for a feasibility cut.
- \(c_w^{k}\): the feasibility-cut scaling factor, \(c_w^{k}=\max(1,1/\bar{s}_w^{k})\).
- \(\mu_{w,i}^{k}\): feasibility-cut coefficient for linking variable \(i\), obtained from the Phase-I restoration duals after row-scale and restoration-objective-scale normalization.

**Constraint containers**

The following Pyomo `ConstraintList` components are populated as the algorithm generates cuts:

- `BendersOptimalityCuts`: continuous recourse optimality cuts and, in the current implementation, feasibility cuts.
- `BendersFeasibilityCuts`: reserved feasibility-cut container (not the destination used by the shared cut routine described in section 5).
- `BendersDiscreteOptimalityCuts`: state-conditioned optimality cuts for discrete master states.
- `ThetaLowerBound`: lower-bound constraints \(\vartheta_w\geq L_w/S\).

**Lower bound on \(\vartheta_w\).** Each worker solves the LP relaxation of its operational model before it reports READY and returns `recourse_upper_bound` \(U_w\), an upper bound on its operational NPV. Because \(Q_w=-\mathrm{OP\_NPV}\), the master receives

$$
L_w = -U_w, \qquad \vartheta_w \ge L_w / S \quad (\text{replaces } \vartheta_w \ge 0)
$$

The LP that produces \(U_w\) is solved over the whole operational model with the linking variables free, so it needs finite bounds on those variables; see "Linking-variable bounds in the recourse LP" in section 8.

`set_recourse_cost_lower_bounds` (`planning.py:216-223`) clears and rebuilds these constraints. The bound is valid for every \(x\), but can be loose, and it can be negative. It matters again for the discrete cuts in section 6, where it sets the big-M. There is no upper bound on \(\vartheta_w\): the maximisation pushes \(\vartheta_w\) down, and the cuts and \(L_w\) stop it.

**Solving the master.** `solve_planning_problem` (`planning.py:39-133`) calls the selected Pyomo solver interface on the current master model. A solve is accepted only when the solver status is `ok` and the termination condition is `optimal`.

1. **Initial solve and retry.** The function first calls `optimiser.solve(instance)`. If the result is not optimal, it obtains the name of the existing Pyomo solver interface, constructs another interface object with `SolverFactory(solver_name)`, checks that the new interface is available, copies the original interface's options, and tries the same Pyomo model again. A **fresh solver object** therefore means a newly constructed Pyomo solver wrapper/plugin instance. It does not mean that the optimization model is rebuilt, that a different solver is selected, or that a completely new external solver process is guaranteed. The retry is conditional on the original interface having a usable name and the new interface being available. There is one retry, not an indefinite sequence of retries or a systematic switch to another algorithm.

2. **Unboundedness fallback and the \(10^8\) cap.** If the termination condition after the retry is `unbounded` or `infeasibleOrUnbounded`, the function examines the active planning variables whose names occur in `planning_variables`. It adds the upper bound `BENDERS_MASTER_VARIABLE_CAP = 1e8` only to variables that are continuous (not integer or binary) and currently have no upper bound. Any upper bound already supplied by GridPath is left unchanged, and integer and binary variables are not capped by this fallback. The model is then solved again using the original `optimiser` object. This is a safeguard against an unbounded master, often involving continuous build variables whose effective restrictions are expected to arise through Benders cuts. It is **not** a general upper bound imposed on all planning variables during initialization. The value \(10^8\) is an arbitrary large cap, not a physically or mathematically established bound; a successful solve after adding it does not prove that the cap is non-binding. If no eligible variable is found, this fallback does not add a cap or perform that additional solve.

3. **Failure handling.** After the retry and any applicable unboundedness fallback, the function again requires solver status `ok` and termination condition `optimal`. If either condition is not met, it writes the current master model to the path specified by the `GRIDPATH_BENDERS_FAILURE_LP` environment variable, if set, using symbolic solver labels, and then raises `RuntimeError` with the final solver status and termination condition. The LP file is a diagnostic aid; exporting it does not repair the model.

**How the master upper bound is initialized and updated.** The algorithm's upper bound \(UB\) depends on the initialization path. This is separate from the \(10^8\) variable cap described above: that cap is a fallback applied only after an unbounded solve, whereas \(UB\) is the algorithmic bound used to track the best master relaxation.

- **Ordinary initialization (without Pecci–Jenkins initialization).** Before the first full master solve, the workers provide a global upper bound \(U_w\) on operating NPV for each subproblem \(w\). Since recourse cost is defined as \(Q_w=-\mathrm{OP\_NPV}_w\), the algorithm converts this into a recourse-cost lower bound \(L_w=-U_w\). It imposes \(\vartheta_w\ge L_w/S\) through `ThetaLowerBound`. The first master is solved with the full Benders objective, including the recourse-cost term \(-S\sum_w\vartheta_w\), and its objective value initializes \(UB\). This is a valid upper bound for the maximization problem because the master uses valid lower bounds on recourse cost and therefore relaxes the full problem.

- **Pecci–Jenkins initialization.** When `UsePjInitialization` is enabled and there are no discrete linking variables, the initialization phase temporarily solves a planning-only master with objective `ePlanningNPV`, omitting the recourse-cost term. The algorithm sets \(UB=+\infty\) during this phase. The planning-only objective is not a valid upper bound on the full planning-plus-recourse objective, so it must not be used as one. The value remains infinite while feasibility-restoration iterations seek a trial plan for which the operational subproblems are feasible and can provide optimality-cut information. Once that phase is complete, the full Benders objective is restored; a subsequent solve of that full master provides a valid objective value for initializing or updating \(UB\).

- **Subsequent full master solves.** Whenever the full Benders master is solved, its objective value is \(UB_{\mathrm{new}}=\mathrm{value}(\mathrm{MasterObjective})\). The algorithm updates the running bound as \(UB\leftarrow\min(UB,UB_{\mathrm{new}})\). Taking the minimum retains the tightest upper bound found for this maximization problem. During Pecci–Jenkins feasibility restoration, \(UB\) remains \(+\infty\); the planning-only objective is not used for convergence or as a substitute upper bound.

The \(10^8\) cap and \(UB\) therefore serve different purposes. The cap is a heuristic bound on selected master variables added in response to unboundedness. \(UB\) is an objective bound maintained by the Benders algorithm; outside the Pecci–Jenkins initialization phase, it comes from solving the full master with the recourse-cost term included.

**Solution returned from the master.** `process_planning_sol` packages the planning-variable values in a `PlanningSolution`. It returns initialized values without rounding integer or binary variables. An uninitialized variable may be omitted only if no operational subproblem requires it; a required linking variable that is missing or has no value causes an error. The separate `planning_cost = value(ePlanningNPV)` is planning-only NPV and excludes the \(\vartheta\)-weighted recourse-cost term. By contrast, \(UB\), when evaluated from `MasterObjective` under the full Benders objective, includes that term.

**Cut re-solves and solver state.** Benders cuts are appended to the Pyomo model as the algorithm progresses, and the master is solved again using the updated instance. The Python code in `planning.py` calls `optimiser.solve(instance)`; it does not explicitly maintain a persistent solver-side model or send incremental cut updates. Whether a particular solver interface internally reuses a process, basis, or other state is solver-interface-specific and cannot be concluded from this code alone. This function also does not explicitly request a warm start.
## 4. Operational subproblems

Each subproblem is a GridPath operational model built once by `create_operational_problem_benders` and kept alive in its own worker process. It loads only the Operational modules and is flagged `_is_benders_subproblem = True`, so capacity becomes a linear expression in the linking variables. Constraints whose names also exist in the master (`get_shared_constraints`, matched by name) are deactivated so master-only rows are not duplicated.

**Linking equalities.** Every iteration `fix_linking_variables` rebuilds the `ConstraintList` `BendersLinkingConstraints` with one row per active linking variable:

$$
x_i = \bar x_i \qquad (i \in \text{active linking variables})
$$

The variables are constrained rather than fixed, so the solver returns a dual for each row. Targets are clipped to the variable's own bounds to remove round-off. A name missing from the received `planning_sol` raises a `KeyError`.

**Recourse value and dual scaling.** The LP is solved with the objective divided by `BENDERS_RECOURSE_OBJECTIVE_SCALE = 1e6` (restored afterwards), so every dual is multiplied back by that scale:

$$
Q_w(\bar{x})=-\mathrm{OP\_NPV}^{*},\qquad \lambda_i=-10^6\pi_i=\frac{\partial Q_w}{\partial\bar{x}_i}
$$

Here \(\pi_i\) is the solver's dual on row i (`process_subproblem_solution`, `subproblems.py:1458`). The sign flip converts a maximised NPV sensitivity into a cost sensitivity. Before a solution is accepted, `primal_violations` checks rows, bounds, linking equalities and integrality at \(10^{-7}\). On a violation `solve_original` retries with primal simplex, then interior point, on row-scaled clones.

**Which duals are used.** In the Pecci–Jenkins relaxed phase (section 6) all present linking variables are active. Afterwards only the *continuous* linking variables are, and binary or integer ones are handled by state-conditioned cuts.

**Result returned to the master** (one dict per subproblem per iteration):

| Key | Meaning |
| --- | --- |
| `feasible` | True if the LP solved to optimality (or the restoration retry showed feasibility) |
| `op_cost` | \(Q_w(\bar{x})\) in dollars |
| `benders_lambda`, `continuous_lambda` | \(\lambda\) for the active linking variables (identical as written) |
| `continuous_point` | \(\bar{x}\) for those variables |
| `theta_coeff` | 1.0 for an optimality cut, 0.0 for a feasibility cut |
| `master_discrete_state` | Master binary and integer values, in the discrete phases only |
| `operational_discrete_state` | Operational integer values \(z^*\), with `fixed_z` only |
| `is_mip`, `operational_mip_variables` | Whether non-linking discrete variables existed, and their names |

**Infeasible subproblems and restoration.** If the LP is not optimal, `restoration.restore` solves a Phase-I problem on a clone. `add_slacks_to_subproblem` (`subproblems.py:151-292`) relaxes every active row with a finite bound, except the linking equalities, by a multiple of one shared variable \(s_{\max}\geq0\), and moves variable bounds into rows so they can be relaxed too. Each row's multiple is its own scale `_constraint_scale`, so a unit of \(s_{\max}\) is comparable across rows:

$$
\min 10^{6} s_{\max}
\quad \text{s.t.}\quad g_r(z) \le b_r + \sigma_r s_{\max} \forall r,\qquad x_L = \bar x, s_{\max} \ge 0
$$

where \(\sigma_r\) is the row scale. The solver runs primal simplex, then interior point if that fails, and `_certificate` recomputes the dual objective and checks primal residuals, dual signs, stationarity and complementarity at \(10^{-8}\). If both algorithms fail the certificate, a `RuntimeError` is raised. The result is `feasible = False` with `theta_coeff = 0`; section 5 gives the cut. If the minimum slack is at most `ZERO_SLACK_TOLERANCE = 1e-9`, the original LP is re-solved and, if that succeeds, treated as feasible after all.

Restoration raises `NotImplementedError` if a non-linking integer variable is still unfixed, because LP duals would then give an invalid feasibility cut.

## 5. Cuts

All cuts are lower bounds on \(\vartheta_w\), written in units of S. They share one routine, `add_relaxed_benders_cut` (`algorithms.py:1788-1860`), which appends the cut to the global `BendersOptimalityCuts` list; the subproblem key enters only through `vTHETA[w]`.

**Optimality cut.** \(Q_w\) is convex in \(x_L\), so its supporting hyperplane at \(\bar{x}\) under-estimates it everywhere:

$$
Q_w(x_L)\geq Q_w(\bar{x}_L)+\lambda_w^{\top}(x_L-\bar{x}_L)
\Longrightarrow
\vartheta_w\geq\frac{Q_w(\bar{x}_L)}{S}+\sum_{i\in I_w}\frac{\lambda_{w,i}}{S}(x_i-\bar{x}_i)
$$

The code adds \(\kappa\vartheta_w-\mathrm{op\_cost}/S-\sum_i(\lambda_i/S)(x_i-\bar{x}_i)\geq0\), with \(\kappa\) equal to `theta_coeff`. The intercept is the solved `op_cost`, not a recomputed constant. Cuts with \(\kappa=1\) are added for every feasible result, in the relaxed phase and in the all-continuous case.

**Feasibility cut.** Let \(\bar{s}\) be the optimal Phase-I slack at \(\bar{x}\) and \(\mu\) the duals of the linking rows, each divided by its row scale and by `RESTORATION_OBJECTIVE_SCALE = 1e6`. The minimum slack \(s(x)\) is convex and zero exactly when the subproblem is feasible, so

$$
s(x) \ge \bar s + \mu^{\top}(x_L - \bar x)
\Longrightarrow
\bar s + \mu^{\top}(x_L - \bar x) \le 0 \quad \text{is necessary for feasibility}
$$

The worker multiplies the intercept and duals by \(c=\max(1,1/\bar{s})\) so a small violation is not ignored, and returns `theta_coeff = 0`. With \(\kappa=0\), the routine uses a scale of 1 and writes \(c\bar{s}+\sum_i\mu_i(x_i-\bar{x}_i)\leq0\), with no \(\vartheta\) term. It goes into the same `BendersOptimalityCuts` list. In the relaxed phase, `benders()` also checks that a new restoration cut does not exclude an earlier LP-feasible sample and raises `RuntimeError` if it does.

**State-conditioned cuts for discrete linking variables.** Once the master's binary or integer variables are restored to their true domains, a dual cut computed at one integer pattern d is valid only at that pattern. `register_benders_state` stores per subproblem and iteration: `op_cost`, `theta_coeff`, `continuous_lambda`, `continuous_point`, `master_discrete_state` and the recourse lower bound. `add_state_recourse_cut` (`algorithms.py:2017-2404`) then adds the cut only when the master's discrete variables equal d, using a binary \(z\) that is zero exactly when the discrete state matches:

$$
\vartheta_w\geq\frac{Q_w(\bar{x})+\lambda^{\top}(x_{\mathrm{cont}}-\bar{x}_{\mathrm{cont}})-Mz}{S},
\qquad
M = \max\bigl(0, Q_w(\bar{x})+\max_{x\in[lb,ub]}\lambda^{\top}(x-\bar{x})-L_w\bigr)
$$

When \(z=1\) the right-hand side is at most \(L_w\)/S, so the cut is slack there. Variable bounds for \(M\) come from `get_continuous_variable_bounds` (section 8). Mismatch indicators are built per variable:

- **Binary** \(x_j\) with stored value \(d_j\): \(m_j\) = 1 − \(x_j\) if \(d_j\) = 1, else \(m_j\) = \(x_j\). Then z ≥ \(m_j\) for every j and z ≤ Σ \(m_j\), so z = 0 only if every variable matches.
- **Integer** \(x_j\) with stored value t: two binaries `below` and `above` with big-M constraints, and a mismatch variable bounded by their sum.

Each cut lives in a Pyomo `Block` named `BendersDiscretePattern_{key}_{id}`.

**No-good cut for infeasible integer patterns.** `add_combinatorial_feasibility_cut` (`algorithms.py:1416-1707`) excludes the visited discrete pattern entirely:

$$
\sum_j \text{mismatch}_j \ge 1
$$

It is used after the master has been made integer when a subproblem is infeasible and native disjunctions are not in use. For integer-valued linking variables it builds two-sided big-M constraints that force `below` and `above` to 1 on the corresponding side, so it is exact. It assumes the discrete pattern alone makes the subproblem infeasible, so it can be stronger than needed (inferred; the code's own comment for the native path says a different continuous allocation might make the same pattern feasible).

**Native alternatives.** With native MOSEK, the state-conditioned cut is not built as big-M rows. States are passed in `planning_problem._benders_native_djc_data` and `mosek_benders.add_benders_djc` adds one disjunction per cut:

$$
\bigl[ \(m(d)=0\) \wedge \kappa\vartheta - \tfrac{Q}{S} - \tfrac{1}{S}\lambda^{\top}(x - \bar x) \ge 0 \bigr] \vee \bigl[ m(d) \ge 1 \bigr]
$$

with \(m(d)\) the Hamming distance to the discrete state \(d\). This needs no big-M. Feasibility states (\(\kappa\) = 0) use the same form.

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

**DJC seeding.** Cuts are valid only in the state where they are generated, so an unvisited state is bounded only by the loose \(L_w\). With `DjcSeedStates` true and \(2^n\) ≤ `DjcSeedMaxStates` (default 1024) binary linking variables, `benders()` builds one seed per joint binary state, with continuous linking variables set to zero, and evaluates each before the first master solve. Seeding is off by default and has no CLI flag; set it through `GRIDPATH_BENDERS_SETUP`.

### Operational discrete methods

`--benders_operational_discrete_method` selects how non-linking integers inside a subproblem are handled (CLI default `fixed_z`):

- **`pj_relax`**: `prepare_lp_for_duals` relaxes non-linking binaries to $[0, 1]$ and integers to the reals, and the dual comes from the LP relaxation. The relaxed value is at most the MIP value (in NPV terms), so the cut is valid as an LP-relaxation cut but looser.
- **`fixed_z`**: `solve_operational_mip_and_fix_z` solves the actual MIP with the linking equalities in place, checks that each integer is within \(10^{-8}\) of integral, fixes them at \(z^*\), and re-solves the LP to get duals. The cut is then conditional on \(z^*\), which is why `operational_discrete_state` is returned. `restore_lp_after_duals` and `restore_operational_discrete_variables` undo the changes because the instance is persistent. If the subproblem has no operational integers, it is solved directly.

During the master's relaxed phase the worker overrides the method to `pj_relax` (`subproblems.py:851-855`), because only the LP relaxation gives a globally valid cut there.

$
Q_w(x_L\mid z^*)\geq Q_w(\bar{x}_L\mid z^*)+\lambda^{\top}(x_L-\bar{x}_L)
\qquad\text{(valid only for fixed }z^*\text{)}
$

Inferred from the formulas, not stated in the code: at other x the MIP may choose a different z with lower cost, so a fixed-z cut can overestimate the true MIP cost. The resulting UB should then be treated as heuristic for subproblems that contain operational integers, while `pj_relax` keeps UB valid at the price of a looser bound.

## 7. Stabilisation (level sets)

The raw master optimum often jumps between extreme investment plans. Stabilisation chooses the *next trial point* by solving a second problem restricted to plans whose master objective is close enough to UB, and picking the one nearest to the last evaluated point. It does not change the master's cuts or its UB; the master is still solved first.

**Level set.** With a maximisation master, the level is

$$
\ell = UB - \gamma(UB - LB), \qquad \gamma \in (0,1),
\qquad
\text{stabilised point: } \arg\min_{x,\vartheta} \Phi(x) \text{s.t.} \mathrm{MasterObjective}(x,\vartheta) \ge \ell
$$

A small \(\gamma\) keeps the level near \(UB\) (tight); a large \(\gamma\) moves it toward \(LB\) (loose). The constraint is `cLevelSet`, and `LevelObjective` replaces `MasterObjective` for the solve and is removed afterwards (`regularization.py:268-510`). The default `--benders_stab_param` is 0.025; setting \(\gamma=0\) with L1 or L2 raises an error.

| `--benders_stabilization_method` | Φ(x) | Notes |
| --- | --- | --- |
| `none` | MasterObjective itself, minimised over the level set | Drives the point to the level boundary. If MOSEK fails, retries once with dual simplex |
| `l1` | \(\sum_i w_i(p_i+n_i)\), with \(x_i-\bar{x}_i=p_i-n_i\) and \(p_i,n_i\geq0\) | Linear. Auxiliary variables and rows are built once and only right-hand sides change |
| `l2` | \(\sum_i w_i(x_i-\bar{x}_i)^2\) | Quadratic; with MOSEK (or `djc`) a rotated quadratic cone on an epigraph variable t is used |

**Centre and weights.** The centre \(\bar{x}\) is the point the workers *just evaluated*, not the new master optimum (centring on the optimum would make the penalty zero there). Weights are all 1 in practice (`stabilization_weights=None`).

**What is penalised.** Only the *continuous* linking variables. Binary and integer variables are free in the level-set problem, so after integrality is restored the level-set problem is a MIP. If there are no continuous linking variables, Φ is zero and the solve only finds a feasible point.

**Fallbacks.** A failed level-set solve prints a warning and returns the plain master optimum, the unstabilised point. The stabilisation is also skipped on alternate iterations when the run is near the noise floor: `len(gap_hist) > 6`, `gap < \(10^{-3}\)`, `gap_hist[-1] > 0.5·gap_hist[-6]` and an even history length (`algorithms.py:1120-1123`).

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

Bounds matter in three places: the lower bound \(L_w\) on each subproblem value (which keeps the first master solves bounded), the big-M in state-conditioned cuts (section 6), and variable bounds used by the level-set problem and the cut generators.

**Scaling constants.**

| Constant | Value | Where it acts |
| --- | --- | --- |
| `BENDERS_THETA_SCALE` (\(S\)) | \(10^9\) | Master carries \(\vartheta_w=\text{value}/S\), so \(\vartheta_w\) is typically order one |
| `BENDERS_RECOURSE_OBJECTIVE_SCALE` | \(10^6\) | Operational objective is divided by this; duals are multiplied back by it (\(\lambda=-10^6\times\text{dual}\)) |
| Phase-I elastic weight | \(10^6\) | Objective min \(10^6\) · \(s_{\max}\) in the restoration LP |
| Certificate tolerance | \(10^{-8}\) | Dual-ray entries below it are treated as zero in feasibility cuts |
| Variable-bound cap | `1e8` | Last-resort bound for an otherwise unbounded continuous linking variable; when it is used it is also enforced in the master and applied in each worker's recourse LP |

**Subproblem lower bound \(L_w\).** At worker startup the LP relaxation of the operational model is solved for an upper bound on the operational NPV; its negation, `-recourse_upper_bound`, is \(L_w\). The master then imposes \(\vartheta_w\geq L_w/S\), which is valid because every real operational NPV is at most the relaxation's. The linking variables in that LP are bounded using the master-derived bounds described next.

**Linking-variable bounds in the recourse LP.** The worker's recourse LP is a clone of the operational model taken before the \(x=\bar{x}\) linking constraints exist. The master's constraints that are shared by name are deactivated in it (section 4), and it has no investment cost. A linking variable with no explicit upper bound can therefore grow without limit, and because `OP_NPV` is maximised and includes revenue terms (for example sold carbon credits) the LP can be unbounded. In a failing run every linking variable reported `lb = 0, ub = None`: `GenNewLin_Build_MW`, `StorNewLin_Build_MW` and `_MWh`, `TxNewLin_Build_MW`, `BENDERS_CARBON_CREDIT_GENERATED_TOTAL` and `BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION`. MOSEK reported the LP as unbounded and the worker raised `RuntimeError`. Which family drives the ray was not isolated; the carbon-credit variables are the most likely cause because they map directly to revenue.

The fix takes the bounds from the master and applies them to the recourse LP only:

1. `benders()` calls `compute_linking_variable_bounds` after the solver is created and before any worker is started. For every continuous linking variable present in the master it calls `get_continuous_variable_bounds` (sources 1 to 4 above) and collects `{name: (lb, ub)}`. Binary and integer linking variables are skipped, as are variables for which no lower bound can be established (those raise `ValueError` inside the helper, which is logged as a warning and skipped).
2. Any variable that ends up on the 1e8 cap (recorded in `planning_problem._benders_assumed_bounds`) also gets that upper bound set on the master variable. Without this the bound used in the workers would not be valid at every master point.
3. The dictionary is passed to each worker as the last element of the `Process` arguments (`linking_var_bounds`).
4. `calculate_recourse_upper_bound(instance, optimiser, linking_var_names, linking_var_bounds)` clones the model, relaxes operational integrality, and calls `apply_linking_variable_bounds`. That function sets a lower or upper bound on the clone only where the variable has none, so an existing explicit bound is never loosened. It returns the names that still lack a bound.
5. If any variable still lacks a bound, a warning lists the count and the first ten names. If the LP is then not optimal, the `RuntimeError` message also gives the count and the first ten names.

The original operational instance is not changed, so the \(x=\bar{x}\) solves in the worker loop are unaffected.

Setup key `RecourseBoundOBBT` (default `True`) controls whether step 3 of the bound sources (OBBT on the master) is tried before falling back to the 1e8 cap. OBBT costs one master solve per linking variable that has neither an explicit bound nor a capacity-constraint bound. Set it to `False` to skip those solves; those variables then take the 1e8 cap directly.

Consequences to keep in mind. A variable on the \(10^8\) cap gives a valid but potentially very loose bound, so \(U_w\) can be large, \(L_w\) very negative, and \(M_w\) in the big-M formula below can grow accordingly. The log line `Using assumed Benders upper bound ... for <name>` marks each such variable, and the INFO line `Linking-variable bounds: N variables bounded (M with an assumed cap ...)` summarises them. Giving those variables real upper bounds in the master tightens everything downstream.

**Variable bounds** (`get_continuous_variable_bounds`). For a continuous linking variable the bound is taken from the first source that gives one:

1. an explicit bound on the Pyomo variable;
2. a capacity-based bound from `CAPACITY_BOUND_CONSTRAINTS` (`find_gridpath_capacity_upper_bound`), derived from the model's own capacity constraints;
3. optimisation-based bound tightening (`obbt_upper_bound`), i.e. maximising the variable over the master feasible set;
4. the 1e8 cap.

The exact capacity-bound formula is not reproduced here; read `find_gridpath_capacity_upper_bound` in the code for it.

**Dual-term bound.** `continuous_dual_term_upper_bound` bounds \(\lambda^{\top}(x-\bar{x})\) over the box of variable bounds, i.e. \(\sum_i\max\{\lambda_i(l_i-\bar{x}_i),\lambda_i(u_i-\bar{x}_i)\}\). This is the term that enters the big-M below.

**Big-M for state-conditioned cuts.**

$$
\(M_w\) = \max\Bigl(0, \mathrm{op\_cost}_w + \max_{x}\lambda^{\top}(x-\bar x) - \(L_w\)\Bigr)
$$

where \(\mathrm{op\_cost}_w=-\mathrm{OP\_NPV}_w(\bar{x})\). The value \(M_w\) is the largest amount by which the cut's right-hand side can exceed the global lower bound, so deactivating the cut (\(z=1\)) leaves \(\vartheta_w\geq L_w/S\) effective. A loose \(L_w\) or loose variable bounds inflate \(M_w\) and weaken the LP relaxation of the master.

**Numerical notes (inferred).** Dividing by S and \(10^6\) keeps coefficients near \(O(1)\) for MOSEK/Gurobi, but a cut whose λ entries are all below tolerance can look numerically empty; the feasibility cut rescaling c = max(1, 1/\(\bar{s}\)) guards the analogous case there.

## 9. Algorithm, convergence and termination

**Main loop** (`algorithms.py`, the Benders driver). Notation: \(UB\) is the best master objective (an upper bound, since the master is a relaxation of a maximisation), \(LB\) is the best evaluated true objective (the incumbent).

$$
\begin{aligned}
&\textbf{Init: } UB=+\infty, LB=-\infty, \gamma=\gamma_0, \text{derive linking-variable bounds from the master, start one worker per scenario key, collect } \(L_w\) \\
&\textbf{Solve master once} \to \bar x^{0} \\
&\textbf{for } k = 0,1,\dots,\mathrm{MaxIter}: \\
&\quad 1.\ \text{send } \bar x^{k} \text{ to all workers (sliding window)} \\
&\quad 2.\ \text{collect } (\(Q_w\), \lambda_w) \text{ or feasibility certificates} \\
&\quad 3.\ \text{true objective } f(\bar x^k) = NPV_{plan}(\bar x^k) - \textstyle\sum_w \(Q_w\) \\
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
| Converged | \(gap\le\mathrm{ConvTol}\) and \(LB\le UB+\mathrm{ConvTol}\cdot |UB|\) | `ConvTol = 1e-10` |
| Stalled | gap improves by less than StallGap over StallIter iterations | 20 iterations, 1e-6 |
| Time limit | wall-clock time exceeds MaxCpuTime | 3600 s |
| Max iterations | \(k\) reaches MaxIter | 200 |

`AbsGapTol` (default 1e-4) is used only for display. `MaxCpuTime` is wall-clock time despite its name. The exact stall test is paraphrased from the research notes; check the code before relying on the wording.

**After the loop.** Unless `--benders_skip_final_export` is set, the driver sends EXPORT\_FINAL so each worker writes its own results at the incumbent. The incumbent investment decisions are written back into the planning instance, and the result status is mapped to OPTIMAL, TIMELIMIT, MAXITER or NONE. A STOP message then shuts the workers down.

**Result object.** `BendersResult` holds the planning results, the status, histories of UB, LB and gap, and `final_export_status` (added in the patch described in section 11).

## 10. Parallel architecture

**Processes.** The master lives in the driver process. Each operational subproblem, keyed by ScenarioKey(weather, hydro, availability, subproblem, stage), runs in its own persistent worker process created with the `spawn` start method (`subproblems.py`, `benders_subproblem_worker`). A worker builds its Pyomo model once, keeps it in memory, and on each iteration only updates the fixed linking values and re-solves.

**Messages.** Each worker has an input and an output queue. The driver sends the planning solution and receives the objective, duals or a feasibility certificate. The main message types are:

| Message | Direction | Effect |
| --- | --- | --- |
| READY | worker → driver | Model built; carries \(L_w\) (the LP-relaxation bound) |
| solve request | driver → worker | Contains the light planning solution (`_light_planning_sol`, only linking-variable values) |
| result | worker → driver | \(Q_w\), λ, or feasibility data |
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
| `--benders_conv_tol` | ConvTol | \(10^{-10}\) | Relative gap tolerance |
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


## 14. References

The references below cover the mathematical foundations of Benders decomposition and the specific algorithmic ideas used in this implementation. The notes indicate the connection to the code; they do not imply that every implementation detail is copied verbatim from a cited paper.

### Benders decomposition and energy-system planning

1. **Benders, J. F. (1962).** “Partitioning procedures for solving mixed-variables programming problems.” *Numerische Mathematik*, 4, 238–252. [https://doi.org/10.1007/BF01386316](https://doi.org/10.1007/BF01386316).  
   The original Benders decomposition framework: fixing complicating (master) variables, solving the resulting subproblem, and using dual information to generate optimality and feasibility cuts.

2. **Geoffrion, A. M. (1972).** “Generalized Benders decomposition.” *Journal of Optimization Theory and Applications*, 10, 237–260. [https://doi.org/10.1007/BF00934810](https://doi.org/10.1007/BF00934810).  
   Extends the classical framework beyond linear subproblems and provides a broader duality-based foundation for decomposition.

3. **Jacobson, A., Pecci, F., Sepulveda, N., Xu, Q., & Jenkins, J. D. (2024; first published online 2023).** “A computationally efficient Benders decomposition for energy systems planning problems with detailed operations and time-coupling constraints.” *INFORMS Journal on Optimization*, 6(1), 32–45. [https://doi.org/10.1287/ijoo.2023.0005](https://doi.org/10.1287/ijoo.2023.0005).  
   The key reference for decomposing energy-system planning from detailed operational subproblems, parallel subproblem solution, and master-side policy-budgeting variables. The article was published online in August 2023 and appears in the 2024 journal issue.

4. **Pecci, F., & Jenkins, J. D. (2025).** “Regularized Benders decomposition for high performance capacity expansion models.” *IEEE Transactions on Power Systems*, 40(4), 3105–3116. [https://doi.org/10.1109/TPWRS.2025.3526413](https://doi.org/10.1109/TPWRS.2025.3526413).  
   The principal reference for the Pecci–Jenkins level-set approach in large-scale capacity-expansion models, including time-coupling constraints, storage-state linking, parallel operational subproblems, and regularization in mixed-integer planning. The current implementation adapts this family of methods; its exact code paths and options are documented in sections 6–7.

### Feasibility cuts, discrete decisions, and big-M formulations

5. **Codato, G., & Fischetti, M. (2006).** “Combinatorial Benders' cuts for mixed-integer linear programming.” *Operations Research*, 54(4), 756–766. [https://doi.org/10.1287/opre.1060.0286](https://doi.org/10.1287/opre.1060.0286).  
   A central reference for combinatorial feasibility cuts. Their method derives inequalities from irreducible infeasible subsystems and is specifically motivated by avoiding dependence on arbitrary big-M coefficients. The no-good feasibility cuts in this code exclude an infeasible discrete linking assignment; they are related combinatorial cuts, but are not claimed to reproduce the full Codato–Fischetti IIS procedure.

6. **Hooker, J. N., & Ottosson, G. (2003).** “Logic-based Benders decomposition.” *Mathematical Programming*, 96, 33–60. [https://doi.org/10.1007/s10107-003-0375-9](https://doi.org/10.1007/s10107-003-0375-9).  
   Develops Benders cuts based on logical inference rather than only LP duality. Relevant background for feasibility logic and cuts that exclude discrete assignments.

7. **Hooker, J. N. (2007).** “Planning and scheduling by logic-based Benders decomposition.” *Operations Research*, 55(3), 588–602. [https://doi.org/10.1287/opre.1060.0371](https://doi.org/10.1287/opre.1060.0371).  
   Further develops logic-based Benders methods for mixed-integer planning and scheduling.

8. **Magnanti, T. L., & Wong, R. T. (1981).** “Accelerating Benders decomposition: Algorithmic enhancement and model selection criteria.” *Operations Research*, 29(3), 464–484. [https://doi.org/10.1287/opre.29.3.464](https://doi.org/10.1287/opre.29.3.464).  
   Foundational work on strengthening Benders cuts by selecting among alternative dual optima (Pareto-optimal cuts) and on the importance of master formulation quality.

9. **Balas, E. (1979).** “Disjunctive programming.” In *Annals of Discrete Mathematics*, Vol. 5, pp. 3–51. [https://doi.org/10.1016/S0167-5060(08)70342-X](https://doi.org/10.1016/S0167-5060(08)70342-X).  
   Foundational treatment of disjunctive programming and convexification of unions of polyhedra, which provides the mathematical background for expressing logical alternatives as disjunctions.

10. **MOSEK ApS.** *MOSEK Optimizer API Documentation*, section 6.9, “Disjunctive constraints,” and section 13.4.5, “Disjunctive constraints.” [DJC formulation documentation](https://docs.mosek.com/latest/pythonapi/tutorial-djc-optimizer.html) · [MIP and big-M discussion](https://docs.mosek.com/latest/capi/mip-optimizer.html).  
    The solver-level reference for the native DJC interface used by the MOSEK integration, and for how DJCs may be transformed internally into mixed-integer formulations, including big-M formulations. These are software documentation references rather than journal articles.

**Big-M interpretation in this implementation.** The state-conditioned optimality cuts use a mismatch indicator and a calculated \(M\) based on recourse lower bounds and finite bounds on the continuous linking variables (section 6). This is distinct from the combinatorial no-good feasibility cut, which excludes a discrete assignment without using a recourse-cost big-M. The Codato–Fischetti and Hooker references provide the relevant background for these discrete-cut ideas; they should not be read as asserting that every cut here is identical to those papers' constructions.

### Stabilization and level-set methods

11. **Ruszczyński, A. (1986).** “A regularized decomposition method for minimizing a sum of polyhedral functions.” *Mathematical Programming*, 35(3), 309–333. [https://doi.org/10.1007/BF01580883](https://doi.org/10.1007/BF01580883).  
    A foundational regularized decomposition method for convex piecewise-linear value functions, using a quadratic regularizing term in the master problem.

12. **Lemaréchal, C., Nemirovskii, A., & Nesterov, Y. (1995).** “New variants of bundle methods.” *Mathematical Programming*, 69(1–3), 111–147. [https://doi.org/10.1007/BF01585555](https://doi.org/10.1007/BF01585555).  
    General background on bundle and level methods for nonsmooth convex optimization, relevant to stabilization of cutting-plane algorithms.

13. **Linderoth, J., & Wright, S. (2003).** “Decomposition algorithms for stochastic programming on a computational grid.” *Computational Optimization and Applications*, 24, 207–250. [Author's PDF](https://jlinderoth.github.io/papers/Linderoth-Wright-03.pdf).  
    Discusses parallel master–worker decomposition and trust-region stabilization for stochastic programming.

14. **Zverovich, V., Fábián, C. I., Ellison, E. F. D., & Mitra, G. (2012).** “A computational study of a solver system for processing two-stage stochastic LPs with enhanced Benders decomposition.” *Mathematical Programming Computation*, 4, 211–238. [https://doi.org/10.1007/s12532-012-0038-z](https://doi.org/10.1007/s12532-012-0038-z).  
    Computational comparison of regularized decomposition, level methods, and trust-region stabilization within Benders decomposition.

15. **Göke, L., Schmidt, F., & Kendziorski, M. (2024).** “Stabilized Benders decomposition for energy planning under climate uncertainty.” *European Journal of Operational Research*, 316(1), 183–199. [https://doi.org/10.1016/j.ejor.2024.01.016](https://doi.org/10.1016/j.ejor.2024.01.016).  
    An energy-planning application of stabilized Benders decomposition under climate uncertainty.

16. **Rahmaniani, R., Crainic, T. G., Gendreau, M., & Rei, W. (2017).** “The Benders decomposition algorithm: A literature review.” *European Journal of Operational Research*, 259(3), 801–817. [https://doi.org/10.1016/j.ejor.2016.12.005](https://doi.org/10.1016/j.ejor.2016.12.005).  
    A broad review of Benders decomposition theory and computational enhancements, including cut generation, stabilization, and applications. This is a useful general reference for methods not tied to a single implementation detail.

**How these references relate to the implemented L1/L2 options.** In `regularization.py`, the level-set constraint restricts the master objective to remain within a specified fraction of the current optimality gap. The optional L1 and L2 secondary objectives then select a nearby trial point using, respectively, a weighted absolute-deviation objective and a weighted squared-deviation objective over the selected continuous linking variables. The L2 objective is also represented by an epigraph and rotated quadratic cone in the MOSEK path. Pecci and Jenkins (2025) is the closest application-specific reference for level-set regularization in capacity-expansion Benders; Ruszczyński (1986), Lemaréchal et al. (1995), Linderoth and Wright (2003), and Zverovich et al. (2012) provide broader background on regularized decomposition, bundle/level methods, and trust-region approaches. The particular L1/L2 objective variants in this repository are implementation choices and should not be attributed as exact formulations from any one of these papers.
