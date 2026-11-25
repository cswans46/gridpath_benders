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
| `benders/mosek_task.py` | Prototype helpers for compiling instantiated Pyomo models into native MOSEK `.task` files and storing index metadata |
| `benders/restoration.py` | Phase-I feasibility restoration with a dual certificate |
| `benders/regularization.py` | Level-set (stabilisation) problem, L1 and L2 |
| `benders/operational_model_creation.py` | Operational model and linking-variable identification |
| `benders/soc_allocation.py` | Shared SOC allocation targets and variables used by master and operational subproblems |
| `system/policy/benders_policy_constraint.py` | Master-side policy allocation constraints, including the period-wide storage SOC balance and capacity limits |
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
| \(Q_w(\bar{x})\) | Recourse cost, `op_cost = -OP_NPV` in dollars |
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

For yearly-balanced storage, `soc_allocation.py` defines the shared SOC allocation targets and variables used by both the master and operational subproblems. It applies to projects with `operational_type == "stor"` and `balancing_type_project == "year"`, and creates nonnegative start- and end-SOC allocation variables for each project, period and subproblem.

The master-side constraints are implemented in `gridpath/system/policy/benders_policy_constraint.py`. For each project, balancing type and period, `Stor_SOC_Period_Balance_Constraint` requires the day-weighted net SOC change of the representative days to be zero, `sum_w n_w (END_w - START_w) = 0`, where `n_w` is the number of days subproblem `w` represents (its average timepoint weight). Each day's start and end SOC are otherwise free, so a day may be a net charging or discharging day. `Stor_SOC_Capacity_Constraint` and `Stor_SOC_End_Capacity_Constraint` limit the allocated start and end SOC, converted to MWh, to the project's energy-storage capacity. (This replaces the earlier cyclic chain that tied the end SOC of one subproblem to the start SOC of the next in index order.) These two constraints are added to the master only along with non-negativity constraints; the operational models receive the corresponding allocation variables through `add_stor_soc_allocation_components` in `soc_allocation.py` to provide the starting and ending SOC of each subproblem.

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

**Variable, Parameter and Set list**

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
However, this is expensive and the algorithm defaults to Pecci-Jenkins initialisation.

- **Pecci–Jenkins initialization.** When `UsePjInitialization` is enabled and there are no discrete linking variables, the initialization phase temporarily solves a planning-only master with objective `ePlanningNPV`, omitting the recourse-cost term. The algorithm sets \(UB=+\infty\) during this phase. The planning-only objective is not a valid upper bound on the full planning-plus-recourse objective, so it must not be used as one. The value remains infinite while feasibility-restoration iterations seek a trial plan for which the operational subproblems are feasible and can provide optimality-cut information. Once that phase is complete, the full Benders objective is restored; a subsequent solve of that full master provides a valid objective value for initializing or updating \(UB\).

- **Subsequent full master solves.** Whenever the full Benders master is solved, its objective value is \(UB_{\mathrm{new}}=\mathrm{value}(\mathrm{MasterObjective})\). The algorithm updates the running bound as \(UB\leftarrow\min(UB,UB_{\mathrm{new}})\). Taking the minimum retains the tightest upper bound found for this maximization problem. During Pecci–Jenkins feasibility restoration, \(UB\) remains \(+\infty\); the planning-only objective is not used for convergence or as a substitute upper bound.

The \(10^8\) cap and \(UB\) therefore serve different purposes. The cap is a heuristic bound on selected master variables added in response to unboundedness. \(UB\) is an objective bound maintained by the Benders algorithm; outside the Pecci–Jenkins initialization phase, it comes from solving the full master with the recourse-cost term included.

**Solution returned from the master.** `process_planning_sol` packages the planning-variable values in a `PlanningSolution`. It returns initialized values without rounding integer or binary variables. An uninitialized variable may be omitted only if no operational subproblem requires it; a required linking variable that is missing or has no value causes an error. The separate `planning_cost = value(ePlanningNPV)` is planning-only NPV and excludes the \(\vartheta\)-weighted recourse-cost term. By contrast, \(UB\), when evaluated from `MasterObjective` under the full Benders objective, includes that term.

**Cut re-solves and what is (and is not) persistent.** The same Python `optimiser` object is created in `algorithms.py` and passed to `solve_planning_problem` on each master iteration. The Pyomo model instance is also the same master object: new rows are appended to its `ConstraintList` components, and the next call is `optimiser.solve(instance)`. This means the Python-side solver wrapper is reused during the normal loop; it does **not** mean that the solver-side mathematical model is necessarily kept alive and incrementally modified.

The distinction is visible in the repository's solver implementations:

- The custom Gurobi and MOSEK classes are based on Pyomo's `GurobiDirect` and `MosekDirect` interfaces (`gurobi_benders.py`, `mosek_benders.py`). Their `_set_instance` methods build the native solver representation and add the implementation-specific indicator/conic/disjunctive constructs. They are direct interfaces, not Pyomo's `PersistentSolver` interface; the Benders driver does not call a persistent-interface `set_instance`/incremental `add_constraint` API when a cut is appended.
- The GAMS adapter (`gams_benders_solver.py`) implements `solve` by delegating to Pyomo's GAMS shell interface and regenerating the solver option file for each call. It does not expose an incremental master-update method.
- For other `SolverFactory` choices, the behaviour belongs to that particular Pyomo plugin. The driver itself does not inspect or control its internal process, basis, presolve state, or model-reuse strategy.

Therefore, the precise repository-level statement is: **the master Pyomo instance and Python solver-wrapper object are reused, but the Benders code does not implement an explicit persistent solver model or incremental cut transfer.** It also does not set warm-start options, attach a warm-start suffix, or explicitly pass a previous solution/basis to the next master solve. Any reuse performed internally by a particular plugin or external solver is outside what this code establishes. The one-off retry in `solve_planning_problem` is a separate case: after a non-optimal result it may construct a second wrapper object and retry the same Pyomo instance; that retry does not change the normal iteration behaviour.
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

- **Binary** \(x_j\) with stored value \(d_j\): \(m_j = 1 - x_j\) if \(d_j = 1\), else \(m_j = x_j\). Then \(z \geq m_j\) for every \(j\) and \(z \leq \sum_j m_j\), so \(z = 0\) only if every variable matches.
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
| `pj_relax` | **Two phases.** (1) Relax integers to the reals and binaries to $[0, 1]$ (`algorithms.py:342-351`), run ordinary Benders until the gap closes. (2) Restore the domains, reset \(LB = -\infty\), clear the incumbent, re-solve the master as a MIP, and continue with state-conditioned cuts (section 5). If the relaxed optimum is already integral the run stops as optimal. |
| `linearized` | Rewritten to `pj_relax` at `algorithms.py:280`, with `use_native_djc = False`, so state cuts use big-M rows. The separate `linearized_phase` code path is therefore never reached (inferred). |
| `djc` | The master is a MIP from the first iteration and state cuts are native disjunctions. Requires native MOSEK; any other solver raises `ValueError` (`algorithms.py:429-434`). Optional seeding below. |
| `none` | Plain Benders for continuous linking variables only. If discrete linking variables exist the code warns and falls back to `pj_relax`. |

The `pj_relax` label is the code's own; the document uses it without claiming the source reference. In the relaxed phase the dual cuts include the (relaxed) discrete linking variables, so they are valid for fractional x and therefore for every integer x. That is why the first phase can use plain cuts.

`use_native_djc` is true when the requested method is not `linearized` and the solver is native MOSEK. It sends states to `mosek_benders.add_benders_djc` instead of building big-M blocks.

**DJC seeding.** Cuts are valid only in the state where they are generated, so an unvisited state is bounded only by the loose \(L_w\). With `DjcSeedStates` true and \(2^n \leq \texttt{DjcSeedMaxStates}\) (default 1024) binary linking variables, `benders()` builds one seed per joint binary state, with continuous linking variables set to zero, and evaluates each before the first master solve. Seeding is off by default and has no CLI flag; set it through `GRIDPATH_BENDERS_SETUP`.

### Operational discrete methods

`--benders_operational_discrete_method` selects how non-linking integers inside a subproblem are handled (CLI default `fixed_z`):

- **`pj_relax`**: `prepare_lp_for_duals` relaxes non-linking binaries to $[0, 1]$ and integers to the reals, and the dual comes from the LP relaxation. The relaxed value is at most the MIP value (in NPV terms), so the cut is valid as an LP-relaxation cut but looser.
- **`fixed_z`**: `solve_operational_mip_and_fix_z` solves the actual MIP with the linking equalities in place, checks that each integer is within \(10^{-8}\) of integral, fixes them at \(z^*\), and re-solves the LP to get duals. The cut is then conditional on \(z^*\), which is why `operational_discrete_state` is returned. `restore_lp_after_duals` and `restore_operational_discrete_variables` undo the changes because the instance is persistent. If the subproblem has no operational integers, it is solved directly.

During the master's relaxed phase the worker overrides the method to `pj_relax` (`subproblems.py:851-855`), because only the LP relaxation gives a globally valid cut there.

$$
Q_w(x_L\mid z^*)\geq Q_w(\bar{x}_L\mid z^*)+\lambda^{\top}(x_L-\bar{x}_L)
\qquad\text{(valid only for fixed }z^*\text{)}
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

A small \(\gamma\) keeps the level near \(UB\) (tight); a large \(\gamma\) moves it toward \(LB\) (loose). The constraint is `cLevelSet`, and `LevelObjective` replaces `MasterObjective` for the solve and is removed afterwards (`regularization.py:268-510`). The default `--benders_stab_param` is 0.025; setting \(\gamma=0\) with L1 or L2 raises an error.

| `--benders_stabilization_method` | Φ(x) | Notes |
| --- | --- | --- |
| `none` | MasterObjective itself, minimised over the level set | Drives the point to the level boundary. If MOSEK fails, retries once with dual simplex |
| `l1` | \(\sum_i w_i(p_i+n_i)\), with \(x_i-\bar{x}_i=p_i-n_i\) and \(p_i,n_i\geq0\) | Linear. Auxiliary variables and rows are built once and only right-hand sides change |
| `l2` | \(\sum_i w_i(x_i-\bar{x}_i)^2\) | Quadratic; with MOSEK (or `djc`) a rotated quadratic cone on an epigraph variable t is used |

**Centre and weights.** The centre \(\bar{x}\) is the point the workers *just evaluated*, not the new master optimum (centring on the optimum would make the penalty zero there). Weights are all 1 in practice (`stabilization_weights=None`).

**What is penalised.** Only the *continuous* linking variables. Binary and integer variables are free in the level-set problem, so after integrality is restored the level-set problem is a MIP. If there are no continuous linking variables, Φ is zero and the solve only finds a feasible point.

**Fallbacks.** A failed level-set solve prints a warning and returns the plain master optimum, the unstabilised point. The stabilisation is also skipped on alternate iterations when the run is near the noise floor: `len(gap_hist) > 6`, `gap < \(10^{-3}\)`, `gap_hist[-1] > 0.5 * gap_hist[-6]` and an even history length (`algorithms.py:1120-1123`).

**Dynamic γ.** With `--benders_stab_dynamic` true (default) and \(k \geq 1\), `update_stab_param` (`algorithms.py:2968-2983`) compares the actual drop in UB with the drop the previous level predicted:

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

The update applies only when \(a_p \geq 0\) and \(p_p \geq 0\). The meaning of tight and loose is inferred from the level formula. Two sharp edges are listed in section 13: \(r\) is computed before the guard, so \(p_p = 0\) raises `ZeroDivisionError`, and \(\gamma\) is not clipped.

## 8. Bounds, big-M and numerical scaling

Bounds matter in three places: the lower bound \(L_w\) on each subproblem value (which keeps the first master solves bounded), the big-M in state-conditioned cuts (section 6), and variable bounds used by the level-set problem and the cut generators.

**Scaling constants.**

| Constant | Value | Where it acts |
| --- | --- | --- |
| `BENDERS_THETA_SCALE` (\(S\)) | \(10^9\) | Master carries \(\vartheta_w=\text{value}/S\), so \(\vartheta_w\) is typically order one |
| `BENDERS_RECOURSE_OBJECTIVE_SCALE` | \(10^6\) | Operational objective is divided by this; duals are multiplied back by it (\(\lambda=-10^6\times\text{dual}\)) |
| Phase-I elastic weight | \(10^6\) | Objective \(\min 10^6 s_{\max}\) in the restoration LP |
| Certificate tolerance | \(10^{-8}\) | Dual-ray entries below it are treated as zero in feasibility cuts |
| Variable-bound cap | `1e8` | Last-resort bound for an otherwise unbounded continuous linking variable; when it is used it is also enforced in the master and applied in each worker's recourse LP |

**Subproblem lower bound \(L_w\).** At worker startup the LP relaxation of the operational model is solved for an upper bound on the operational NPV; its negation, `-recourse_upper_bound`, is \(L_w\). The master then imposes \(\vartheta_w\geq L_w/S\), which is valid because every real operational NPV is at most the relaxation's. The linking variables in that LP are bounded using the master-derived bounds described next.

**Linking-variable bounds in the recourse LP.** The worker's recourse LP is a clone of the operational model taken before the \(x=\bar{x}\) linking constraints exist. The master's constraints that are shared by name are deactivated in it (section 4), and it has no investment cost. A linking variable with no explicit upper bound can therefore grow without limit, and because `OP_NPV` is maximised and includes revenue terms (for example sold carbon credits) the LP can be unbounded. In a failing run every linking variable reported `lb = 0, ub = None`: `GenNewLin_Build_MW`, `StorNewLin_Build_MW` and `_MWh`, `TxNewLin_Build_MW`, `BENDERS_CARBON_CREDIT_GENERATED_TOTAL` and `BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION`. MOSEK reported the LP as unbounded and the worker raised `RuntimeError`. Which family drives the ray was not isolated; the carbon-credit variables are the most likely cause because they map directly to revenue.

The bounds are derived on the master and then supplied to the worker's *recourse-bound LP*. This is needed because that LP is a clone of the operational model before the linking equalities \(x_L=\bar{x}_L\) are added. It also omits the master's capacity/potential constraints and investment costs. A linking variable that is free in this clone can therefore move independently of the master and, in a revenue-maximising operational objective, may make the LP unbounded.

**Mathematical definition of OBBT.** Suppose continuous linking variable \(x_i\) has no usable explicit upper bound and no upper bound that can be derived from a GridPath capacity/potential constraint. The code temporarily replaces the master objective by \(\max x_i\), relaxes the master binary variables to \([0,1]\) and integer variables to continuous variables, and solves the resulting LP relaxation:

$$
u_i^{\mathrm{OBBT}} =
\max\{x_i : (x,\ldots)\in X_{\mathrm{master}}^{\mathrm{LP}}\}.
$$

If this problem is solved to optimality, its optimal value is a valid upper bound for the original mixed-integer master: the LP relaxation contains every feasible integer master point, so its maximum cannot be smaller than the maximum over the original feasible set. The original master objective and discrete domains are restored in a `finally` block after the OBBT solve. The code performs an upper-bound OBBT solve only; the lower bound is taken from the variable's explicit bound or Pyomo domain.

**Algorithm and execution order.** Unless Pecci–Jenkins initialization is active, `benders()` calls `compute_linking_variable_bounds` after creating the solver and before starting any worker processes. For each name returned by `define_linking_vars`, the routine:

1. Looks up the component in the master. Binary and integer linking variables are skipped because this bound dictionary is for continuous linking variables. If the component is absent from the master, no master-derived bound is produced for it.
2. Checks for an explicit finite Pyomo upper bound. If present, it pairs that with a finite lower bound from the explicit variable bound or domain.
3. Otherwise searches GridPath's `CAPACITY_BOUND_CONSTRAINTS` for a capacity/potential constraint that gives a valid upper bound, again requiring a finite lower bound.
4. Otherwise, if `RecourseBoundOBBT=True` (the default), solves the OBBT problem above. This is one additional master optimization solve for each variable that reaches this stage—not one OBBT solve for every linking variable.
5. If OBBT is disabled or does not return an optimal finite value (for example, the target is unbounded in the relaxed master), assigns the assumed upper cap \(10^8\) and logs a warning that it has not been independently verified. The assumed cap is recorded in `planning_problem._benders_assumed_bounds`.
6. If no finite lower bound can be obtained, the helper raises `ValueError`; `compute_linking_variable_bounds` logs the variable as skipped rather than inventing a lower bound.

For a variable assigned the assumed \(10^8\) upper cap, `compute_linking_variable_bounds` also enforces that upper bound on the corresponding master variable. This is essential: otherwise the worker would use a bound that did not hold for all master-feasible trial points. The resulting dictionary \(\{i:(\ell_i,u_i)\}\) is passed to every worker process as `linking_var_bounds`.

At worker startup, `calculate_recourse_upper_bound` clones the operational instance, relaxes operational integrality, and calls `apply_linking_variable_bounds`. For each linking variable present in that worker, the function copies a master-derived lower or upper bound only when the corresponding bound on the clone is missing; it never overwrites an existing operational bound. The original operational instance is not modified, so the later worker solves with linking equalities \(x_L=\bar{x}_L\) are unaffected.

**Variables that exist only in a subproblem.** In this branch, the canonical `linking_vars` list is created from the planning/master model by `define_linking_vars`. A variable that exists only in an operational subproblem is therefore not included in that list: it is not treated as a master--worker linking variable, and no master-derived bound is calculated or passed for it. It remains an ordinary operational variable and keeps whatever bounds its own Pyomo declaration/domain and operational constraints provide. Importantly, `apply_linking_variable_bounds` only checks names in the worker's `linking_var_names` list, so a truly subproblem-only variable is **not necessarily named in its “still unbounded linking variables” warning**. If it is unbounded and causes the recourse-bound LP to be unbounded or otherwise non-optimal, the worker still fails with the generic recourse-bound `RuntimeError`; the missing variable may need to be diagnosed from the model or solver output. If a name is in the passed linking list and exists in the worker but has no finite bound, then it is explicitly reported by `apply_linking_variable_bounds`.

**Cost and effect of the fallback.** The OBBT solves happen serially in the parent process before worker startup, so a large number of variables reaching step 4 can add noticeable startup time. Set `RecourseBoundOBBT=False` to skip those solves; eligible variables then go directly to the assumed cap. The cap is not a proven physical limit. It can make the worker's operational-NPV upper bound \(U_w\) very loose, which makes the recourse-cost lower bound \(L_w=-U_w\) very negative and can inflate the big-\(M\) coefficient used by state-conditioned cuts. The warning `Using assumed Benders upper bound ...` identifies each capped variable, while the INFO summary reports the total bounded, assumed-cap, and skipped counts. Real model-derived upper bounds are preferable because they tighten the recourse LP and downstream cuts.

**Variable bounds** (`get_continuous_variable_bounds`). For a continuous linking variable the bound is taken from the first source that gives one:

1. an explicit bound on the Pyomo variable;
2. a capacity-based bound from `CAPACITY_BOUND_CONSTRAINTS` (`find_gridpath_capacity_upper_bound`), derived from the model's own capacity constraints;
3. optimisation-based bound tightening (`obbt_upper_bound`), i.e. maximising the variable over the master feasible set;
4. the 1e8 cap.

The exact capacity-bound formula is not reproduced here; read `find_gridpath_capacity_upper_bound` in the code for it.

**Dual-term bound.** `continuous_dual_term_upper_bound` bounds \(\lambda^{\top}(x-\bar{x})\) over the box of variable bounds, i.e. \(\sum_i\max\{\lambda_i(l_i-\bar{x}_i),\lambda_i(u_i-\bar{x}_i)\}\). This is the term that enters the big-M below.

**Big-M for state-conditioned cuts.**

$$
M_w = \max\left\{0, \mathrm{op\_cost}_w + \max_{x\in X_L}\lambda^{\top}(x-\bar{x}) - L_w\right\}
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

**Operational integrality relaxation and domain bounds.** The current `relax_operational_integrality` implementation in `subproblems.py` explicitly preserves the effective bounds before changing a variable's domain. For each unfixed binary or integer variable it first stores `lb, ub = var.bounds`, changes the domain to `Reals`, and then reapplies those stored bounds with `setlb` and `setub`. Consequently, an implicit domain bound such as the zero lower bound of `NonNegativeIntegers` is retained in the LP relaxation; the earlier concern that this routine silently drops domain-derived bounds does not describe the current branch. Fixed variables are left unchanged. This is distinct from OBBT's temporary relaxation of master integer domains, which changes domains and restores the original domains afterwards.

**Numerical edge cases**

- `update_stab_param` can raise `ZeroDivisionError` when pp = 0 and does not clip γ.
- `add_state_recourse_cut` (around line 2156) passes a string name to `get_continuous_variable_bounds`, which expects a Var, for integer linking variables.

**Dead or unused code**

- `add_discrete_pattern_indicator`, `add_binary_pattern_constraints`, `add_integer_pattern_constraints` have no callers; the `linearized` phase is dead.
- The Gurobi native indicator path is never enabled, since `_benders_native_indicator_data` is never set.
- DJC and native paths raise `NotImplementedError` for non-binary integer linking variables.
- The Benders code does not explicitly request or pass warm starts to the master solver. The direct Gurobi/MOSEK wrappers and GAMS adapter should be assessed separately if solver-side state reuse is important; see the solver-state discussion in section 3.

**Documentation mismatches**

- `AbsGapTol` is display-only; `MaxCpuTime` is wall time.
- The CLI help for the master discrete method omits `none` and `linearized`.
- `OperationalDiscreteMethod` defaults to pj\_relax in the `.get` call but fixed\_z on the CLI; the CLI wins.
- The L2 docstring mentions division by objective\_scale² that is not applied.
- The label "P&J" / pj\_relax is the code's own; I did not verify the attribution to a paper.

**Scope**

- Memory holds all N operational models for the whole run (section 10).
- This text describes the code at HEAD of `new_gridpath_w_BD_imp` plus my reporting and efficiency patches where stated.




## 14. Quick start and essential information

This section documents the Benders-specific command-line interface in this branch. The Benders flags are defined in \`get_benders_flag_parser()\` in \`gridpath/common_functions.py\`; their values are assembled into the setup dictionary in \`gridpath/run_scenario.py\` and consumed by \`gridpath/benders/algorithms.py\`.

### 14.1. Minimum requirements and command

Before running:

1. Install this repository in the Python environment you intend to use (an editable install is useful during development).
2. Activate that environment and ensure the selected solver integration and licence are available.
3. Generate the scenario input directory.
4. Run from a working directory where the scenario path resolves correctly.

The minimum command is:

\`\`\`bash
gridpath_run --scenario YOUR_SCENARIO --benders 1
\`\`\`

- \`--scenario\` identifies the scenario directory.
- \`--benders\` is a required integer flag. Use \`1\` to enable Benders; \`0\` selects the ordinary solve path. Any nonzero integer is treated as true.
- By default, GridPath looks for the scenario at \`../scenarios/YOUR_SCENARIO\`, relative to the current working directory.
- To use a different parent directory, add \`--scenario_location /path/to/scenarios\`. GridPath appends the scenario name to this parent path.

For example:

\`\`\`bash
gridpath_run --scenario YOUR_SCENARIO \\
  --scenario_location /path/to/scenarios \\
  --benders 1
\`\`\`

Use \`gridpath_run --help\` to inspect the arguments exposed by the installed checkout. The command-line options documented below are the Benders controls plus the scenario, solver, and parallel-execution arguments needed to run them.

### 14.2. Essential Benders options

The defaults below are those declared by the parser in this branch. Boolean options parsed with \`str2bool\` accept \`true/false\`, \`t/f\`, \`yes/no\`, \`y/n\`, and \`1/0\` (case-insensitive).

| Option | Default / accepted values | Detailed explanation |
| --- | --- | --- |
| \`--benders\` | Required integer; use \`1\` | Selects the Benders decomposition path. This argument is required even when its value is zero. A nonzero value invokes the Benders driver; zero follows the standard scenario-solving path. |
| \`--max_benders_iter\` | \`200\` | Maximum number of Benders iterations. Each iteration solves the master, evaluates the resulting linking-variable point in the operational workers, and adds optimality or feasibility information. Reaching this limit terminates the algorithm without implying convergence. |
| \`--benders_conv_tol\` | \`1e-10\` | Relative convergence tolerance for the master upper bound and best complete-plan lower bound. A smaller tolerance asks for a tighter relative optimality gap and can require more iterations. |
| \`--benders_abs_gap_tol\` | \`1e-4\` | Absolute optimality-gap tolerance in the objective's units (NPV dollars in this implementation). It complements the relative tolerance, particularly when objective values are close to zero or the relative gap is numerically uninformative. |
| \`--max_benders_cpu_time\` | \`3600\` seconds | Time limit passed into the operational subproblem solve configuration. It is not the overall wall-clock limit for the entire Benders run; total run time also includes master solves, worker coordination, and other overhead. |
| \`--benders_master_discrete_method\` | \`pj_relax\`; implementation accepts \`none\`, \`pj_relax\`, \`linearized\`, \`djc\` | Selects how discrete master/linking decisions are handled. \`pj_relax\` begins with a relaxed master phase and later handles discrete states using state-conditioned cuts. \`linearized\` uses the relaxed phase and linearized activation/mismatch constraints for state-conditioned cuts. \`djc\` represents the state-conditioned logic with native MOSEK disjunctive constraints and therefore requires the native MOSEK integration. \`none\` is intended for a master with continuous linking variables; if discrete linking variables are present, the implementation warns and falls back to \`pj_relax\`. |
| \`--benders_operational_discrete_method\` | \`fixed_z\`; \`fixed_z\` or \`pj_relax\` | Controls non-linking integer variables in the operational subproblems. \`fixed_z\` fixes the operational integer state for the LP solve from which dual sensitivities are obtained. Those cuts are conditional on that fixed state. \`pj_relax\` relaxes operational integrality for the recourse LP instead. This setting does not select the treatment of master linking binaries; use \`--benders_master_discrete_method\` for those. |
| \`--benders_use_pj_initialization\` | \`true\` | Enables Pecci--Jenkins-style initialization when the master has no discrete linking variables. It avoids the usual initial global recourse-bound calculation and uses the P\&J initialization sequence. If discrete linking variables are present, the implementation warns and retains the alternative initialization. |
| \`--benders_stabilization_method\` | \`none\`; \`none\`, \`l1\`, \`l2\` | Selects the secondary objective used to choose the next trial point. \`none\` adds no L1/L2 distance objective; with the default nonzero level parameter, the level-set restriction still provides P\&J-style regularization. \`l1\` selects a point by minimizing a weighted sum of absolute deviations from the current centre. \`l2\` minimizes a weighted squared distance and uses the corresponding quadratic/conic representation. The distance penalties apply to the continuous linking variables selected by the implementation. |
| \`--benders_stab_param\` | \`0.025\` | Sets the level-set parameter \(\gamma\). The level is formed from the current bounds as \(\ell=UB-\gamma(UB-LB)\). A smaller \(\gamma\) places the level closer to the upper bound; a larger value moves it toward the lower bound. L1 and L2 stabilization require a nonzero parameter. |
| \`--benders_stab_dynamic\` | \`true\` | Enables iteration-by-iteration adjustment of \(\gamma\) based on the observed progress in the upper bound relative to the previous level prediction. Set it to false to retain the supplied \`--benders_stab_param\` throughout the run. |
| \`--benders_operational_solver\` | \`automatic\`; \`automatic\`, \`interior_point\`, \`primal_simplex\` | Selects the LP algorithm for MOSEK or Gurobi operational relaxations in the \`pj_relax\` path. \`automatic\` leaves the choice to the solver; the other values request the corresponding interior-point or primal-simplex algorithm. It does not choose the solver interface itself; use \`--solver\` for that. |
| \`--benders_skip_final_export\` | Off; flag takes no value | Skips the final worker-side operational solves and per-subproblem result exports at the best incumbent. The planning-level results and Benders convergence files are still produced. Use this only when final operational result files are not required. |

**Parser-help caveat.** The help string for \`--benders_master_discrete_method\` currently advertises only \`pj_relax\` and \`djc\`, but the algorithm implementation also accepts \`none\` and \`linearized\`. The accepted values described in the table reflect the implementation, not just the abbreviated help string.

### 14.3. Solver and parallel-worker arguments

These are not algorithmic Benders parameters, but they are directly relevant to configuring a Benders run.

| Option | Default / accepted values | Detailed explanation |
| --- | --- | --- |
| \`--solver\` | If unspecified, HiGHS is selected unless the scenario's \`solver_options.csv\` selects an interface | Selects the Pyomo solver interface used by the run. The choice must be compatible with the requested master method and installed solver integration. In particular, \`--benders_master_discrete_method djc\` requires native MOSEK; specifying GAMS with MOSEK as its backend is not the same as the native MOSEK interface for DJCs. The solver name in \`solver_options.csv\`, when that file is used, must agree with a solver supplied on the command line. |
| \`--solver_executable\` | Unspecified | Supplies an explicit executable path for the selected solver interface. It does not change which solver is selected. |
| \`--n_parallel_solve\` | \`1\` | Number of operational subproblems solved concurrently. The master remains in the driver process, while operational workers solve in parallel. Raising this value can reduce elapsed time when enough independent subproblems exist, but increases concurrent memory use, CPU demand, and solver-licence demand. Set it with the size of the operational LPs and available resources in mind. |

### 14.4. Recommended command patterns

**Default Benders run**

\`\`\`bash
gridpath_run --scenario YOUR_SCENARIO --benders 1
\`\`\`

**Set iteration and convergence limits**

\`\`\`bash
gridpath_run --scenario YOUR_SCENARIO --benders 1 \\
  --max_benders_iter 300 \\
  --benders_conv_tol 1e-6 \\
  --benders_abs_gap_tol 1e-3
\`\`\`

**Use L1 stabilization**

\`\`\`bash
gridpath_run --scenario YOUR_SCENARIO --benders 1 \\
  --benders_stabilization_method l1 \\
  --benders_stab_param 0.025
\`\`\`

**Use L2 stabilization with a fixed level parameter**

\`\`\`bash
gridpath_run --scenario YOUR_SCENARIO --benders 1 \\
  --benders_stabilization_method l2 \\
  --benders_stab_param 0.025 \\
  --benders_stab_dynamic false
\`\`\`

**Use native MOSEK disjunctive constraints for discrete master linking variables**

\`\`\`bash
gridpath_run --scenario YOUR_SCENARIO --benders 1 \\
  --solver mosek \\
  --benders_master_discrete_method djc
\`\`\`

**Run multiple operational workers concurrently**

\`\`\`bash
gridpath_run --scenario YOUR_SCENARIO --benders 1 \\
  --n_parallel_solve 4
\`\`\`

The examples are templates: confirm the scenario path, solver option file, solver installation, licence availability, and available memory before starting a large run.

### 14.5. Outputs and interpretation

The Benders driver writes the planning-level results and, unless \`--benders_skip_final_export\` is set, requests a final operational solve/export at the best incumbent. It also writes:

- \`results/benders_convergence.csv\`: iteration-level lower-bound, upper-bound, gap, and elapsed-time history.
- \`results/benders_summary.json\`: final termination status, iteration count, final bounds and gap, elapsed time, and available incumbent/export information.

The master upper bound and the best evaluated complete-plan lower bound are the quantities used to assess convergence. A run that stops because it reaches \`--max_benders_iter\` or a time limit is not necessarily converged; inspect the final status and reported gap in \`benders_summary.json\`.

### 14.6. Advanced setup override

For controlled experiments, \`run_scenario.py\` also reads the \`GRIDPATH_BENDERS_SETUP\` environment variable as a JSON object and merges it over the setup dictionary constructed from the command-line options. For example, in a POSIX shell:

\`\`\`bash
export GRIDPATH_BENDERS_SETUP='{"StabilizationMethod": "l1"}'
gridpath_run --scenario YOUR_SCENARIO --benders 1
\`\`\`

The setup keys assembled from the CLI are \`MaxIter\`, \`ConvTol\`, \`AbsGapTol\`, \`MaxCpuTime\`, \`StabDynamic\`, \`StabParam\`, \`MasterDiscreteMethod\`, \`OperationalDiscreteMethod\`, \`StabilizationMethod\`, and \`UsePjInitialization\`. The environment override is an advanced developer/testing mechanism; prefer the documented CLI options for ordinary runs. The operational LP algorithm selector \`--benders_operational_solver\` and the worker count \`--n_parallel_solve\` are passed separately rather than being keys in this setup dictionary.


## 15. Dynamic memory management for persistent Benders workers

### 15.1 Motivation and resource constraint

GridPath starts one persistent process for each operational subproblem. Each process retains its Pyomo model between Benders iterations, while its additional memory requirement changes over time (model construction, solver representation, optimization, feasibility restoration and final export). A fixed per-worker memory quota would leave capacity unused whenever a worker is idle or has a small temporary requirement.

The memory coordinator therefore manages a shared, time-varying reservation. At every reservation event, the admission-control condition is

$$
\\sum_{w\\in W} \\left(R_w(t)+T_w(t)\\right) \\leq M_{\\mathrm{budget}}
\\qquad \\text{for every time }t.
$$

Here, \\(R_w(t)\\) is worker \\(w\\)'s persistent reservation for its resident model and retained process memory, while \\(T_w(t)\\) is its temporary reservation for the operation currently in progress. Reservations are summed across workers at the same time; they are **not** summed across seconds. A worker can increase or decrease its temporary reservation as its phase changes, and unused capacity is available to other workers.

The budget is calculated from physical RAM using the following defaults, and the parent process's current RSS (which includes the planning master) is subtracted before allocating reservations to workers:

$$
M_{\\mathrm{budget}} = \\min\\left(
0.75M_{\\mathrm{total}},
M_{\\mathrm{total}}-\\max(0.10M_{\\mathrm{total}},32\\;\\mathrm{GiB})
\\right).
$$

The defaults can be changed with `GRIDPATH_BENDERS_MEMORY_FRACTION`, `GRIDPATH_BENDERS_RESERVE_FRACTION`, and `GRIDPATH_BENDERS_MINIMUM_RESERVE_GB`.

### 15.2 Coordinator and worker lifecycle

`BendersMemoryCoordinator` in `benders/algorithms.py` is created by the parent process and shared with all spawned workers. Its shared reservation arrays are protected by a condition lock, making each reservation change atomic: the coordinator checks the aggregate reservation and grants the change only when it fits within the budget. Otherwise, the requesting worker waits for another worker to release capacity. A request larger than the entire budget, or one that cannot be admitted within the configured wait timeout, raises a descriptive `MemoryError` rather than silently exceeding the reservation budget.

The worker integration in `benders/subproblems.py` follows this lifecycle:

1. **Startup reservation.** Before building its operational model, each worker reserves initial persistent and temporary memory estimates. The existing startup semaphore continues to limit simultaneous model construction and initial recourse-bound solves.
2. **Persistent model reservation.** After model construction and the initial recourse-bound calculation, the worker measures its full resident-set size (RSS), including its Python/Pyomo process baseline, and reconciles its persistent reservation to that RSS multiplied by a safety factor. The startup temporary reservation is released before the worker reports `READY`. The worker first releases that temporary allowance and then reconciles its persistent reservation, avoiding a circular wait when multiple workers need to increase their resident estimates. After each recourse solve, it repeats this process: release the temporary reservation, update persistent memory to full current RSS, and retain an estimate of the transient peak for its next operation.
3. **Temporary operation reservation.** Before each recourse solve or final export, the worker requests a temporary reservation. It releases that reservation after returning the result. The persistent reservation remains held for as long as the worker retains its model.
4. **Worker shutdown.** The worker releases both reservations in its `finally` block, including when initialization or processing raises an exception.

### 15.3 Configuration

| Environment variable | Default | Purpose |
| --- | ---: | --- |
| `GRIDPATH_BENDERS_MEMORY_FRACTION` | `0.75` | Maximum fraction of physical RAM available to Benders before applying the reserve rule |
| `GRIDPATH_BENDERS_RESERVE_FRACTION` | `0.10` | Proportional system-memory reserve |
| `GRIDPATH_BENDERS_MINIMUM_RESERVE_GB` | `32` | Minimum absolute system-memory reserve in GiB |
| `GRIDPATH_BENDERS_INITIAL_PERSISTENT_GB` | `4` | Initial per-worker persistent reservation during model construction |
| `GRIDPATH_BENDERS_INITIAL_TEMPORARY_GB` | `4` | Initial per-worker temporary startup reservation |
| `GRIDPATH_BENDERS_OPERATION_TEMPORARY_GB` | `4` | Initial temporary reservation used before the first recourse operation |
| `GRIDPATH_BENDERS_MIN_OPERATION_TEMPORARY_GB` | `0.5` | Minimum temporary reservation retained after RSS-based adaptation |
| `GRIDPATH_BENDERS_MEMORY_SAFETY_FACTOR` | `1.25` | Multiplier applied when reconciling measured resident and transient memory |
| `GRIDPATH_BENDERS_MEMORY_WAIT_TIMEOUT` | `600` | Maximum seconds a worker waits for a reservation before raising `MemoryError` |

### 15.4 Interpretation and limitations

This is **cooperative admission control**, not an operating-system memory limit. Pyomo, solver interfaces and native solver libraries can allocate memory without consulting the coordinator. The coordinator therefore enforces the sum of *worker reservations*, after subtracting the parent's measured RSS from the calculated worker budget, not a mathematical guarantee that actual RSS can never exceed the budget. The initial and temporary estimates must be chosen conservatively for the model and solver in use. RSS is a process-level diagnostic and can include shared pages; it is not identical to private committed memory.

For recourse operations, each worker samples its RSS at 100 ms intervals. The observed increase over the operation's starting RSS, multiplied by the safety factor, becomes that worker's temporary reservation estimate for its next operation (subject to a configurable minimum). This lets workers develop different temporary-memory estimates from their own observed workloads. The estimate is updated between operations; it does not retroactively prevent an allocation that exceeds the current reservation, and very short-lived peaks may fall between samples. The first operation uses the configured initial temporary estimate. Final export uses the current estimate once. If the retained models alone cannot fit within the budget, workers will wait and eventually raise a `MemoryError`; reducing `n_parallel_solve` only limits simultaneous startup/dispatch and does not make the total resident model footprint disappear. In that case, reduce model memory, increase available RAM or change the worker lifecycle so models can be unloaded and reconstructed.

In native offload mode (Section 15.5) the persistent reservation of an idle worker is its small post-offload footprint, not a resident model, which is what makes the limitations above less binding.

### 15.5 Native MOSEK task path with disk offload

This section documents the change that moves each worker's operational model out of Pyomo and RAM. It describes what was done, why, how it fits the global memory rule of Section 15.1, how to configure it, and what is still unverified. Words with a specific meaning here are defined in 15.5.1 and are written the same way throughout.

#### 15.5.1 Glossary

| Term | Meaning in this document |
| --- | --- |
| **Subproblem / worker** | One operational subproblem (for example one day-group) and the persistent operating-system process (a *spawned* Python process) that solves it. There is one worker per subproblem. |
| **Pyomo model (instance)** | The algebraic model object built by `create_operational_problem_benders`. It is large because it stores every variable, constraint and expression as Python objects. |
| **MOSEK Task** | MOSEK's own in-memory problem object (a sparse matrix, bounds, objective and parameters). It is much smaller than the Pyomo model and is what MOSEK actually optimizes. |
| **`.task` file** | The MOSEK Task written to disk. Reading it back recreates the Task exactly. |
| **Compile** | Converting the Pyomo model into a MOSEK Task once, through Pyomo's MOSEK interface (so rows and columns match the Pyomo path), and writing the `.task` file. |
| **Dispose / release** | Explicitly destroying the in-memory MOSEK Task so its memory returns to the operating system. |
| **Offload** | Disposing the Task after every trial and reading it from the `.task` file for the next one. After an offload the worker holds almost no model memory. |
| **Resident** | Held in RAM. *Resident mode* (`GRIDPATH_BENDERS_TASK_OFFLOAD=0`) keeps the Task in RAM between trials. |
| **RSS (resident set size)** | The physical memory the operating system currently assigns to a process (the "Working set" column in Windows Task Manager). All memory figures in the logs are RSS. |
| **Idle footprint** | A worker's RSS when it is waiting for a message: the Python interpreter and imported libraries plus whatever model it keeps. |
| **Persistent reservation** \\(R_w\\) | The amount of memory worker \\(w\\) tells the coordinator it keeps for as long as it lives (its idle footprint times the safety factor). |
| **Temporary reservation** \\(T_w\\) | The extra amount worker \\(w\\) tells the coordinator it needs only while a trial, build or export is running. It is released afterwards. |
| **Budget** \\(M_{\\mathrm{budget}}\\) | The total memory Benders may reserve (Section 15.1). |
| **Cooperative** | The coordinator only adds up what workers declare. The operating system does not stop a worker that allocates more than it declared. |
| **Trial** | One evaluation of one subproblem at one master (planning) solution during a Benders iteration. |
| **Linking variable** | A planning variable that also appears in a subproblem. Its value comes from the master. |
| **Linking row** | A permanent equality constraint `linking variable == value` added to the subproblem. Only its right-hand side changes between trials. Its **dual** is that variable's Benders cut coefficient. |
| **Dual (\\(y\\)) and \\(\\lambda\\)** | \\(y\\) is MOSEK's dual value of a row. The cut coefficient is \\(\\lambda = -y\\cdot s\\), where \\(s\\) is the objective scale below. |
| **Objective scale** \\(s\\) | The subproblem objective is divided by `BENDERS_RECOURSE_OBJECTIVE_SCALE` (\\(10^{6}\\)) before solving for numerical conditioning; costs and duals are multiplied back by \\(s\\). |
| **LP / MIP** | Linear program (no integer variables) / mixed-integer program. Duals exist only for LPs. |
| **Relaxed** | An integer or binary variable treated as continuous (its type is changed to continuous). |
| **\\(z\\)** | The operational integer or binary variables (those that are not linking variables). |
| **`pj_relax` / `fixed_z`** | Operational discrete methods. `pj_relax`: relax every \\(z\\) and solve one LP. `fixed_z`: solve the MIP, round and fix \\(z^\\ast\\), then solve an LP to get duals. |
| **Master phases** | `pj_relaxed` phase: master integrality relaxed. `restored`: master integrality restored after the relaxed phase. `linearized` and `djc` use discrete cuts. |
| **Basic (`bas`), interior (`itr`), integer (`itg`) solution** | MOSEK's three solution types. For an LP the basic solution (from simplex, or from interior point followed by crossover) is used because it contains duals. `itg` is for MIPs. |
| **Primal violation** | By how much the returned solution breaks a constraint or variable bound, divided by `max(1, abs(bound))` (a *relative* violation). |
| **Tolerance** | The largest relative violation accepted: `1e-7` for the native check. |
| **Fallback** | The rare path that rebuilds the Pyomo model (15.5.7). |
| **Phase-I restoration** | The existing procedure that handles an infeasible subproblem (adds penalized slack to recover a feasible point and a valid cut). It needs the Pyomo model. |
| **`DynamicComponents`** | A registry object that model modules append cost, revenue and load-balance terms to while a model is built. |
| **Env** | MOSEK's environment handle (license and global state). Every Task is created from one and the Env must stay alive while any of its Tasks exist. |
| **`putparam`** | The MOSEK call that sets a solver parameter on a Task. |

#### 15.5.2 Problem and cause

In a 72-subproblem run on the legacy Pyomo worker the memory log showed each worker's RSS rising during its first solve (from about 0.44–0.71 GiB to 1.6–2.8 GiB) and staying there. Roughly 68 workers at about 2.5 GiB is about 167 GiB resident, and about 210 GiB once the 1.25 safety factor multiplies each persistent reservation. The budget was then full of persistent reservations, so a worker asking for even the minimum 0.5 GiB temporary reservation at the start of its next trial waited until the 600-second timeout and raised `MemoryError`. The failure was therefore at the start of a trial (after the first solves), not during the model build. Waiting cannot help in that state, because persistent reservations are never released while workers live.

#### 15.5.3 How this satisfies the global memory rule

The rule of Section 15.1 is unchanged:

$$
\\sum_{w\\in W}\\left(R_w(t)+T_w(t)\\right)\\leq M_{\\mathrm{budget}}\\quad\\text{for all }t.
$$

What changes is the *size and timing* of the terms. In offload mode:

* **Build phase.** Each worker reserves its startup estimate, builds, solves the recourse upper-bound LP and compiles the task. The startup semaphore (`n_parallel_solve`) still limits simultaneous builds.
* **After compile.** The Pyomo model, solver object and index maps are deleted, the Task is disposed, and the worker's temporary reservation is released before it reports `READY`. \\(R_w\\) falls to the small idle footprint. This memory is immediately available to workers still building, so a finished worker does not hold memory that others need.
* **Trial.** The worker requests \\(T_w\\) (load, solve and check). Several workers can run trials at once whenever the sum fits. A worker that cannot get its reservation waits, then continues when others finish (up to the timeout).
* **Adaptation.** A background sampler records the worker's RSS every 100 ms. After each trial \\(T_w\\) becomes (peak RSS during the trial minus RSS after it) times the safety factor, never below the minimum (0.5 GiB). \\(R_w\\) becomes the RSS after the trial times the safety factor.
* **Pyomo rebuild paths.** At startup the worker also measures its own build-plus-bound-solve peak above its idle footprint (`build_temp_reservation`). The final export and the fallback request the larger of that value and the trial estimate, because a rebuild needs far more than a task trial.
* **Results.** A worker returns its result after updating its reservations. The master waits for all workers in a window before building cuts, so one slow worker delays the whole round (see 15.5.9).

Resident mode keeps the task in RAM, so \\(R_w\\) stays larger but each trial is faster.

#### 15.5.4 Lifecycle of a worker

1. **Build** the Pyomo operational model with a fresh `DynamicComponents`, and compute the recourse bound (existing logic).
2. **Compile.** Add one linking row per present linking variable, divide the objective by \\(s\\), create a `mosek_persistent` Pyomo solver, call `set_instance`, and write `<subproblem folder>/benders_task/subproblem.task` plus `subproblem.task.json` (metadata). Restore the objective expression.
3. **Release.** Delete the Pyomo model and solver, and (offload mode) dispose the Task.
4. **Trial** (repeated each iteration): open the Task if not resident; apply parameters; set linking-row right-hand sides; run the stage logic of 15.5.6; check the solution (15.5.7); read \\(\\lambda\\) and the cost; end the trial (dispose in offload mode, or reset the touched columns in resident mode).
5. **Export final results** (`EXPORT_FINAL`): rebuild the Pyomo model once (fresh `DynamicComponents`), solve and write GridPath's normal result files, then discard it.
6. **Shutdown:** dispose the Task, delete the `benders_task` folder (unless kept) and release reservations.

Only the metadata needed for Benders stays in RAM: linking rows and columns, the integer and binary columns, and the original bounds of the columns Benders may change. No Pyomo object or per-variable name map for ordinary variables is kept.

#### 15.5.5 Files and folders

Task files live in the subproblem's own folder, next to its input files and results (the D: drive when the scenario is there):

`<scenario>/<weather>/<hydro>/<availability>/<subproblem>/<stage>/benders_task/subproblem.task`

`GRIDPATH_BENDERS_TASK_DIR` replaces `<scenario>` as the root and keeps the rest of the path. The folder is deleted when the worker stops unless `GRIDPATH_BENDERS_KEEP_TASKS=1`.

#### 15.5.6 Trial logic by option

| Situation | \\(z\\) (operational integers) | Discrete linking columns | Lambdas returned for |
| --- | --- | --- | --- |
| Master `pj_relax`, not yet restored | relaxed | relaxed | all present linking variables |
| Operational `pj_relax` (after restoration, or master `linearized`/`djc`) | relaxed | relaxed for the LP | continuous linking variables |
| Operational `fixed_z` with \\(z\\) present | MIP at the trial point, rounded \\(z^\\ast\\) fixed through column bounds, then an LP | integer in the MIP, relaxed in the LP | continuous linking variables |
| Operational `fixed_z`, no \\(z\\) | none | relaxed | continuous linking variables |

Other rules:

* Discrete linking columns are relaxed in **every** LP stage; otherwise no duals exist.
* \\(z^\\ast\\) is rounded to the nearest integer; a deviation above \\(10^{-6}\\) raises an error.
* `master_discrete_state` is returned under the legacy `discrete_cut_phase` rule (master `pj_relax` after restoration, `linearized`, or `djc`, and only when discrete linking variables exist).
* `--benders_operational_solver` (`automatic`, `interior_point`, `primal_simplex`) sets `MSK_IPAR_OPTIMIZER` for the operational-`pj_relax` LP only, as in the legacy worker. All other stages use the configured optimizer.
* All `MSK_*` solver options, the thread cap and `MSK_IPAR_LOG=0` are re-applied with `putparam` after every load and before every stage, because the options passed to Pyomo are not stored in the task.

#### 15.5.7 Native primal check, native retry and Pyomo fallback

1. **Native primal check.** The legacy worker re-evaluated every Pyomo row and bound with `primal_violations()`. That needs the Pyomo model, so the native path computes, with NumPy on the solved task, the worst relative violation of every row activity and every column value against its bounds, and accepts at most \\(10^{-7}\\). The row scale is the bound magnitude; the legacy check also included the sum of absolute coefficients, so the two checks are not identical.
2. **Native retry.** If the check fails, the same, already-fixed LP is re-solved on the same task: first primal simplex with `MSK_DPAR_BASIS_TOL_X = 1e-9`, then interior point with feasibility and gap tolerances \\(10^{-10}\\). The first attempt that solves and passes is used, and a warning "Native primal check passed after retry (...)" names the attempt. Tolerances are then reset to the user's values or MOSEK's defaults so they do not leak into later trials.
3. **Pyomo fallback.** Only if both retries fail, or the native solve is infeasible, the model is rebuilt (fresh `DynamicComponents`, measured build-sized reservation) and the legacy path runs: `solve_original` re-solves on cloned models for a failed check, and Phase-I restoration handles infeasibility. The model is discarded afterwards.
4. **Final export** also uses a rebuilt model.
5. **Non-optimal native LP.** "Not optimal" does not mean "infeasible": MOSEK can also stop on numerical trouble (for example `trm_max_num_setbacks`, solution status `unknown`). The worker records MOSEK's problem and solution status, and whether MOSEK produced a primal infeasibility certificate. Only when there is no certificate does it retry on the same task (primal simplex with basis tolerance 1e-9, then interior point with tolerances 1e-10) and continue natively if a retry solves. Phase-I restoration on the Pyomo model starts only when the retries also fail or MOSEK proved infeasibility; the warning shows `MOSEK infeasibility certificate: YES` or `NO (numerical/unknown)` and the statuses.

The retry exists because the fallback is slow (rebuild plus two high-accuracy solves) and, since the master waits for all workers, two slow workers stall every worker. In the first native run subproblems 31 and 65 failed the check with relative violations of 1.01e-06 and 4.01e-07 (MOSEK's default basis tolerance is \\(10^{-6}\\)).

#### 15.5.8 Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `GRIDPATH_BENDERS_NATIVE_TASK` | `1` | `0` forces the legacy Pyomo worker. |
| `GRIDPATH_BENDERS_TASK_OFFLOAD` | `1` | `1` offload mode; `0` resident mode (faster, more RAM). |
| `GRIDPATH_BENDERS_TASK_DIR` | scenario folder | Root for task files (same sub-layout). |
| `GRIDPATH_BENDERS_KEEP_TASKS` | `0` | `1` keeps `benders_task` after the run. |
| `GRIDPATH_BENDERS_NATIVE_TASK_GAMS` | `0` | The native path is off for the GAMS adapter unless `1` (GAMS option names are not translated). Gurobi and other solvers always use the Pyomo worker. |
| `GRIDPATH_BENDERS_WORKER_BACKEND` | `pyomo` | Kept for compatibility. `mosek_task` only sets `GRIDPATH_BENDERS_NATIVE_TASK=1`. The separate `mosek_task_worker.py` was removed. |
| `GRIDPATH_BENDERS_SOLVER_THREADS` | unset | Caps MOSEK threads per worker (`MSK_IPAR_NUM_THREADS`). |
| `GRIDPATH_BENDERS_MEMLOG` | `0` | `1` prints a memory line per trial. |
| `GRIDPATH_BENDERS_CRASH_LOG` | `1` | Each worker enables Python's `faulthandler`, writing to `<scenario>/benders_crash_logs/worker_<subproblem>_pid<N>.log`. After a native crash (Windows exit code 3221225477, access violation) that file holds the Python stack at the crash, preceded by a line naming the LP solve that was running; it stays empty otherwise. `0` disables. |
| `GRIDPATH_BENDERS_LEAN_EXPORT` | `1` | Final export solves MOSEK models with Pyomo's `save_results=False` path and numeric labels (15.5.16). `0` restores the legacy call. |
| `GRIDPATH_BENDERS_EXPORT_ATTEMPTS` | `4` | Attempts at the final export of one subproblem when it fails for lack of memory; each retry reserves 1.5 times more and pauses first. |
| `GRIDPATH_BENDERS_FALLBACK_ATTEMPTS` | `4` | Attempts at the Pyomo fallback (Phase-I restoration or certificate-checked re-solve) of one trial when it fails for lack of memory; each retry reserves 1.5 times more and pauses first. |
| `GRIDPATH_BENDERS_FALLBACK_FIRST_FACTOR` | `2.0` | Before a worker has measured a real fallback, it reserves this multiple of its build peak for one, because restoration clones the model and builds a second MOSEK task. |
| `GRIDPATH_BENDERS_MEMORY_WAIT_TIMEOUT` | `600` | Seconds to wait for a reservation. Once offload is working, 1800–3600 is reasonable; it should stay finite. |

The Section 15.3 variables keep their meaning.

#### 15.5.9 Waiting, timeouts and stalls

* A worker whose reservation does not fit **waits**, re-checking about once a second and whenever another worker releases memory. It continues when the total fits. After the timeout it raises `MemoryError`. A single request larger than the whole budget fails immediately.
* Waiting helps only for **temporary** reservations, which are released after each trial. If persistent reservations alone fill the budget, nothing is released and the timeout is reached. Offload reduces the persistent part.
* The master collects every worker's result in a window before building cuts, so one slow worker (typically one in the Pyomo fallback) holds up the round. A log with fewer `[benders-mem]` lines than subproblems means the missing ones are still running.
* The coordinator does not control actual allocation (it is cooperative), and raising the budget above the safe fraction of RAM risks an operating-system failure.

#### 15.5.10 Defects fixed in the prototype

| Defect | Effect | Fix |
| --- | --- | --- |
| `set_instance` called on `mosek_direct` | `AttributeError` at compile | `mosek_persistent` solver (or `_set_instance`) |
| Worker's populated `DynamicComponents` reused when rebuilding | Costs, revenues and load balance double-counted in validation, restoration and export models | A fresh `DynamicComponents` on every rebuild |
| Solver options not applied to the saved task | Default threads, optimizer and tolerances | `putparam` after each load and stage |
| Discrete linking columns left integer in operational `pj_relax` outside the relaxed phase | MIP, so no valid duals | Relaxed in every LP stage |
| Unrounded \\(z^\\ast\\) used as fixed bounds | Slightly infeasible LP | Rounded before fixing |
| Preferred solution type mismatch | Duals from the wrong solution | Basic solution first, interior second |
| MOSEK `Env` not kept alive | Possible invalid Task | Runtime keeps the Env reference |
| Native branch skipped adaptive temporary reservation | Coordinator never learned peaks | Per-trial RSS sampling |
| Rebuilds reserved task-sized memory | Under-reservation for export and fallback | Measured `build_temp_reservation` |
| `solve_int_level_set_problem` rejected `required_variables` | `TypeError` at the first stabilised iteration; all planning variables required to have values | Parameter added and passed to `process_planning_sol` |
| Task files in a temporary folder | Not in the subproblem folder | `benders_task` under the subproblem folder |
| Two diverging worker implementations | Inconsistent behaviour | `mosek_task_worker.py` removed |

#### 15.5.11 Observed result

In the first native run all 72 subproblems built and solved, task files appeared in the subproblem folders, and workers' RSS after a trial was about 0.68 GiB (subproblems 1–12), about 1.0–1.06 GiB (subproblems 13–72), compared with 1.5–2.8 GiB (legacy) after the same trial. The idle footprint is not zero: the interpreter and imported libraries remain, so about 0.45–1.0 GiB per worker stays resident, and trial peaks (about 0.7–1.1 GiB) are transient. The run then stalled on the two workers in the Pyomo fallback (31 and 65) before the native retry existed, and failed at the second iteration on the level-set defect above. Both are fixed; the next run is the test of those fixes.

#### 15.5.12 Limitations and what is not done

* **Order inside a worker.** The recourse upper-bound LP is still solved on the Pyomo model *before* compile. The worker therefore does not offload and then solve; the bound could be computed from the task instead.
* **Build admission is count-based.** Concurrent builds are limited by the `n_parallel_solve` semaphore and a flat startup estimate (4 GiB persistent + 4 GiB temporary), not by measured free memory.
* **Cooperative accounting** (15.5.9).
* **Freed Python memory may not shrink RSS.** Deleting the Pyomo model does not guarantee the interpreter returns memory to the operating system; MOSEK memory does return when the Task is disposed.
* **Offload costs a file read per trial.** If load time dominates solve time use resident mode.
* **Native path scope:** MOSEK only (Gurobi, and GAMS unless enabled, use the Pyomo worker).
* **Native check is not identical to the legacy check** (15.5.7).
* **Verification status.** The code was written and compile-checked without MOSEK or Pyomo installed. Equivalence to the legacy worker has not been demonstrated; see 15.5.13.

#### 15.5.13 Verification and log guide

Equivalence test: for each master method (`none`, `pj_relax`, `linearized`, `djc`) crossed with each operational method (`pj_relax`, `fixed_z`), run a few subproblems and trial points with the native path and with `GRIDPATH_BENDERS_NATIVE_TASK=0`, and compare `op_cost`, every linking dual, \\(z^\\ast\\) and the final export to solver tolerance.

| Log text | Meaning |
| --- | --- |
| `[benders-mem] native subproblem=k rss_start=… rss_peak=… rss_end=…` | A native trial finished. `rss_end` close to the idle footprint means the offload works. |
| `Native primal check passed after retry (…)` | The check failed once and the same-task retry fixed it. |
| `Native primal check failed … re-solving through the Pyomo path.` | Both retries failed; the slow fallback is running for that subproblem. |
| `Worker … could not obtain its memory reservation within … seconds` | Persistent reservations filled the budget (15.5.9). |

#### 15.5.14 Files changed

`gridpath/benders/mosek_task.py` (rewritten: compile and `MosekTaskRuntime`), `gridpath/benders/subproblems.py` (native compile, offload, trial, check, retry, fallback, reservations), `gridpath/benders/algorithms.py` (backend switch is an alias; stale import removed), `gridpath/benders/regularization.py` (`required_variables`), `gridpath/benders/linking_bounds.py` (new: FBBT and the bounds cache, 15.5.17), `gridpath/benders/mosek_task_worker.py` (removed), and this document.


#### 15.5.15 Observations from the first multi-iteration native run

* **Stopping is not convergence.** `converged` requires the relative gap to be at most `--benders_conv_tol` (default `1e-10`). The loop also stops on `TIMELIMIT` (`--max_benders_cpu_time`, default 3600 s of wall-clock time measured from the start of `benders()`, which includes building the workers), on `MAXITER` (`--max_benders_iter`; the loop makes `max_iter + 1` passes), and on a stalled gap. A line "Benders terminated: status=… " is now printed before the final export; only `status=OPTIMAL` with a relative gap at or below the tolerance is a converged run. Results exported after `TIMELIMIT` or `MAXITER` belong to the best incumbent found so far, not to an optimum.
* **Why the bounds are enormous in the first iterations.** The problem is a maximization of net present value. The upper bound (UB) is the master objective; the lower bound (LB) is the best value found by evaluating the subproblems at a master solution. In iteration 1 the master has no useful cuts, is unbounded, and the code caps every unbounded planning variable at `1e8` (`BENDERS_MASTER_VARIABLE_CAP`). The master objective is then simply the capped capacity times its cost coefficients (here about `5e18`) and carries no information. Cuts generated at that extreme point are valid but weak, so UB falls only to about `1e18` in iteration 2. The absolute gap `|UB − LB|` and the relative gap `|UB − LB| / |UB|` are computed correctly; the relative gap prints as `1.000` because `|LB|` (about `1e13`) is six orders of magnitude below `|UB|`.
* **Why the ASCII graph looks flat.** The vertical axis spans `[LB, UB]` of all iterations. While UB is about `5e18`, an LB near `1e13` is indistinguishable from zero. The graph becomes informative only once UB falls to the scale of LB.
* **Memory spikes are the Pyomo fallback, not the native path.** In iteration 2 about 20 of 72 workers had an infeasible subproblem at the master's trial point and ran Phase-I restoration on a rebuilt Pyomo model (peaks 3.6–8.3 GiB, against 1.3–1.8 GiB for ordinary native trials). Each trial begins from the unchanged `.task` file, so nothing accumulates in the task and the task is never rewritten (cuts live in the master). The rebuilt Pyomo model is deleted after each fallback, but the operating system keeps part of the freed memory assigned to the process: workers that used the fallback ended with 1.3–2.6 GiB resident, against 0.7–1.0 GiB before. Peaks from simultaneous fallbacks, plus this retained memory, are what approached the memory limit.
* **A real out-of-memory error in the final export.** The final export rebuilds the Pyomo model in every worker. `MemoryError` raised from `_add_block` inside Pyomo's MOSEK interface means Python could not allocate memory at that moment; it is a genuine memory failure, not a modelling error. The coordinator had admitted the export on the basis of `rebuild_temp_reservation`, which was measured at startup and could be smaller than a restoration or export rebuild needs.
* **Change made.** Each worker now keeps `rebuild_temp_reservation`, which starts at the measured build peak and is raised after every Pyomo fallback to the largest transient (peak RSS minus RSS at the start of that trial, times the safety factor) it has needed. Fallback and export reserve this amount. A fallback no longer overwrites the (much smaller) reservation used for ordinary native trials. The memory log marks such trials with `native (pyomo-fallback)`.
* **Why subproblems fall back to Pyomo.** In that run the cause of each fallback was not recorded: the code treated every non-optimal native LP as infeasible, including numerical stalls (subproblem 61 ended with `trm_max_num_setbacks`). The status report and retry in 15.5.7 item 5 now distinguish a proved infeasibility from a numerical failure. Phase-I restoration logs its slack for each subproblem; a tiny slack means a tiny infeasibility.
* **Still open.** Memory the operating system keeps assigned after a Pyomo rebuild is not returned by deleting objects. The robust remedy is to run the fallback and the export rebuild in a short-lived child process, whose memory is returned in full when it exits; this has not been implemented. Reducing the number of simultaneous exports (for example a separate export semaphore) would also lower the peak.
#### 15.5.16 Final export: out-of-memory failure and fix

**Observed.** The run stopped on `TIMELIMIT` (elapsed 7740 s). The final export, which rebuilds each subproblem's Pyomo model in its worker, then raised `MemoryError` inside Pyomo's MOSEK interface in several workers, and one worker died when pandas reported `C error: out of memory` while reading an input CSV. A dead worker is fatal to the run.

**Cause.** The export called `solve(instance, symbolic_solver_labels=True)`. With Pyomo's default `save_results=True`, `mosek_direct._postsolve` builds, for every variable and constraint, a Python name string and a dictionary entry in the `Solution` object, then `load_from` builds another dictionary of all components. For a large subproblem this is several gigabytes, the largest memory peak of the run. Symbolic labels also store a name in the MOSEK task for every row and column. GridPath's export uses neither.

**Fix.**
1. *Lean solve.* For MOSEK the export calls `solve(instance, symbolic_solver_labels=False, save_results=False)`. Pyomo copies primal values (and duals, when an import `dual` suffix exists) straight from the task into the model. Exported values are the same.
2. *Retry on out-of-memory.* An out-of-memory failure (Python `MemoryError` or a library error saying "out of memory") frees the worker's model, asks the coordinator for a reservation 1.5 times larger (so the worker waits for other workers to release memory), pauses 5, 10, 15 s, and retries, up to `GRIDPATH_BENDERS_EXPORT_ATTEMPTS` times. Other exceptions are re-raised.
3. *Timing.* Each native trial memory log line ends with `elapsed=...s`.

**Why the run stopped early.** `status=TIMELIMIT` is not convergence and not a crash. The default `--max_benders_cpu_time` is 3600 s and is counted from the start of `benders()`, including the worker build. Iteration 2 alone took about 1.5 hours, consistent with roughly 20 subproblems running Phase-I restoration on rebuilt Pyomo models (subproblem 61 ended its first solve with `trm_max_num_setbacks`). Raise `--max_benders_cpu_time` to let the algorithm continue.

#### 15.5.17 Bounded first master: FBBT propagation and the bounds cache

**What the switch is.** With `--benders_use_pj_initialization true` (the default) the first master maximises the planning objective alone, with no operating-cost term and no cuts. Any continuous linking variable without an upper bound is then unbounded, the solver reports unbounded, and the code caps those variables at 1e8 (`Planning problem unbounded; capped N variable(s)`). The first trial point is meaningless, many subproblems are infeasible at it, and the upper bound is not informative. With `--benders_use_pj_initialization false` every continuous linking variable is bounded before the first master solve, each worker computes a global bound on its recourse cost, and the first master includes that bound. This is the bounded first master. It is also used automatically when the model has discrete linking variables.

**Definitions.**
* *Linking variable*: a master variable that appears in a subproblem (for example a capacity). *Continuous* excludes binary and integer ones.
* *FBBT (feasibility-based bound tightening)*: interval arithmetic through the master's constraints. If `x + 2y <= 10` with `x, y >= 0`, then `x <= 10` and `y <= 5`. It repeats until no bound improves by more than a tolerance (Pyomo `pyomo.contrib.fbbt.fbbt.fbbt`). It makes no solver call. A bound it derives is implied by the constraints, so it holds at every feasible master point.
* *OBBT (optimization-based bound tightening)*: maximise the variable over the master (integrality relaxed). One LP per variable, so it is slow for many variables.
* *Assumed cap*: if no tier bounds a variable, `BENDERS_MASTER_VARIABLE_CAP` (1e8) is used and is also enforced in the master so that the recourse bound stays valid.
* *Fingerprint*: a SHA-256 digest of every master variable (name, domain, bounds, fixed value) and every active master constraint (name, sides, linear coefficients), plus the settings below. The objective is excluded because OBBT replaces it. The order of components does not matter.

**Order in which a variable's upper bound is sought** (first hit wins):
1. its explicit Pyomo bound;
2. GridPath's capacity / potential constraints (`CAPACITY_BOUND_CONSTRAINTS`), now through an index built once instead of re-reading every capacity constraint for every variable;
3. FBBT, run once for all variables that tiers 1 and 2 cannot bound. FBBT tightens variable bounds in place, so every variable's bounds are recorded first and restored afterwards; the master is not changed. A result of 1e10 or more is discarded as an artefact of big-M chains;
4. OBBT, one master LP per remaining variable, if allowed;
5. the assumed cap.

**Cache.** After the bounds are computed they are written to `<scenario directory>/benders_bounds_cache/linking_bounds_<fingerprint>.json` (atomically). At the next run the fingerprint is recomputed from the master. If a file with that fingerprint exists and covers every linking variable, the bounds and the set of assumed-cap variables are loaded, the caps are re-enforced in the master, and no bound-related solve is done. If the master changed in any variable or constraint (or any setting below changed), the fingerprint differs and the bounds are recomputed; a corrupt file is ignored. Bounds are also kept on the model, so the later call that clears the per-iteration bound cache (after the pj_relax restoration) reuses them instead of re-running OBBT.

**Command-line options.**

| Option | Default | Meaning |
|---|---|---|
| `--benders_use_pj_initialization` | `true` | `false` selects the bounded first master described above. |
| `--benders_bound_propagation` | `true` | Run FBBT (tier 3). |
| `--benders_recourse_bound_obbt` | `true` | Allow OBBT (tier 4). `false` gives the assumed cap to anything FBBT could not bound. |
| `--benders_bounds_cache` | `true` | Read and write the bounds cache. |
| `--benders_bounds_cache_dir` | scenario directory + `benders_bounds_cache` | Folder of the cache. |

The two switches and the cap value are part of the fingerprint, so changing them never returns an old result.

**What this does not speed up.** With `--benders_use_pj_initialization false` each worker also solves a global LP relaxation of its recourse problem to bound its operating cost. That is one LP per subproblem, done in the worker build, and it is not cached by this change.

**Verification status.** The bound logic and cache were exercised on a small Pyomo model with a stand-in solver (explicit, GridPath, FBBT-only, chained FBBT, unbounded variable; cache hit, miss after a constraint change, objective-only change keeps the key, corrupt file ignored, master bounds restored after FBBT). It has not been run on a GridPath model or with MOSEK.

* *Why FBBT failed on the first real run.* It raised `InfeasibleConstraintException('Detected an infeasible constraint.')` and the whole propagation was discarded. That message comes from the check that a numeric constant in a constraint still equals its propagated bounds, using an absolute tolerance (default 1e-8). With coefficients near 1e9 the rounding error alone exceeds 1e-8, so a feasible master is called infeasible. A larger tolerance is more conservative (it never removes feasible points).
* *Tolerance ladder.* FBBT is now tried on the whole model at tolerance 1e-8, 1e-6, 1e-4, 1e-2 and the first pass that succeeds is used. If all fail, FBBT runs constraint by constraint (3 sweeps, tolerance 1e-2), skips and names each constraint it calls infeasible, and keeps everything it learned from the others. The bounds of the master are restored after every attempt. `GRIDPATH_BENDERS_FBBT_MAX_SECONDS` (default 900) limits the constraint-by-constraint mode.
* *Recession-ray shortcut.* A variable `x` whose constraints all stay satisfied when only `x` increases (a row with a finite upper side has coefficient <= 0 on `x`, a row with a finite lower side has coefficient >= 0, no equality row contains `x`) can grow without limit at any feasible master point. OBBT would prove this with one full master solve per variable ("Problem is dual infeasible") and the code would then use the assumed cap. The structural test reaches the same result with one pass over the constraints and no solves; such variables go straight to the assumed cap. The `BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION[...]` variables in the first run were of this kind. Variables in non-linear constraints are never classified this way.
* The bounds-cache version was raised to 2, so caches written by the earlier code are ignored.


**Addendum to 15.5.17: `--benders_zero_linking_vars`.**

Some linking variables are known, from the model rather than from the optimisation, to be zero (for example the performance-standard allocation in the last period). Deriving a bound for them wastes work and gives `1e8` caps and "No finite lower bound" warnings. The option `--benders_zero_linking_vars NAME[:PERIOD][,NAME[:PERIOD]...]` sets the lower and upper bound of those master variables to 0 before any bound is derived.

* `NAME` is the component name without an index (for example `PERFORMANCE_STANDARD_MW_ALLOCATION`).
* `PERIOD`, if given, restricts the rule to variables whose index contains that value (the allocation variables are indexed by zone, period, subproblem).
* Entries are separated by commas or semicolons. Default: none.
* Because the bound is explicit, the variable is skipped by FBBT and OBBT (the explicit-bound tier is taken first), the master cannot move it, and it works with or without `--benders_use_pj_initialization`.
* It is an assumption about the model and is not checked. If the master needs a non-zero value, the master becomes infeasible or more costly.

Example: `--benders_zero_linking_vars PERFORMANCE_STANDARD_MW_ALLOCATION:2050`

**Addendum to 15.5.17: variables without a finite lower bound, and diagnostics.**

* *Why some variables cost solves and got nothing.* A linking variable with no finite lower bound (for example `PERFORMANCE_STANDARD_MW_ALLOCATION[zone, period, subproblem]`, which is free and tied only by "sum over subproblems = 0") can never be given bounds: every route of the bounds search ends by rejecting it ("No finite lower bound found"). It sits in an equality row, so the no-solve test for variables that can grow without limit does not apply, and OBBT used to run one full master solve per variable, once per zone, period and subproblem, before the result was discarded.
* *Fix.* The lower-bound test now comes first, in `compute_linking_variable_bounds` and in `get_continuous_variable_bounds`. Such variables are left out of FBBT, the recession test and OBBT, reported in one summary line (`Benders: N linking variable(s) have no finite lower bound and are skipped without solves`), recorded as skipped in the bounds cache, and receive no recourse bounds, exactly as before. Only the solves are gone.
* *`--benders_zero_linking_vars` diagnostics.* At the start of the bounds step the log prints `Benders: --benders_zero_linking_vars = '<value>'` (`(not set)` if the flag did not arrive). If the flag is set but no variable matches, a warning lists the linking-variable component names of the model. The component name is matched case-insensitively and quotes are ignored.

#### 15.5.18 Cheaper, validated master start: scaled caps, seeded start, interior-point master

**Why.** With the blanket 1e8 cap (15.5.17) the first master returns meaningless points, most subproblems are infeasible there, each infeasible subproblem triggers a slow Phase-I restoration, and the 1e8 values enter the cuts as numerically toxic right-hand sides. Proving a tight bound for every variable first (FBBT, then one LP per variable, OBBT) is not what the literature does: regularised Benders for capacity expansion (arXiv 2403.02559) starts from a plain master, adds a level-set regularisation and uses an interior-point master without crossover; trust-region Benders (arXiv 2610.32862) warm-starts from a cheap relaxed point. The three pieces below follow that line. They are independent options; each can be used alone or together. Nothing here changes the default behaviour.

**Definitions.**
* *Artificial cap (artificial bound)*: an upper bound, or a lower bound for a variable that is free below, that the algorithm adds only to keep the master bounded. It is not part of the model. Every artificial bound is recorded in a registry on the master (`_benders_artificial_caps`).
* *Variable family*: the component name without its index, for example `Cap` for `Cap[zone1,2030]`.
* *Family reference*: the largest finite explicit, GridPath-capacity or FBBT bound among the variables of a family (bounds of 1e7 or more are ignored as artefacts). *Global reference*: the median of the family references. A variable whose family has no reference uses the global reference; if the whole model has none, 1e6.
* *Scaled cap*: `max(1, F x reference)`, where `F` is `--benders_cap_factor` (default 10).
* *Active cap*: a cap that the master optimum sits on (within 1e-6 relative).
* *Master optimum*: the point returned by the master solve itself, not a stabilised (level-set) trial point.
* *Concave maximisation*: the master maximises a concave objective over a convex set (a linear program once cuts are added), so a local optimum is a global optimum.
* *Seed plan*: a set of planning-variable values evaluated by the subproblems before the first master optimum is used.
* *Projection*: the point of the master's feasible set closest to the seed plan, measured by the weighted sum of absolute differences (weighted L1, weight `1/max(1,|seed value|)`), found with one LP over the master's constraints, bounds and any cuts.
* *Crossover*: the step after an interior-point solve that moves to a vertex (basic solution). *Interior-point master*: the master solved by the barrier method with crossover switched off.

**A. Scaled caps: `--benders_bound_strategy scaled_caps` (default `legacy`).**
1. No OBBT is run. FBBT, explicit bounds and GridPath capacity bounds are used exactly as before.
2. Every continuous linking variable that still has no upper bound gets a scaled cap (counted in the line `Benders scaled caps: ...`). This step happens in the bounds routine, so the workers receive the same caps for their recourse bound, and the caps are enforced in the master.
3. Before the first master solve every other continuous master variable without an upper bound gets a scaled cap, and every continuous variable with no finite lower bound gets a scaled lower bound of the same size. The recourse-cost variables (`vTHETA`) and stabilisation variables are never capped.
4. If a master solve is infeasible with scaled caps present (a cap too small for a requirement such as a reserve margin), all artificial bounds are enlarged by `F` and the solve is repeated, up to 8 times.
5. *Validation before stopping.* When the gap is within the convergence tolerance the master optimum is checked. If no artificial cap is active, the optimum is optimal without the caps. If some are active, they are enlarged by `F` and the master is solved again. If the optimal value did not rise by more than the convergence tolerance, the caps were not binding (for example a variable the objective does not use that the solver left at a bound) and the run stops. If it rose, the new value replaces the upper bound (bounds from the smaller box are discarded) and the check repeats from the new optimum, up to 8 enlargements. This is exact for a concave maximisation: a better point outside the enlarged box would imply, along the segment from the optimum, a better point inside it. The log line `Benders cap validation: ...` states how many caps were active, how many enlargements were needed and whether the bound rose.
6. With `legacy` the same activity check runs and only warns.

*Limit: the recourse lower bounds.* With `--benders_use_pj_initialization false` each worker computes a lower bound on its operating cost from an LP that is restricted by the caps it was given. That bound is exact inside the caps and not beyond them. After validation the code warns if any such bound is active at the final master optimum (`recourse cost lower bound(s) are active`); if it does not warn, those bounds did not shape the final master. They are not recomputed when caps are enlarged. With `--benders_use_pj_initialization true` no such bound is used.

**B. Seeded start: `--benders_seed_plan` (default `none`).**
* `none`: as before.
* `lower_bounds`: every continuous variable the subproblems use at its lower bound (0 if it has none, or only an artificial one). This is the "existing system, no new build" plan.
* `auto`: the saved plan `<scenario directory>/benders_incumbent_plan.json` if it exists, otherwise `lower_bounds`.
* a path: a plan file. A missing file stops the run with an error.
The seed is projected onto the master (one LP; the master objective is swapped for the distance and restored), and the projected point is evaluated through the same mechanism as the DJC seeds: the subproblems solve at it, cuts and the incumbent lower bound are recorded, and only then is the first master optimum used. The log prints `Benders seed plan: ...` and `Benders seed plan projected onto the master: moved <sum>`. A seed plan whose projection fails is skipped with a warning.
`--benders_save_plan` (default `true`) writes the incumbent plan, with its objective, to `benders_incumbent_plan.json` whenever the lower bound improves (atomically). A later run on the same or a similar scenario can pass `--benders_seed_plan auto`. Plan files are matched by variable name; names that are not in the master are ignored. A stale plan is only a starting point: the projection makes it feasible for the new master and Benders still proves optimality.

**C. Interior-point master: `--benders_master_interior_point` (default `false`).** MOSEK: `MSK_IPAR_OPTIMIZER = MSK_OPTIMIZER_INTPNT` and `MSK_IPAR_INTPNT_BASIS = MSK_BI_NEVER`. Gurobi: `Method = 2`, `Crossover = 0`. Other solvers: the option is ignored with a warning. It affects only the master optimiser; the subproblem solver choice (`--benders_operational_solver`) is unchanged. Cuts remain valid because they come from the subproblems. The master then returns a point in the interior of its optimal face rather than a vertex, so consecutive trial points move less. It can be combined with A and B. It overrides an optimiser option of the same name from the solver options file. Where the master is a mixed-integer problem the option affects only its continuous relaxations.

**Command-line options.**

| Option | Default | Meaning |
|---|---|---|
| `--benders_bound_strategy` | `legacy` | `legacy` or `scaled_caps` (A). |
| `--benders_cap_factor` | `10` | `F`: scaled-cap multiplier and enlargement factor; must exceed 1. |
| `--benders_seed_plan` | `none` | `none`, `lower_bounds`, `auto` or a plan-file path (B). |
| `--benders_save_plan` | `true` | Write `benders_incumbent_plan.json` on each lower-bound improvement. |
| `--benders_master_interior_point` | `false` | Interior-point master, no crossover (C). |

The bound strategy and `F` are part of the bounds-cache fingerprint, so changing them never returns an old result. Suggested trial for the current batch file: add `--benders_bound_strategy scaled_caps --benders_seed_plan auto --benders_master_interior_point true`; it can be run with `--benders_use_pj_initialization False` as now, or `True`.

**Item checked, not built: can MOSEK's native feasibility repair replace `restore()`?**
`restore()` (restoration.py) builds a Phase-I LP on a clone of the rebuilt Pyomo model: one maximum-slack variable over scaled rows and bounds, the linking rows left unrelaxed, then reads the duals of the linking rows as a feasibility cut, after a primal and dual certificate check. This clone and rebuild is where the observed `MemoryError` and the access violation occurred.
* *What MOSEK's `primalrepair` does* (MOSEK documentation, "Automatic Repair of Infeasible Problems"): it adds non-negative elasticity variables to the bounds of rows and variables, minimises their weighted sum, and then changes the bounds of the task in place by the optimal elasticity values. A negative weight forbids relaxing that bound, so the linking rows can be protected. It works on the task, so no Pyomo clone is needed. The penalty is read from `MSK_DINF_PRIMAL_REPAIR_PENALTY_OBJ`.
* *Why it cannot be dropped in as a replacement.* (1) A Benders feasibility cut needs the dual values of the linking rows of the Phase-I problem. `primalrepair` returns the repaired bounds and the penalty; the documentation does not say that the duals of its internal penalty problem are returned. Solving the repaired task afterwards optimises the original objective, so its duals are those of the original problem, not of Phase I. Using them would give an invalid cut. (2) The penalty is a weighted sum of violations (L1) in absolute units, whereas `restore()` minimises the largest scaled violation (L-infinity) with row scaling; the cut would differ and the existing certificate checks would need re-deriving. (3) I could not run MOSEK here, so the exact behaviour of the duals after repair is unverified.
* *What would work.* Build the same Phase-I problem natively on the MOSEK task: append the elastic columns and the objective with the task API (`appendvars`, `putacol`/`putaij`, `putcslice`), solve, and read the duals of the linking rows (`gety` / `getyslice`). That is `restore()` without the Pyomo clone, so it removes the memory spike and keeps the cut definition and the certificate. A second route is the primal-infeasibility certificate that MOSEK returns when the subproblem is infeasible (the Farkas ray): it gives a valid cut directly with no extra solve, but its scale is arbitrary and it can be weaker; `restore()` was written to avoid those numerical problems. Neither is implemented. I recommend the native Phase-I build as the next change, and it is only worth doing if the restoration memory failures continue after A and B reduce the number of infeasible subproblems.

**Files changed.** `gridpath/benders/master_start.py` (new), `planning.py`, `algorithms.py`, `common_functions.py`, `run_scenario.py`, and this document.

**Verification status.** Exercised on small Pyomo models with a SciPy LP stand-in solver (no MOSEK, no GridPath run): scaled caps by family and by global reference, no cap on `vTHETA`; free variables get a lower cap; active-cap detection and enlargement; infeasible-inside-caps recovery; validation that accepts a cap that is active but not binding, and enlarges a binding cap until the model's own row binds (value 5000 reached after 2 enlargements); projection with the objective restored; seed targets (none, lower bounds, auto, file, missing file); plan save and load; the interior-point option settings; linking-variable scaled caps through `get_continuous_variable_bounds`; the unbounded-master fallback for both strategies. The integration into the master loop (seed evaluation, validation before stopping, plan saving) compiles and was read through but has not been run, because it needs the worker processes. Treat the first real run as the test, and send the log lines starting `Benders scaled caps`, `Benders seed plan` and `Benders cap validation`.

**Addendum to 15.5.18: first real run (scaled caps).** The bounds step took 67 s (4349 variables bounded by FBBT, 4048 by scaled caps, 840 explicit, 17640 skipped for having no finite lower bound). The run then stopped in the workers' recourse-bound LP with `ValueError: No value for uninitialized VarData object GenNewLin_Build_MW[...,2030]` raised by `value(lp_instance.OP_NPV)`.
* *Cause.* A variable that has a zero objective coefficient (here a build variable that appears only in linking rows) is not sent to the solver, so after the solve it has no value, and evaluating the objective *expression* touches it. This is independent of the new options in principle; it surfaced because the bounds changed which variables the solver kept.
* *Fix.* `_initialise_unloaded_objective_vars` gives every objective variable that has no value the value 0 clamped to its bounds before the objective is evaluated (the coefficient is zero, so the objective is unchanged). It is applied in the recourse upper-bound calculation and at the other places that read `OP_NPV` after a solve.
* *Not a defect.* The repeated warning `Recourse bound LP: 232 linking variable(s) still lack a bound` is the free `PERFORMANCE_STANDARD_MW_ALLOCATION` variables (no finite lower bound by design, tied by a sum constraint).
* *Verification.* Compiled and tested only on a small Pyomo model; not run with MOSEK.

**Addendum to 15.5.18: interior-point master read the wrong solution.** In the second real run the workers built and the first master solve (with 44,983 artificial bounds) failed with `mosek.Error: rescode.err_undef_solution(22000): The solution with code 1 is not defined`, raised by Pyomo's `getsolsta`.
* *Cause.* Pyomo's MOSEK adapter reads the *basic* solution (`soltype.bas`) for a linear problem. With `--benders_master_interior_point` MOSEK runs the interior-point method with crossover off (`MSK_IPAR_INTPNT_BASIS = MSK_BI_NEVER`), so no basic solution exists and reading it fails. Quadratic and conic problems (the level-set master) already read the interior solution, which is why only the linear master failed.
* *Fix.* `MosekBendersDirect._set_whichsol` now reads the interior solution (`soltype.itr`) for a continuous linear master when the interior-point option is on (`apply_master_interior_point` sets the flag on the optimiser). Mixed-integer masters are unchanged. Checked only against a stand-in for the MOSEK task: the default still selects `bas`, the interior option selects `itr`.
* *Earlier start-up failure.* The first log of this run ended with a worker waiting 600 s for a memory reservation. The run after it, with the same settings, built all workers, so that failure was about memory pressure at start-up and not about the new options.

**Addendum to 15.5.18: third real run (queued restorations).** Iteration 1 (the seed) finished with LB -1.162e13 and UB -6.607e12. In iteration 2 about 35 subproblems (numbers 25 to 59 and later, the 2035 and later periods) were infeasible and all went to Phase-I restoration at once. Each restoration asks the memory coordinator for a temporary reservation of about twice the build peak, so only a few fit in the budget at a time and the rest queue. After 3600 s the queued workers raised `MemoryError ... could not obtain its memory reservation` and the run died after 1 h 41 min, although the workers at the head of the queue were still making progress.
* *Fix.* The wait timeout in `BendersMemoryCoordinator.set_reservation` now restarts whenever the total reserved memory changes. A worker waiting its turn behind progressing workers no longer fails; only a wait in which no reservation changed for `GRIDPATH_BENDERS_MEMORY_WAIT_TIMEOUT` seconds (a real stall) raises. Checked with a thread-based test: four workers needing 10 GiB each against a 20 GiB budget and a 0.5 s timeout all finish; a holder that never releases still raises.
* *Not fixed by this.* The restorations still run only a few at a time, because each rebuilds a Pyomo model. The remaining cost is the number of infeasible subproblems and the time of each restoration. See the native Phase-I option in the "Item checked, not built" paragraph above.
* *Suspected cause of the infeasibility (unconfirmed).* The infeasible subproblems are the periods that carry policy allocation variables (performance standard, and probably carbon cap and energy target). The master splits each policy target across a period's subproblems and has no information about the headroom each needs until feasibility cuts arrive.

### 15.5.19 Decomposition audit fixes, native Phase-I and allocation seeding

**Performance-standard MW allowance.** Every subproblem carries the full capacity-based allowance `rate_MW * Capacity` on its own left-hand side, so summing the K subproblem constraints of a period counted it K times. The master constraint is now `sum_w a_w = -(K-1) * rate_MW * Capacity`, where `Capacity` is the master expression `Benders_Performance_Standard_Capacity_MW` built from `Capacity_MW` and the project/zone mapping read from `project_performance_standard_zones.tab`. The decomposed constraint then equals the monolithic one.

**Modules.** Eight modules that were in neither the master nor the operational list were silently dropped from a Benders run (transmission curtailment costs, transmission energy-target contributions and losses, transmission policy contributions, generic-policy transmission contributions, water results consolidation). They are operational and are now in the operational list. `objective.max_npv` stays excluded because the Benders objective replaces it.

**Linking expressions.** `define_linking_vars` now also scans `Hyb_Gen_Capacity_MW`, `Hyb_Stor_Capacity_MW`, `Energy_MWh` and the three fuel capacity expressions (when present), so a build variable that reaches operations only through one of them is a linking variable.

**Generic policy.** Requirements that span subproblems (annual horizons and month-hour requirements) have no allocation. A warning is logged when a subproblem contains such rows.

**Native Phase-I.** An infeasible native LP is no longer sent to the Pyomo restoration. `MosekTaskRuntime.phase_one` adds non-negative slack columns to every non-linking row of the loaded task (`p` for rows with a lower bound, `n` for rows with an upper bound), replaces the objective by `min sum_i w_i (p_i + n_i)` with `w_i = 1/max(1,|l_i|,|u_i|)`, keeps the linking rows and column bounds exact, and solves. The optimum `v(xbar)` is convex and piecewise linear in the linking right-hand sides and is zero exactly when the subproblem is feasible, so the linking-row duals give the feasibility cut `v + sum_j lambda_j (x_j - xbar_j) <= 0`. The cut is rejected (and the Pyomo restoration used) when the solve is not optimal, the violation is zero, or the relative duality gap exceeds 1e-6. Operational MIP subproblems keep the Pyomo path. Set `GRIDPATH_BENDERS_NATIVE_PHASE1=0` to disable. The task is disposed afterwards, so the next trial re-reads the pristine task.

**Phase-I diagnostic.** For every native Phase-I the log reports the constraint groups with the largest weighted violation (using a row-to-group map stored in the task metadata; re-export the tasks if it is missing) and the five linking variables with the largest sensitivity. Set `GRIDPATH_BENDERS_PHASE1_DIAG=0` to silence it.

**Allocation seeding.** During the seed projection each period budget is pulled toward the day-weighted split `a_w = (n_w / sum n) * sum a` (a weak extra L1 term, so builds keep priority). Allocations found in a saved incumbent plan keep the plan's values. The lower-bound seed no longer drives allocation variables to zero.

### 15.5.20 Elastic allocations, generic-policy allocation and master coverage of allocation types

**Why.** The policy allocations are chosen by the master before it knows anything about each representative day, so a subproblem can be handed a share it cannot meet. The recourse problem then lacks *relatively complete recourse*: not every master-feasible point leaves a feasible subproblem, and the master needs feasibility (Phase-I) cuts, which are more expensive and carry less information than optimality cuts.

**Elastic allocations (`--benders_elastic_allocations true`).** For the first `--benders_elastic_iterations` iterations (default 10) every non-linking row that has a non-zero in a policy-allocation column may be violated. The loaded MOSEK task gets non-negative slack columns on those rows (`p` for a lower bound, `n` for an upper bound), priced at
`penalty_k * ||c||_1 * w_i` per unit, with `w_i = 1/max(1,|l_i|,|u_i|)`, `||c||_1` the sum of absolute objective coefficients of the subproblem, and `penalty_k = --benders_elastic_penalty * --benders_elastic_growth^k` (defaults 0.1 and 2). The subproblem is therefore always feasible with respect to the allocations and returns an ordinary optimality cut. After the last elastic iteration the rows are strict again and any remaining infeasibility goes through the native Phase-I.

Validity. Let `Q_P(x)` be the recourse cost with penalty `P` and `Q(x)` the true recourse cost (infinite when infeasible). Then `Q_P(x) <= Q(x)` and `Q_P` is convex, so an optimality cut built from an elastic solve is a valid lower bound on the true cost, whatever the penalty. The master bound therefore stays a valid upper bound on the NPV when the penalty changes between iterations. The lower bound is different: an evaluation whose elastic violation exceeds 1e-6 is not a feasible plan, so it never updates the incumbent (`elastic_violation` is returned by the worker and checked in `benders()`); only violation-free evaluations do. Elastic mode works with native MOSEK workers only (the Pyomo path ignores it). The elastic trial modifies the task, so it is disposed afterwards and re-read on the next trial.

Tuning. A penalty that is too small returns cheap but uninformative cuts (the subproblem simply pays); too large and the cut slopes are huge and badly scaled for the master. The geometric ramp starts weak and strengthens; if the log shows violations persisting at the last elastic iteration, raise `--benders_elastic_iterations` or `--benders_elastic_penalty`.

**Generic policy.** `Policy_Requirement_Constraint` rows with balancing type `year` and a horizon equal to a model period now use an allocation. In the subproblem the right-hand side is `GENERIC_POLICY_ALLOCATION[policy,zone,bt,h,subproblem] + (Policy_Zone_Horizon_Requirement - policy_requirement)`: the scalar part of the requirement is split by the master, and the percent-of-load part stays per subproblem because each subproblem only sees its own load. The master has `Generic_Policy_Allocation_Constraint`: `sum over subproblems of GENERIC_POLICY_ALLOCATION >= policy_requirement`. The generic-policy requirements module cannot be loaded in the master (its requirement expression needs load), so the master reads the scalar requirements itself from `policy_requirements.tab` (`BENDERS_GENERIC_POLICY_REQ_ROWS`, `benders_generic_policy_requirement`). Before this change every subproblem enforced the full annual scalar requirement, K times too strict.

*Not done: month-hour requirements.* `Policy_Month_Hour_Requirement_Constraint` is still enforced in full in every subproblem and a warning is logged. An allocation needs to know which representative days contain each (month, hour) pair, otherwise the master hands a share to days that cannot contribute; that map must come from the temporal inputs on both sides. Until it is built, set `allow_violation` on month-hour policies or expect infeasible subproblems.

**Fuel-burn and transmission-target allocations in the master.** The master only builds these allocation variables and sum constraints when the target sets exist, and those sets were declared by the balance modules, which are operational-only. A scenario that used them failed at the first master solve with `Required linking variable ... was not found in the active master variables`. The parameter-only modules `system.policy.fuel_burn_limits.fuel_burn_limits` and `system.policy.transmission_targets.transmission_target` are now in the shared "Sets" list, and `benders_policy_constraint` declares the annual target sets by calling the balance modules' own helper functions. The first real run of this path failed with `TypeError: 'int' object is not iterable` while constructing `TRANSMISSION_TARGET_POS_DIR_MIN_ALLOCATION_INDEX`: the index-set initialiser was a two-argument lambda (`lambda mod, target_set=target_set: ...`), which Pyomo treats as a counter-style rule and calls with an integer. It was dead code until the master loaded the target sets. It now uses the same one-argument factory (`_make_period_allocation_index_rule`) as the other allocation types (patch 22).

**Capacity-factor allocations: still missing in the master (investigated).** The master constraint is `sum_w a_w = cf * Possible_Power_Provision_in_Horizon`, and `Possible_Power_Provision_in_Horizon_Expression` needs the availability and capacity-factor profile of every timepoint of every representative day, which exist only in the operational modules. A scenario with capacity-factor limits therefore fails with the same missing-variable error. The fix needs data plumbing: each subproblem's possible provision is linear in the project's capacity, `Possible_w = coef_w * Capacity_MW[prj, period]`, so each worker can report `coef_w` once at start-up and the master can build `sum_w coef_w * Capacity` from them. Until then, do not combine `--benders` with capacity-factor limits.

**Phase-I row weights.** The Phase-I weights use only the row bounds, not the row coefficients. Using the coefficients would improve scaling but would not make the Phase-I solve cheaper (same number of rows and columns), so it was left unchanged. The cost reduction comes from running Phase-I natively and, with elastic allocations, from running it less often.

### 15.5.21 Master infeasible inside the artificial caps (`err_undef_solution`)

*Symptom.* The first master solve printed `Problem is primal infeasible` and the run then stopped with `mosek.Error: rescode.err_undef_solution(22000): The solution with code 1 is not defined` raised from `solve_planning_problem` (`planning.py`, the retry line).

*Cause.* Two things. (1) The scaled caps on the unbounded allocation variables (`PERFORMANCE_STANDARD_MW_ALLOCATION`, which has no natural bound) start from a fallback of 1e6 per variable, and the corrected master constraint `sum_w a_w = -(K-1) * rate * Capacity` can require a total larger than K times that, so the first master is infeasible inside the caps. That is the situation the cap-enlargement loop is for. (2) Before enlarging, `solve_planning_problem` retried every non-optimal result with a fresh stock `SolverFactory` solver. A stock solver reads the basic solution, which does not exist when the master is solved with the interior-point method and no crossover, hence the undefined-solution error. The retry crashed before the enlargement loop could run.

*Fix.* (a) The fresh-solver retry is skipped when the configured optimiser is the interior-point master or when the result is already `infeasible` (re-solving cannot change an infeasibility proof). (b) `MosekBendersDirect._postsolve` now catches `err_undef_solution` and reports it as an infeasible result (or `unknown` if MOSEK reports no infeasibility status), so presolve-detected infeasibility reaches the cap-enlargement loop instead of raising. The caps are then enlarged by `--benders_cap_factor` up to eight times and the master re-solved.

*Better starting caps for allocation variables.* To avoid spending enlargement rounds on this, `scaled_cap_references` now takes the budget each allocation family has to carry from the master's own sum constraints (`allocation_required_totals` in `master_start.py`):
- `PERFORMANCE_STANDARD_MW_ALLOCATION`: `max over (zone, period) of (K-1) * rate_MW * UB(Capacity)`, where `UB(Capacity)` is the upper bound of `Benders_Performance_Standard_Capacity_MW` from interval arithmetic (`pyomo.contrib.fbbt`) on the current build-variable bounds. Skipped if the capacity has no finite upper bound.
- `PERIOD_ENERGY_TARGET_ALLOCATION`, `CARBON_CAP_ALLOCATION`: the largest `Period_Energy_Target` / `carbon_cap_target`.
- `GENERIC_POLICY_ALLOCATION`: the largest scalar requirement.
The cap is the usual `--benders_cap_factor` times that reference (default 10x). These totals are used only as family references: they do not enter the global reference that other unbounded families fall back to, so builds and similar variables are not capped more loosely because of them. If the day weights are not available on the instance, equal weights over `SUBPROBLEMS_BY_PERIOD` are used.

*Note.* The log also shows `Recourse bound LP: N linking variable(s) still lack a bound` for `PERFORMANCE_STANDARD_MW_ALLOCATION`. That is expected: the allocation has no natural bound and the warning is informational.

## 16. References

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
