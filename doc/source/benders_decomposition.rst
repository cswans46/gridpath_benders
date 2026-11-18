###################################
GridPath Benders Decomposition
###################################

In this chapter, we discuss GridPath's Benders decomposition integration. 
Benders Decomposition is a mathematical technique used to solve large-scale 
optimization problems by decomposing them into smaller
parts. It leverages the strong duality assumption of linear, mixed-integer, conic, 
and convex quadratic programs as well as the
block-diagonal structure of common capacity-expansion and optimal dispatch
problems. Gridpath's Benders Decomposition algorithm is robust to various 
problem structures including operational and investment MIPs, convex QCQPs, 
multi-planning-period dynamic optimisation, and stochastic programming. However,
please note that the Benders algorithm may not be used for subproblem sizes that
exceed the balancing horizon nor with "stages" that are currently set up to link
planning periods. This section is subdivided into three sections:

1. :ref:`core-integration-section-ref`
2. :ref:`file-map-section-ref`
3. :ref:`extensions-section-ref`

.. _core-integration-section-ref:

***************
Core Benders Integration
***************

Initialisation
==========================

In order for Benders decomposition to be effective, n_subproblems must be greater than 1 in 
the input files. Benders decomposition will work with hydro, availability, and weather
iterations.

If Benders decomposition is desired, activate it using the '--benders 1' flag 
when running run_end_to_end.py, such as this:

  python gridpath/run_end_to_end.py --scenario_location /PATH/TO/SCENARIO --database /PATH/TO/DATABASE --scenario_id SCENARIO_ID --benders 1

Convergence criteria are options that may be included as a line item in the above, but default to the following:

- `--max_benders_iter`: 200
- `--benders_conv_tol`: 1e-10
- `--benders_abs_gap_tol`: 1e-4
- `--max_benders_cpu_time`: 3600

`--max_benders_iter` controls the maximum number of iterations to achieve the 
`--benders_conv_tol` relative optimality gap or the `--benders_abs_gap_tol` absolute
optimality gap (denominated in the objective function's units ($)). 
`--max_benders_cpu_time` limits
the total CPU time required to converge in seconds.

Supplementary options may be added as desired for Benders decomposition for cases of 
MIP master and/or operational problems and desired stabilisation methods. 
These are set to the following defaults:

- `--benders_stab_dynamic`: True
- `--benders_stab_param`: 0.025
- `--benders_stabilization_method`: "none"
- `--benders_master_discrete_method`: "pj_relax"
- `--benders_operational_discrete_method`: "fixed_z"

`--benders_stab_dynamic` is a boolean option to request stabilisation, which is then
controlled by `--benders_stab_param`, which may be :math:`benders_stab_param \in [0,1]`.
:math:`--benders_stabilization_method \in \{\text{"none"}, \text{"l1"}, \text{"l2"}\}` where
:math:`--benders_stabilization_method` specifies the type of stabilisation to apply, 
`none` either refers to no stabilisation or standard Pecci \& Jenkins (2024) regularisation
depending on the values of `--benders_stab_dynamic` and `--benders_stab_param`. If these
are off, then no stabilisation is applied with `--benders_stabilization_method = "none"`. If
these are activated (both :math:`--benders_stab_dynamic=True` and :math:`--benders_stab_param > 0`), 
then `none` refers to Pecci \& Jenkins (2024) regularisation. `l1` and `l2` refer to
alternative stabilisation methods that penalise first-order (absolute) and second-order 
deviations from the previous iteration. Please see additional details 
on stabilisation methods in :ref:`extensions-section-ref`. Note that, if using MOSEK, the
`l2` stabilisation method can only be used with MOSEK's conic solver and therefore can only
be used with the nonlinear or conic extension on the base MOSEK license.

:math:`--benders_master_discrete_method \in \{\text{"none"},\text{"pj_relax"}, \text{"linearized"}, \text{"djc"}\}` 
specifies the method requested to handle a discrete master problem. `\text{"none"}` can only be used
in the case where there are no master binary or integer variables. `\text{"pj_relax"}` refers to the 
standard linearisation and subsequent integrality restoration procedure in Pecci \& Jenkins (2024).
`\text{"linearized"}` maintains master integrality and represents state-conditioned recourse cuts with a linearise
activation/mismatch formulation (Big-M). Finally, `\text{"djc"}` uses state-conditioned disjunctive
constraints for binary (not integer) linking variables. Please see additional details 
on discrete master methods in :ref:`extensions-section-ref`.

:math:`--benders_operational_discrete_method \in \{\text{"none"}, \text{"pj_relax"}, \text{"fixed_z"}\}`
specifies the method used for handling a discrete operational problem where genuine operational variables
that are not linked to the master problem are discrete. `\text{"none"}` can only be used when there are
no genuinely discrete operational variables. `\text{"pj_relax"}` refers to the standard linearisation
without integrality restoration used in Pecci \& Jenkins (2024). `\text{"fixed_z"}` solves the operational
problem twice, once as a MIP and once as an LP with MIP values of operational discrete variables fixed 
using a linear constraint. 
Please see additional details on discrete operational methods in :ref:`extensions-section-ref`.

Pre-Benders Routine
==========================

When `run_end_to_end.py` is executed, the scenario .tab files are generated.
With multi-period Benders, investment-related capacity and policy .tab files
are truncated only to include the relevant periods for that subproblem. These
tab files are placed in the input subfolder within additional subfolders corresponding to the
name of the subproblem within the  scenario directory. These are then aggregated to form the inputs
for the master problem. In this way, we exclude all superfluous or potentially fatal projects and periods
since only those projects and periods that are relevant to the current subproblem are optimised.
For capacity-related variables, since cumulative capacity informs capacity decisions in the
current time-period, units with capacity vintages less than or equal to the year associated 
with each subproblem are kept. This requires periods set to be a weak subset of
the vintages set in each subproblem whereas they are synonymous in a non-Benders run.

Once the input files for all subproblems are generated, the algorithm that
pools key information separately for each subproblem and stochastic iteration
to form the directory to which results will be written for each subproblem has
been altered. The keys associated with each subproblem and stochastic iteration
are collected in the `run_scenario` function 
then pushed through to the model construction stage as they are
not required for building the master problem and the operational subproblems are built
at a later stage of the algorithm. Note that the `n_parallel_subproblems` parameter can either
be left unset or set to 1. The algorithm endogenously calculates 
the number of subproblems.

The master problem is created and instantiated first in `run_optimization_for_subproblem_stage_benders` which
calls `create_problem`, only passing "" for each of the stochastic and subproblem iteration arguments.
This ensures that the abstract master problem is constructed and instantiated
using data from the aggregate input files. Within `create_problem`, `set_up_gridpath_modules` is called which calls `determine_modules`
which splits investment and operational modules. This is an initial rough split because 
some model objects required for the master problem are constructed in operational modules. Later in the
algorithm, all operational constraints and variables are excluded from the master and vice versa
except for those required linking variables. In `run_optimization_for_subproblem_stage_benders`,
we create a key (`benders_subproblems`) which holds all iterable subproblem information for the problem which is then passed
to `solve_problem`. `solve` defines the linking variables in `define_linking_vars` (which is where the
manual list of linking variables is hard-coded). The setup variables are then associated with their keynames which
will be used in the `benders` function here. Then `benders` function in `algorithms.py` is then called.

Benders Routine: Initiation, Master Problem Setup, and Multiprocessing
=======================================================================

`benders` begins by adding the approximate operational cost to the master objective function such that 
::math::`\text{Master} - BENDERS_THETA_SCALE * \sum_w theta_w`. An initial lower bound is added to each `theta_w`
such that the feasible space ::math::`theta_w \in (-\infty, \infty)` to ::math::`theta_w \in (ILB, \infty)`, where 
`ILB` is the initial lower bound value. We do not constrain `theta_w` from above as it would be pointless since
the model looks to choose the lowest possible feasible value for `theta_w`, representing operational costs. For the
initial master solve, `theta_w` is unassociated with an cuts from the subproblems, so the model could theoretically
choose an unrealistically low value for `theta_w` which would artificially boost the planning objective. This
initial point would make convergence less tractable. The master objective is then replaced by the Master with constrained
operational cost approximation above.

The master variables are then split (if required) binary, integer, and continuous variables based on their domains.
In the `pj_relax` `benders_master_discrete_method` case, the binary and integer investment variables are relaxed (::math::`x \in \{0,1\}` to ::math::`x \in [0,1]` 
and ::math::`y \in \mathbb{Z}` to ::math::`y \in \mathbb{R}`). For all other `benders_master_discrete_method` cases 
where there exist binary and/or integer variables, the binary and integer investment variables remain discrete.

Workers are then split using python's native multiprocessing module for parallel multiprocessing of subproblems. Note
that this algorithm requires the availability of multiple CPU cores and logical processors. Ideally the number of which
would be at least as great as the number of subproblems being solved in parallel. Any excess subproblems above the logical processor
amount will be solved sequentially, locally to each logical processor. Recall that this also requires 
a license that permits multiple simultaneous solves (one per logical processor). For Gurobi, this appears (as of the current
date, October 8th, 2026) to be the Computer Server License. For IBM, this appears to be the IBM ILOG CPLEX Optimizer Cluster Edition.
For MOSEK, this is the floating node-locked license PTON. The MOSEK nonlinear and conic extension would permit the use
of the `l2` stabilisation method described in :ref:`extensions-section-ref`. the `ctx.Process()` calls the target function
and initiates its parallel execution for all combinations of `key` giving one value of the rest of the arguments to each `key`.

Benders Routine: Initiation, Master Problem Setup, and Multiprocessing
=======================================================================

In `benders_subproblem_worker` in `subproblems.py`, first, each operational abstract model is 
created and instantiated before the loop. This begins in `benders_subproblem_worker`
where the parent process launches all worker processes at once without staggering their creation as each worker
creates a _Startupguard and calls it using guard.acquire(). Semaphore is used to guard against
spillage as when multiple workers attempt to initialise subproblems simultaneously, this can
lead to RAM-to-harddrive spillage which can severely reduce performance. Additionaly, launching a large
number of processes simultaneously without a Semaphore guard can cause the operating system to constantly pause and resume
processes as space is used up. Because the number of subproblems is ideally large, this could liead to memory wastage through
cycles swapping memory pages and caches. `_benders_subproblem_worker_impl` is then called from 
the Semaphore wrapper/guard function.
Share constraints between the operational and investment models
are deactivated including all uniquely associated variables. Linking variables are then classified
in `classify_linking_variables` into binary, integer, and continuous categories. For the "linearized"
`benders_master_discrete_method`, the recourse upper bound is computed taking the objective value after having 
relaxed operational integrality and solved the linearised problem. This upper bound
local to each worker then negated before being used as a valid lower bound for the net-operating cost (`theta_w`).
This is then optimised in the planning problem bounding the net-operating cost from below.

Each worker's `output_queue` is what communicates the result 
(including recourse upper bound, linking variables, and objectives) 
and the status of each subproblem worker with the master problem and `input_queue`
communicates the input data for the subproblem worker from the planning problem.
`benders_subproblem_worker` begins by deactivating shared constraints that are non-linking
between the operational and master problems. This is to ensure that any residual constraints
left over after module separation in `module_list.py` and `module_organisation.py` do not
interfere with convergence. 

Benders Routine: LP restoration, Subproblem Execution, and Master Linkage
=========================================================================

Within each worker, the operational model is iterated with the current linked
master variables. Non-optimal operational LPs enter a separate feasibility
restoration problem. An optimal solver status is also checked against the
original operational model before accepting an incumbent. Restoration and
numerical recovery are described in :ref:`benders-restoration-section-ref`.
The type of feasibility cut depends on the master discrete method; see
:ref:`extensions-section-ref`.

The algorithm then begins by dealing with master discrete linking variables
according to the method chosen. In the `pj_relaxed` phase, all discrete linking
variables are relaxed in the subproblem such that the subproblem becomes a true
continuous LP. In other cases, only continuous linking variables are kept and used
to form the continuous affine Benders cuts whereas binary linking variables are dealt with
separately (see :ref:`extensions-section-ref`). Linking variables are then fixed to
the preceding master solution in `fix_linking_variables` by adding a linear constraint
rather than fixing the variable bounds for dual extractability. 
`detect_active_mip_variables` is used to identify genuinely
operational and active binary or integer variables in the initial problem 
(integrality would have already been relaxed in the pj_relax phase). How operational MIPs are handled then
depends on `benders_operational_discrete_method`. By default `fixed_z` is chosen
so as to solve two separate problems, one MIP then one LP with integer or
binary variables constrained to their MIP solution. However, when computational
efficiency is worth more than the true integrality of the operational subproblem,
`pj_relax` is a widely performed substitute which simply linearises 
all integer and binary variables (over their respective :math:`x\in\{0,1\}` for binary and :math:`x\in\mathbb{X}` for integer variables)
and solves one operational LP. However, the initial parameters of each variable
are retained for integrality restoration (such as initial value, domain, 
lower, and upper bounds). In the case of `pj_relax` and with MOSEK or Gurobi,
the LP algorithm is set to a solution method dependent on the value of the user
defined `--benders_operational_solver` which defaults to `automatic`
but is recommended to be set to `interior_point` or `primal_simplex`. 

For MOSEK these map to
`MSK_IPAR_OPTIMIZER=MSK_OPTIMIZER_FREE`, `MSK_OPTIMIZER_INTPNT`, and
`MSK_OPTIMIZER_PRIMAL_SIMPLEX`, respectively.
For Gurobi they map to `Method=-1` (automatic), `Method=2` (barrier),
and `Method=0` (primal simplex).
The solver option and scaled objective are restored after each solve, including
when the solve raises an exception. This option controls the `pj_relax` solve;
fixed-z and recourse-bound LPs retain an existing algorithm option, or use
automatic selection if none is set. Feasibility restoration and unchanged-model
recovery use primal simplex first, followed by one interior-point attempt if
numerical validation fails. Solver-specific tolerances are scoped to these solves.
Dual simplex is not an explicit selection or retry option. Automatic selection
does not prohibit the solver from internally choosing dual simplex; use an
explicit `interior_point` or `primal_simplex` selection for the `pj_relax` solve
when that distinction matters. Poor scaling can produce inaccurate objectives
and duals despite an optimal solver status, so restoration additionally validates
the numerical certificate rather than relying only on the selected algorithm.

Finally, in the case of an operational LP, the LP is solved and duals are extracted conventionally.
In the `fixed_z` case and if the MIP termination condition is optimal, binary and integer variables
are now relaxed then constrained to their presolved binary or integer optimal values.
The LP is then solved in `_solve_scaled_recourse_lp` just as in the `pj_relaxed` case
described above in the case of an operational MIP. Otherwise, in the case of an optimal terminal
condition in the other `benders_operational_discrete_method` cases, the `lp_results=mip_results` at this
point.



.. _benders-restoration-section-ref:

Feasibility Restoration
========================

Purpose and original-model feasibility
--------------------------------------

An infeasible operational LP at a particular master trial point is expected
during Benders decomposition. It does not establish that the complete scenario
is infeasible. For example, adding a zero coal-burn limit or a finite gas-burn
limit can make an existing investment/policy allocation infeasible; restoration
must then guide the master toward a different allocation.

``gridpath.benders.restoration`` constructs Phase I on a disposable clone of the
operational LP. The persistent worker's physical constraints, bounds, domains,
objective, and linking equalities are not relaxed in place. Finite non-linking
rows and non-fixed continuous-variable bounds are made elastic, while linking
equalities retain the trial point. Operational integer recourse must first be
handled by the configured discrete method: LP-dual feasibility cuts are not
valid for genuinely infeasible unfixed operational integer recourse.

Phase I minimizes a nonnegative maximum normalized slack, :math:`t`. For row
scale :math:`s_i`, an equality is represented by
:math:`a_i y - s_i e_i = b_i` with :math:`-t \leq e_i \leq t`;
finite inequality sides and continuous-variable bounds receive corresponding
elastic terms. Sides with infinite bounds are omitted. In particular, a
completely unconstrained row is not converted into an expression containing
infinity plus a slack variable.

Small Phase-I slack is **not** permission to widen the original constraints or
variable bounds. If a reliable positive slack exceeds ``1e-9``, a feasibility
cut is returned, even when the slack is below the former ``5e-5`` threshold.
A slack at or below ``1e-9`` triggers a scaled retry of the unchanged operational
LP. Only an optimal retry passing original-model checks can provide a feasible
incumbent. If Phase I is near zero but the unchanged LP remains infeasible,
the algorithm reports numerical ambiguity rather than manufacturing feasibility.

Scaling and numerical certificates
----------------------------------

Elastic row scales normalize violations, but do not by themselves condition the
solver matrix. Before solving the clone, each finite row is divided by the
maximum of one and its largest absolute coefficient. Linking rows are included
in this numerical scaling, but are never made elastic. Trial-point RHS magnitude
is deliberately not used for the divisor: an extreme master allocation must not
shrink an essential linking coefficient below the solver's numerical resolution.
The Phase-I objective is multiplied by ``1e6`` to avoid very small objective
coefficients and associated duals.

Restoration tries primal simplex, then at most one interior-point solve.
For MOSEK, these attempts use ``MSK_DPAR_BASIS_TOL_X=1e-9`` and
``MSK_DPAR_INTPNT_TOL_PFEAS``, ``MSK_DPAR_INTPNT_TOL_DFEAS``, and
``MSK_DPAR_INTPNT_TOL_REL_GAP`` set to ``1e-10``. Gurobi uses
``FeasibilityTol=1e-9``, ``OptimalityTol=1e-9``, and ``BarConvTol=1e-10``.
HiGHS uses primal and dual feasibility tolerances of ``1e-9``.
Options are restored even if the solver raises an exception; no license path
or machine-specific solver setting is stored in the model.
If a solver interface raises an exception because a non-optimal result has no
loadable incumbent, termination is explicitly classified with solution loading
disabled. Such a result is never used as an operational solution or certificate.

An optimal status alone is insufficient. Before accepting a Phase-I solve, the
code checks finite values, primal feasibility, dual/reduced-cost signs,
stationarity, complementarity, and the primal-dual objective gap. Certificate
checks use ``1e-8`` scale-aware tolerances. Primal backward-error checks include
the sum of absolute row activities to distinguish floating-point cancellation
at extreme trial points from a meaningful infeasibility. This backward-error
allowance is for the Phase-I certificate only; it does not widen the original
operational model.

Ordinary optimal operational solutions and unchanged-model retries are checked
against original rows, bounds, linking equalities, and integrality, using a
``1e-7`` residual tolerance normalized by row RHS/coefficient magnitudes or
variable bounds. Missing or non-finite values are errors, not zero residuals.
A nominally optimal solution failing these checks is retried on a scaled clone.

Feasibility cuts and failure handling
------------------------------------------------------------

The cut intercept and linking duals come from the same validated Phase-I solve.
For a numerical row divisor :math:`d_i` and objective multiplier :math:`C`,
the original-unit dual is
:math:`\lambda_i = \lambda_i^{solver}/(d_i C)`.
The intercept is the smaller of the primal slack and the validated dual objective
in original slack units. This avoids overstating the intercept by the
primal-dual gap.

The resulting feasibility inequality has the form

.. math::

   0 \geq \phi(\bar{x}) + \lambda^\mathsf{T}(x-\bar{x}).

Its entire intercept and gradient are multiplied by
:math:`\max(1,1/\phi(\bar{x}))`, keeping a meaningful small infeasibility visible
to the master solver without changing the half-space. Restoration results retain
the unscaled ``restoration_slack``, ``restoration_dual_gap``, and
``restoration_cut_scale`` for diagnostics. These are not operating costs.

Rejected certificates are logged with the subproblem and numerical reason.
After the bounded attempts are exhausted, an explicit exception stops the run.
``GRIDPATH_BENDERS_RESTORATION_LP`` optionally writes the final attempted
restoration LP with a subproblem suffix and symbolic row names.
``GRIDPATH_BENDERS_INFEAS_DIAG`` logs the slack, dual gap, and cut scale;
``GRIDPATH_BENDERS_CUTLOG`` appends the master point and scaled cut coefficients
in JSON-lines format. Diagnostic output does not re-solve or modify the model.

Regression tests in ``tests/test_benders_restoration.py`` cover zero coal and
finite gas limits, near-infeasible trial points, infinite rows, certificate
rejection, dual scaling, cut support at perturbed/feasible points, original-model
preservation, repeated restoration transitions, and solver-option cleanup.
.. _file-map-section-ref:
