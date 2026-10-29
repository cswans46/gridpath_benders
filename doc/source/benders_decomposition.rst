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
created and instantiated before the loop. Share constraints between the operational and investment models
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

***************
File Map
***************

.. _extensions-section-ref:

Discrete Master Methods
==========================

Discrete Operational Methods
==========================

Infeasibility
==========================



.. automodule:: gridpath.auxiliary.module_list
    :members: all_modules_list
    Additional Benders modules are included here.
Investment and operational modules are separated in 

GridPath can be used in production-cost simulation or capacity-expansion mode
depending on whether "projects" of the "new" capacity types are included
in the model.

Production-Cost Simulation
==========================

Production-cost simulation models, also called unit-commitment and dispatch
models, simulate the operations of a specified power system with a high
level of fidelity -- at a high temporal resolution (e.g. hours to 5-minute
segments) and considering the detailed operating characteristics of
generators -- but generally over a fairly short period of time (e.g.
optimizing a year one day at a time). These models are adept at optimizing
the day-to-day operations of a fixed electric power system, provide
information on system reliability, assess transmission congestion, and
produce simulated locational marginal prices. They can also be used to
evaluate the impact of additions or retirements of capacity. As the number
of resources under considerations increases, however, so does the
number of possible combinations we need to simulate, making analysis using
production-cost simulation increasingly intensive and cumbersome. Answering
questions about how to develop the grid in the future as demand,
technologies, and policies change therefore requires additional types of
modeling capability.

Capacity-Expansion
==================

While production cost simulation models seek to optimize the operations of a
power system with a fixed set of resources specified by the user,
capacity-expansion models are designed to understand how the system should
evolve over time: they try to answer the question of what resources to
invest in among many options in order to meet system goals over time, i.e.
what grid infrastructure is most cost-effective while ensuring that the
system operates reliably and meeting policy targets.

The capacity expansion model minimizes the overall system cost over some
planning horizon, considering both capital costs (generators, transmission,
storage, any asset) and variable or operating costs subject to various
technical (e.g. generator limits, wind and solar availability, transmission
limits across corridors, hydro limits) and policy constraints (e.g.
renewable energy mandates, GHG targets).

Because capacity-expansion models have to optimize over several years or
decades, selecting generation, and transmission assets from many different
available options, the problem can get large quickly. In order to have
reasonable runtime, these models often simplify aspects of the electricity
grid, both in space and time. Spatially, most models will consider only
balancing areas or states as nodes (so all substations with the BA are
clubbed together). Temporally, only representative days and hours may be
used, and then given weights to represent a whole year e.g. one day per
month, and either 24 hours, or 6 time blocks (each representing 4 hours).
This simplification makes the linear optimization problem tractable. If the
spatial resolution is small, the temporal resolution may be increased, and
vice versa. An advantage of GridPath is that, unlike other similar
platforms, it leaves the decision for where to simplify and where to add
resolution to the user, making it possible to tailor the problem formulation
to the question at hand, the available computational resources, and the
available time.

After the system is “built” by a capacity-expansion problem, the system may
be simulated for the entire year (or years) using the production-cost
functionality to ensure that the decisions made using representative time
slices produce a system that can operate reliably at every time point of the
year. Capacity-expansion and production cost models are therefore
complementary. The former allows us to quickly explore many options for how
the power system ought to evolve over time and find the optimal solution;
the latter can help us ensure that the system we design does in fact perform
as we intended (e.g. that it serves load reliably and meets policy targets).

GridPath's architecture makes it possible for the same modules to be re-used
in production-cost or capacity-expansion modeling setting, allowing for a
seamless transition from one approach to the other, as datasets can be more
easily shared and reused.

Asset-Valuation
===============

GridPath can include market functionality and can optimize the dispatch of a resource
or a set of resources to maximize market revenue while meeting other system constraints.

Resource Adequacy
=================

GridPath can be used to evaluate the system's resource adequacy, e.g., by using
dispatch modeling over many load, renewable output, and generator and transmission
outage conditions.


Linear, Mixed-Integer, and Non-Linear Formulations
==================================================

Depending on how modules are combined, linear, mixed-integer, and non-linear
problem formulations are possible in GridPath. Some modules are
interchangeable, with the variable domain (e.g. binary vs. continuous with 0
to 1 bounds) the only difference in the final formulation.

.. _basic-functionality-section-ref:

*******************
Basic Functionality
*******************

GridPath can be used to create optimization problems with different features
and varying levels of complexity. For the basic model, the user must
define:

* :ref:`temporal-setup-section-ref` : The model's temporal setup (e.g. 365
  individual days at an hourly resolution),
* :ref:`geographic-setup-section-ref` : The geographic setup (e.g. a single
  load zone),
* :ref:`project-setup-section-ref` : The availability and operating
  characteristics of the generation infrastructure (e.g. an existing 500-MW coal
  plant with a specified heat rate),
* :ref:`objective-section-ref` : The desired objective (e.g. minimize
  cost).

.. _temporal-setup-section-ref:

Temporal Setup
==============

GridPath's temporal span and resolution are flexible: the user can decide on
a temporal setup by assigning appropriate weights to and relationships among
GridPath's temporal units.

The temporal units include:

.. _timepoints-sub-section-ref:

Timepoints
----------

.. automodule:: gridpath.temporal.operations.timepoints

See :ref:`subproblems-sub-section-ref` and
:ref:`stages-sub-section-ref` for more information.

.. _horizons-sub-section-ref:

Balancing Types and Horizons
----------------------------

.. automodule:: gridpath.temporal.operations.horizons

.. _periods-sub-section-ref:

Periods
-------

.. automodule:: gridpath.temporal.investment.periods

.. _subproblems-sub-section-ref:

Subproblems
-----------

In production-cost simulation, we often model operations during a full
year, e.g. at an hourly resolution (8760 *timepoints*). However, not all
*timepoints* are optimized together. Usually, each day or week of the year
is optimized separately. In GridPath, these separate optimizations are
called *subproblems*. Each *subproblem* can then contain different balancing
types and horizons. For example, if we are optimizing the year's operations
a week at a time, we would have 52 *subproblems*. Each one of those
*subproblems* could then have two *balancing types* -- a week and a day --
meaning each *timepoint* in the *subproblem* would belong to one of seven
'day' *horizons* and the single 'week' *horizon* (i.e. some resources would
have to be balanced on each day of the week and some over the entire week).

Within a production-cost simulation subproblem, some timepoints can be part
of a spin-up or look-ahead segment. These are additional timepoints that are
added to the subproblem's start and end but are ultimately discarded when
calculating result metrics. They are there to more faithfully model the
beginning and start of the subproblem. For example, if we had weekly
subproblems we could add 24 hourly spinup timepoins (1 day) before the week,
and 24 hourly lookahead timepoints (1 day) after the week, resulting in 52
subproblems of 9 days. After solving each subproblem independently, the 2
edge days in each subproblem are discarded .

Unlike in production-cost simulation, in capacity-expansion mode, we usually
have only a single subproblem (and no spinup or lookahead timepoints).

.. _stages-sub-section-ref:

Stages
------

GridPath also has multi-stage commitment functionality, i.e. commitment
decisions made for a subproblem can be fixed and then fed into a next stage
with some updated parameters (e.g. an updated load and renewable output
forecast). The number of stages is flexible and the timepoint resolution can
change from stage to stage.

For instance, the same day (subproblem) could first be modeled using a
day-ahead hourly forecast ("DA stage"), then with an hour-ahead hourly
forecast ("HA stage"), and finally with a real-time 5-minute forecast ("RT
stage"). Commitment decisions for e.g. coal generators could be fixed in the
DA stage, whereas commitment for e.g. gas turbines could be changed until
the final RT stage.

In practice, this is achieved by assigning a stage to each timepoint and
specifying a map between timepoints in each stage (see
:ref:`timepoints-sub-section-ref`). Load and renewable forecast inputs are
indexed by timepoint and stage, allowing for different forecasts (and,
optionally, timepoint resolutions) in each stage.

Examples
--------

A typical temporal setup for capacity-expansion modeling is shown below. Note
the presence op multiple investment periods, and a single subproblem and stage
(i.e. a single optimization).

.. image:: ../graphics/temporal_cap_exp.png


A typical temporal setup for production-cost modeling is shown below. Note the
presence of a single period, and multiple subproblems and stages (i.e. an
optimization will be solved for every subproblem-stage combination).

.. image:: ../graphics/temporal_prod_cost.png

.. _geographic-setup-section-ref:

Geographic Setup
================

Load Zones
----------

The main geographic unit in GridPath is the *load zone*. The load zone is
the level at which the load-balance constraints are enforced. In GridPath,
we can model a single load zone (copper plate) or multiple load zones, which
can be connected with transmission. This flexibility makes it possible to
apply to different regions with different geographic set-up or to take
different geographic approaches in modeling the same region (e.g. higher or
lower zonal resolution for the same region).

Other Geographic Units
----------------------

Optional levels of geographic resolution include *balancing areas* (BAs) for
operational and reliability reserve requirements and *policy zones* for
policy requirements. In GridPath, it is possible for generators in the same
load zone to contribute to different reserve balancing areas and/or policy
zones. Two possible configurations are shown below.

Geographic Configuration 1:

.. image:: ../graphics/lz_ba_rel1.png

Geographic Configuration 2:

.. image:: ../graphics/lz_ba_rel2.png

.. _project-setup-section-ref:

Projects
========

Generation, storage, and load-side resources in GridPath are called
*projects*. Each project is associated with a *load zone* whose load-balance
constraint it contributes to. In addition, each project must be assigned a
*capacity type*, an *availability type*, and an *operational type*. These
types are described in more detail below.

.. _project-capacity-type-section-ref:

Project Capacity Types
----------------------
Each project in GridPath must be assigned a *capacity type*. The *capacity
type* determines the capacity and the capacity-associated costs of
generation, storage, and demand-side infrastructure *projects* in the
optimization problem. The currently implemented capacity types include:

Specified Generation (*gen_spec*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.capacity.capacity_types.gen_spec

Specified Generation with Linear Economic Retirement (*gen_ret_lin*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.capacity.capacity_types.gen_ret_lin

Specified Generation with Binary Economic Retirement (*gen_ret_bin*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.capacity.capacity_types.gen_ret_bin

Linear New-Build Generation (*gen_new_lin*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.capacity.capacity_types.gen_new_lin

Binary New-Build Generation (*gen_new_bin*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.capacity.capacity_types.gen_new_bin

Specified Storage (*stor_spec*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.capacity.capacity_types.stor_spec

Linear New-Build Storage (*stor_new_lin*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.capacity.capacity_types.stor_new_lin

Binary New-Build Storage (*stor_new_bin*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.capacity.capacity_types.stor_new_bin

Specified Generation-Storage Hybrid (*gen_stor_hyb_spec*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.capacity.capacity_types.gen_stor_hyb_spec

Shiftable Load Supply Curve (*dr_new*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.capacity.capacity_types.dr_new

Specified Fuel Production Facilities (*fuel_prod_spec*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.capacity.capacity_types.fuel_prod_spec

Candidate Fuel Production Facilities (*fuel_prod_new*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.capacity.capacity_types.fuel_prod_new

.. _project-availability-type-section-ref:

Project Availability Types
--------------------------
Each *project* in GridPath must be assigned an *availability type* that
determines how much of a project's capacity is available to operate in each
*timepoint*. For example, some or all of a project's capacity may be
unavailable due to maintenance and other planned or unplanned outages. The
following *availability types* have been implemented.

Exogenous
^^^^^^^^^
.. automodule:: gridpath.project.availability.availability_types.exogenous

Binary
^^^^^^
.. automodule:: gridpath.project.availability.availability_types.binary

Continuous
^^^^^^^^^^
.. automodule:: gridpath.project.availability.availability_types.continuous

Project Operational Types
-------------------------
Each project in GridPath must be assigned an *operational type*. The
*operational_type* determines the operational capabilities of a project. The
currently implemented operational types include:

Simple Generation (*gen_simple*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.operations.operational_types.gen_simple

Must-Run Generation (*gen_must_run*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.operations.operational_types.gen_must_run

Always-On Generation (*gen_always_on*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.operations.operational_types.gen_always_on

Binary-Commit Generation (*gen_commit_bin*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.operations.operational_types.gen_commit_bin

Continuous-Commit Generation (*gen_commit_lin*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.operations.operational_types.gen_commit_lin

Capacity-Commit Generation (*gen_commit_cap*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.operations.operational_types.gen_commit_cap

Curtailable Hydro Generation (*gen_hydro*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.operations.operational_types.gen_hydro

Non-Curtailable Hydro Generation (*gen_hydro_must_take*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.operations.operational_types.gen_hydro_must_take

Curtailable Variable Generation (*gen_var*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.operations.operational_types.gen_var

Non-curtailable Variable Generation (*gen_var_must_take*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.operations.operational_types.gen_var_must_take

Storage (*stor*)
^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.operations.operational_types.stor

Curtailable Variable Generation-Storage Hybrid (*gen_var_stor_hyb*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.operations.operational_types.gen_var_stor_hyb

Flexible Load (*flex_load*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.operations.operational_types.flex_load

Note that a previous module (*dr*) for flexible loads is now deprecated.

Fuel Production (*fuel_prod*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.operations.operational_types.fuel_prod

Dispatchable Load (*dispatchable_load*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.project.operations.operational_types.dispatchable_load

.. _load-balance-section-ref:

Load Balance
============

.. automodule:: gridpath.system.load_balance.load_balance

.. _objective-section-ref:

Objective Function
==================

.. automodule:: gridpath.objective.max_npv

.. _optional-functionality-section-ref:

**********************
Optional Functionality
**********************

GridPath contains a number of modules that are optional. These modules are:

* :ref:`transmission-section-ref` : to model tranmission flows between multiple
  load zones,
* :ref:`operating-reserves-section-ref` : to model reserves such as frequency
  regulation or spinning reserves,
* :ref:`policy-section-ref` : to model policy measures such as a Renewable
  Portfolio Standard (RPS) or a carbon cap,
* :ref:`reliability-section-ref`: to model system reliability through a planning
  reserve marging (PRM) constraint, and
* :ref:`custom-modules-section-ref` : GridPath's modularity allows for easy
  addition of custom constraints specific to a power grid.

.. _transmission-section-ref:

Transmission
============
In GridPath, the user can include transmission line flows and transmission
topography by selecting the 'transmission' feature and specifying the
available transmission lines and which load zones they connect.

For each load zone and timepoint, the net flow on all transmission lines
connected to the load zone is aggregated and added as a production
component to the load balance constraint (see
:ref:`load-balance-section-ref`).

.. note:: If there is a net flow *out* of a load zone, the load-balance
    constraint 'production' component is a negative number.

Transmission features modules also add a transmission-capacity-costs
component and a transmission-operational-costs component to the objective
function (see :ref:`objective-section-ref`).

Like with GridPath 'projects,' transmission lines must be assigned a
capacity type, which determines their capacity and costs, and an operational
type, which determines their operational characteristics and costs.

The transmission network in GridPath can currently be modeled using a linear
transport model or using DC power flow (see Transmission Operational Types). In
either approach, resistive losses are assumed to be negligible.


Transmission Capacity Types
---------------------------
Each transmission line in GridPath must be assigned a *capacity type*. The
line's *capacity type* determines the available transmission capacity and the
capacity-associated costs. The currently implemented capacity types include:

Specified Transmission (*tx_spec*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.transmission.capacity.capacity_types.tx_spec

Linear New-Build Transmission (*tx_new_lin*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.transmission.capacity.capacity_types.tx_new_lin

Transmission Operational Types
------------------------------
Transmission lines in GridPath must be assigned an *operational type*. The
*operational type* determines the formulation of the operational
capabilities of the transmission line. The *operational types* currently
implemented include:

Linear Transport Transmission (*tx_simple*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.transmission.operations.operational_types.tx_simple

DC Power Flow (*tx_dcopf*)
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: gridpath.transmission.operations.operational_types.tx_dcopf

.. _operating-reserves-section-ref:

Operating Reserves
==================
GridPath can optionally model a range of operating reserve types, including
regulation up and down, spinning reserves, load-following up and down, inertia reserves and
frequency response. The implementation of each reserve type is standardized.
The user must define the reserve balancing areas along with any penalties
for violation of the reserve-balance constraints. For each balancing area,
the reserve requirement for each timepoint must be specified. Only
exogenously-specified reserves are implemented at this stage. Each project
that can provide the reserve type must then be assigned a balancing area to
whose reserve-balance constraint it can contribute. The project-level
reserve-provision variables are dynamically added to the project's operating
constraints if the project can provide each reserve type. Total reserve
provision by projects in each balancing area is then aggregated and
constrained to equal the BA's reserve requirement in each timepoint. The
user can optionally allow these constraints to be violated at a cost. Any
reserve-balance constraint violation penalty costs are added to the
objective function.

.. _reliability-section-ref:

Reliability
===========
GridPath can optionally model a planning-reserve margin requirement (PRM).
The user must the define the zones with a PRM requirement and the
requirement level for each PRM zone and period. Each project that can
contribute capacity (i.e. expected load-carrying capability -- ELCC --
greater than 0) must be assigned a PRM zone to whose reserve-balance
constraint it can contribute. The PRM reserve-balance constraint is a
period-level constraint. Projects can contribute a fraction of their
capacity as their ELCC via the *prm_simple* module. See
:ref:`custom-modules-section-ref`) for some advanced reliability functionality.

.. _policy-section-ref:

Markets
=======

GridPath can allow a resource or a set of resources to participate in markets with
pre-specified price streams, assuming resources are price-takers and subject to
market volume limits.

Policy
======

Energy Targets (e.g., Renewable Portfolio Standard)
---------------------------------------------------
GridPath can optionally impose targests for energy production by eligible
projects, e.g., renewable portfolio standard (RPS) requirements. The user must first
define the zones with an RPS requirement. The RPS requirement is a period- or
horizon-level constraint. Each RPS-eligible project must be assigned an RPS zone to
whose requirement it can contribute. The amount of RPS-eligible energy a project
contributes in each timepoint is determined by its operational type (e.g. a must-run
biomass plant will contribute its full capacity times the timepoint duration in every
timepoint while a wind project will contribute its capacity factor times its capacity).
The model aggregates all projects' contributions for each period and ensures
that the RPS requirement is met in each RPS zone and period.

Instantaneous Penetration
---------------------------------------------------
GridPath can optionally impose minimal and maximal constraints on the instantaneous
penetration of eligible projects. The user must first define the zones with
instantaneous penetration requirements. The instantaneous penetration requirement
is a timepoint constraint. Each instantaneous penetration-eligible project must be
assigned an instantaneous penetration zone, and the model aggregates all projects
contributions for each timepoint ensuring that the total penetration respects the
set minimum and maximum requirement. The minimum and maximum penetration requirements
are set as the sum of:

* Exogenously input power levels (MW) at a timepoint resolution.
* A percentage of the load at every timepoint.
* The capacity of other projects.
* The Power output of other projects.

As a default, all requirements are set at 0 except for the percentage of the load for
the maximum penetration which is set to 100%.


Carbon Cap or Carbon Cost
-------------------------
GridPath can optionally impose an carbon cap constraint. The user must
first define the zones with an emissions cap and the cap level by period (not
all periods must have a requirement). Each carbon-emitting project must be
assigned a *carbon cap zone* to whose emissions it can contribute. The amount
of carbon emissions from a project in each timepoint is determined by its
operational type and fuel. The model aggregates all projects' contributions
for each period and ensures that the total emissions stay below the cap in
each *carbon cap zone* and *period*.

GridPath can also optionally apply an emissions factor to energy imports
into an emissions zone. For the purpose, the relevant transmission lines
(i.e. transmission lines that connect the emissions zone to other zones)
must be assigned an emissions zone and an emissions intensity per unit
energy. These emissions are then added to the emissions cap constraint.

The emissions cap could be applied to carbon emissions or to other types of
emissions.

Alternatively, GridPath can model a cost on carbon emissions and optimize the amount
of emissions.

Fuel Use Limits
---------------

GridPath can be used to impose constraints on the amount of fuel available to
generators over certain time periods.

Performance Standard
--------------------

GridPath can optionally impose a performance standard requirement, i.e. limiting the
emissions intensity of energy produced by a set of projects.

.. _custom-modules-section-ref:

Custom Modules
==============
GridPath can include custom modules depending on the region or
system models. For example, for studies in the California
Integrated Resource Planning proceedings, GridPath includes
constraints on transmission simultaneous flow limits and advanced reliability
functionality such as:

* *ELCC surface module*: this module has substantial exogenous data
  requirements, but makes it possible to dynamically adjust the ELCC of some
  projects depending on the resource build-out (e.g. as more solar is built,
  the marginal ELCC becomes smaller)
* *Energy-only / partial deliverability*: ability to de-rate ELCC eligibility
  to less than the full project capacity (before applying the simple PRM
  fraction or the ELCC surface), since in some cases full deliverability may
  require additional costs to be incurred (e.g. for transmission, etc.)
* *Energy-limits*: additional limits on ELCC based on energy-limitations
  (e.g. for storage)

Similar custom functionality can be added for other systems and easily excluded when
not needed.
