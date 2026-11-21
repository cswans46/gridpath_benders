import pyomo.environ as pyo
import math
from pyomo.environ import (
    Var,
    ConstraintList,
    Objective,
    ObjectiveList,
    value,
    minimize,
    SolverFactory,
    NonNegativeReals,
    TerminationCondition,
    Reals,
    Constraint,
    Suffix
)
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.contrib.solver.common.util import NoFeasibleSolutionError
from gridpath.benders.gams_benders_solver import create_gams_solver
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import singledispatch
from typing import Any, Dict, List, NamedTuple
from gridpath.auxiliary.dynamic_components import DynamicComponents
from gridpath.benders.operational_model_creation import (
    create_operational_problem_benders
)
import os
import logging
import psutil
import gc
import threading
import tempfile
import shutil
class BendersState(NamedTuple):
    values: tuple

BENDERS_RECOURSE_OBJECTIVE_SCALE = 1e6

def _use_symbolic_labels(optimiser):
    """Numeric labels for Pyomo's direct MOSEK/Gurobi interfaces (see
    _solve_with_lp_optimizer); other solvers, e.g. GAMS, keep symbolic labels.
    Set GRIDPATH_BENDERS_SYMBOLIC_LABELS=1 to force symbolic labels."""
    solver_name = str(getattr(optimiser, "name", "")).lower()
    return not (
        solver_name.startswith(("mosek", "gurobi"))
        and os.getenv("GRIDPATH_BENDERS_SYMBOLIC_LABELS", "0") != "1"
    )


def _solve_with_lp_optimizer(
    instance,
    optimiser,
    optimizer_name,
    extra_options=None,
):
    """Solve once with a scoped MOSEK or Gurobi LP algorithm choice."""
    solver_name = str(getattr(optimiser, "name", "")).lower()
    if solver_name.startswith("mosek"):
        optimizer_key = "MSK_IPAR_OPTIMIZER"
        optimizer_values = {
            "primal_simplex": "MSK_OPTIMIZER_PRIMAL_SIMPLEX",
            "automatic": "MSK_OPTIMIZER_FREE",
            "interior_point": "MSK_OPTIMIZER_INTPNT",
        }
    elif solver_name.startswith("gurobi"):
        optimizer_key = "Method"
        optimizer_values = {
            "primal_simplex": 0,
            "automatic": -1,
            "interior_point": 2,
        }
    else:
        optimizer_key = None
        optimizer_values = {}

    scoped_options = ({
        optimizer_key: (
            optimizer_values[optimizer_name]
            if optimizer_name is not None
            else optimiser.options.get(
                optimizer_key, optimizer_values["automatic"]
            )
        )
    } if optimizer_key is not None else {})
    # Optional per-worker solver thread cap. With many resident workers the
    # default (all cores per solve) multiplies solver workspace memory.
    solver_threads = os.getenv("GRIDPATH_BENDERS_SOLVER_THREADS")
    if solver_threads:
        if solver_name.startswith("mosek"):
            scoped_options["MSK_IPAR_NUM_THREADS"] = int(solver_threads)
        elif solver_name.startswith("gurobi"):
            scoped_options["Threads"] = int(solver_threads)
    scoped_options.update(extra_options or {})

    # Pyomo's MOSEKDirect plugin passes every option through Task.putparam(),
    # whose Python binding requires the parameter value to be a string.
    # In particular, GRIDPATH_BENDERS_SOLVER_THREADS is parsed as an int
    # above, and caller-supplied extra_options may also contain numeric values.
    # Normalize only the MOSEK options; Gurobi expects native numeric types.
    if solver_name.startswith("mosek"):
        scoped_options = {
            key: str(option_value)
            for key, option_value in scoped_options.items()
        }

    # Pyomo's direct MOSEK/Gurobi interfaces build one Python label string per
    # row/column when symbolic labels are on (the MemoryError in
    # mosek_direct._add_constraints). Duals/solutions are mapped by object, not
    # by label, so numeric labels are safe for these two interfaces. Other
    # solvers (e.g. GAMS) keep symbolic labels.
    use_symbolic_labels = _use_symbolic_labels(optimiser)
    # Keep the complete pre-call option set so temporary normalization
    # does not leak into later solves.
    previous_options = dict(optimiser.options)
    try:
        optimiser.options.update(scoped_options)
        if solver_name.startswith("mosek"):
            # MOSEKDirect applies *all* entries in solver.options using
            # Task.putparam(name, value), whose Python binding requires str.
            # Normalize the full option dictionary, not just options added
            # above: user-supplied solver_options may contain ints/floats.
            for key, option_value in list(optimiser.options.items()):
                if not isinstance(option_value, str):
                    optimiser.options[key] = str(option_value)
        try:
            return optimiser.solve(
                instance,
                tee=False,
                symbolic_solver_labels=use_symbolic_labels,
            )
        except NoFeasibleSolutionError:
            logging.getLogger(__name__).warning(
                "Solver returned no loadable incumbent; classifying termination "
                "with solution loading disabled."
            )
            results = optimiser.solve(
                instance, tee=False, symbolic_solver_labels=use_symbolic_labels,
                load_solutions=False,
            )
            if results.solver.termination_condition == TerminationCondition.optimal:
                raise RuntimeError(
                    "Solver reported optimality without a loadable incumbent."
                )
            return results
    finally:
        optimiser.options.clear()
        optimiser.options.update(previous_options)

def _solve_scaled_recourse_lp(instance, optimiser, optimizer_name=None):
    """Solve an operational NPV LP with a normalized objective."""
    objective = instance.OP_NPV
    original_expression = objective.expr
    objective.set_value(
        original_expression / BENDERS_RECOURSE_OBJECTIVE_SCALE
    )
    try:
        results = _solve_with_lp_optimizer(
            instance,
            optimiser,
            optimizer_name,
        )
    finally:
        objective.set_value(original_expression)

    return results, BENDERS_RECOURSE_OBJECTIVE_SCALE

def _constraint_scale(constraint):
    scale = 1.0
    for bound in (constraint.lower, constraint.upper):
        bound_value = pyo.value(bound, exception=False)
        if bound_value is not None and math.isfinite(float(bound_value)):
            scale = max(scale, abs(float(bound_value)))

    repn = generate_standard_repn(constraint.body, compute_values=True)
    if repn.is_linear():
        magnitude = abs(float(pyo.value(repn.constant) or 0.0))
        for variable, coefficient in zip(repn.linear_vars, repn.linear_coefs):
            coefficient_value = pyo.value(coefficient, exception=False)
            if coefficient_value is None:
                continue
            variable_scale = 1.0
            for bound in (variable.lb, variable.ub):
                bound_value = pyo.value(bound, exception=False)
                if bound_value is not None and math.isfinite(float(bound_value)):
                    variable_scale = max(variable_scale, abs(float(bound_value)))
            magnitude += abs(float(coefficient_value)) * variable_scale
        scale = max(scale, magnitude)
    else:
        body_value = pyo.value(constraint.body, exception=False)
        if body_value is not None and math.isfinite(float(body_value)):
            scale = max(scale, abs(float(body_value)))

    return scale

def add_slacks_to_subproblem(model):
    """Add normalized elastic slacks to rows and continuous variable bounds."""
    if hasattr(model, "slack_max"):
        return

    constraints = list(
        model.component_data_objects(
            Constraint,
            active=True,
            descend_into=True,
        )
    )
    constraints = [
        constraint for constraint in constraints
        if any(
            bound is not None and math.isfinite(float(pyo.value(bound)))
            for bound in (constraint.lower, constraint.upper)
        )
    ]
    linking_constraints = getattr(model, "BendersLinkingConstraints", None)
    if linking_constraints is not None:
        constraints = [
            constraint
            for constraint in constraints
            if constraint.parent_component() is not linking_constraints
        ]
    equality_constraints = [constraint for constraint in constraints if constraint.equality]
    constraint_scales = {
        constraint.name: _constraint_scale(constraint)
        for constraint in constraints
    }

    model.slack_max = Var(domain=NonNegativeReals)
    model.slack_eq = Var(range(len(equality_constraints)), domain=Reals)
    model.FeasibilitySlackConstraints = ConstraintList()
    model.FeasibilityBoundConstraints = ConstraintList()

    linking_var_names = set(
        getattr(model, "BendersLinkingConstraintMap", {})
    )
    model._feasibility_bound_info = []
    for var in list(model.component_data_objects(
        pyo.Var,
        active=True,
        descend_into=True,
    )):
        if (
            var.parent_component() is model.slack_max
            or var.parent_component() is model.slack_eq
            or var.name in linking_var_names
            or var.fixed
            or var.is_binary()
            or var.is_integer()
        ):
            continue

        lower = pyo.value(var.lb, exception=False)
        upper = pyo.value(var.ub, exception=False)
        if lower is None and upper is None:
            continue

        lower = float(lower) if lower is not None else None
        upper = float(upper) if upper is not None else None
        scale = max(
            1.0,
            abs(lower) if lower is not None else 0.0,
            abs(upper) if upper is not None else 0.0,
        )
        info = {
            "var": var,
            "domain": var.domain,
            "lower": lower,
            "upper": upper,
            "scale": scale,
        }
        var.domain = Reals
        var.setlb(None)
        var.setub(None)
        info["lower_constraint"] = (
            model.FeasibilityBoundConstraints.add(
                var + scale * model.slack_max >= lower
            )
            if lower is not None
            else None
        )
        info["upper_constraint"] = (
            model.FeasibilityBoundConstraints.add(
                var - scale * model.slack_max <= upper
            )
            if upper is not None
            else None
        )
        model._feasibility_bound_info.append(info)

    equality_index = 0
    model._feasibility_slack_row_info = []
    for constraint in constraints:
        lower = (
            constraint.lower if constraint.lower is not None
            and math.isfinite(float(pyo.value(constraint.lower))) else None
        )
        upper = (
            constraint.upper if constraint.upper is not None
            and math.isfinite(float(pyo.value(constraint.upper))) else None
        )
        body = constraint.body
        scale = constraint_scales[constraint.name]
        row_info = {
            "constraint": constraint,
            "original_expression": constraint.expr,
            "body": body,
            "lower": pyo.value(lower, exception=False),
            "upper": pyo.value(upper, exception=False),
            "scale": scale,
            "equality": constraint.equality,
        }

        if constraint.equality:
            row_info["slack_var"] = model.slack_eq[equality_index]
            constraint.set_value(
            body - scale * model.slack_eq[equality_index] == lower
            )
            model.FeasibilitySlackConstraints.add(
                model.slack_eq[equality_index] <= model.slack_max
            )
            model.FeasibilitySlackConstraints.add(
                -model.slack_eq[equality_index] <= model.slack_max
            )
            equality_index += 1
        elif lower is not None and upper is not None:
            constraint.set_value(body - scale * model.slack_max <= upper)
            model.FeasibilitySlackConstraints.add(
                body + scale * model.slack_max >= lower
            )
        elif upper is not None:
            constraint.set_value(body <= upper + scale * model.slack_max)
        elif lower is not None:
            constraint.set_value(body >= lower - scale * model.slack_max)
        row_info["elastic_expression"] = constraint.expr
        model._feasibility_slack_row_info.append(row_info)

    model.slack_fix = Constraint(expr=model.slack_max == 0)

def fix_linking_variables(model, planning_sol, linking_vars):
    """
    Fix only linking variables that actually exist in the operational
    subproblem.

    Returns
    -------
    present_linking_vars : list[str]
        Linking variables from `linking_vars` that exist in this
        operational model.
    """

    if not hasattr(model, "BendersLinkingConstraints"):
        model.BendersLinkingConstraints = ConstraintList()

    else:
        model.BendersLinkingConstraints.clear()

    model.BendersLinkingConstraintMap = {}

    present_linking_vars = []

    for name in linking_vars:

        var = model.find_component(name)

        if var is None:
            # This is expected when a master linking variable does not
            # participate in this particular operational subproblem.
            continue

        if name not in planning_sol.values:
            raise KeyError(
                f"Operational subproblem contains linking variable "
                f"'{name}', but it is missing from planning_sol.values."
            )

        target = float(planning_sol.values[name])
        # Master round-off can overshoot a linking variable's own bound by
        # ~1e-12 relative, which makes the restoration LP infeasible.
        if var.lb is not None and target < pyo.value(var.lb):
            target = float(pyo.value(var.lb))
        if var.ub is not None and target > pyo.value(var.ub):
            target = float(pyo.value(var.ub))
        constraint = model.BendersLinkingConstraints.add(var == target)

        model.BendersLinkingConstraintMap[name] = constraint
        present_linking_vars.append(name)

    return present_linking_vars

def get_discrete_linking_vars(linking_vars_by_type):
    """
    Return all discrete linking variable names in a deterministic order.
    """
    return (
        linking_vars_by_type["binary"]
        + linking_vars_by_type["integer"]
    )

def get_master_discrete_state(
    instance,
    linking_variables_sub_by_type,
):
    """
    Return the current values of binary/integer master/linking
    variables.

    These are master variables x, not operational recourse variables z.
    """

    state = {
        "binary": {},
        "integer": {},
    }

    for name in linking_variables_sub_by_type["binary"]:
        var = instance.find_component(name)

        if var is None:
            continue

        if var.value is not None:
            state["binary"][name] = int(round(pyo.value(var)))

    for name in linking_variables_sub_by_type["integer"]:
        var = instance.find_component(name)

        if var is None:
            continue

        if var.value is not None:
            state["integer"][name] = int(round(pyo.value(var)))

    return state


def get_continuous_point(
    planning_sol,
    continuous_linking_vars,
):
    missing = [
        name
        for name in continuous_linking_vars
        if name not in planning_sol.values
    ]

    if missing:
        raise KeyError(
            "Continuous linking variables missing from planning "
            f"solution: {missing}"
        )

    return {
        name: float(planning_sol.values[name])
        for name in continuous_linking_vars
        if name in planning_sol.values
    }


def set_linking_variable_domains(
    instance,
    linking_vars_by_type,
    relaxed,
):
    """
    Temporarily relax discrete master/linking variables.

    Returns a dictionary containing their original domains.
    """

    original_domains = {}

    for name in (
        linking_vars_by_type["binary"]
        + linking_vars_by_type["integer"]
    ):

        var = instance.find_component(name)

        if var is None:
            continue

        original_domains[name] = var.domain

        if relaxed:

            if var.is_binary():
                var.domain = pyo.UnitInterval

            elif var.is_integer():
                var.domain = pyo.Reals

    return original_domains

import multiprocessing as mp
def _export_final_subproblem(
    instance,
    dynamic_components,
    optimiser,
    key,
    planning_sol,
    present_linking_vars,
    scenario_directory,
    multi_stage,
    parsed_arguments,
):
    """
    Solve one operational subproblem at the incumbent investment decisions
    and export its results like a regular (non-Benders) GridPath subproblem.

    Runs inside the persistent worker, which already holds the instance and
    its DynamicComponents. Output goes to
    <scenario>/<weather>/<hydro>/<availability>/<subproblem>/<stage>/results.

    Unlike the trial-point solves, operational integrality is NOT relaxed
    here, so the exported dispatch/commitment is a true (MIP) solution.
    Duals are only exported when the subproblem is a pure LP.

    Returns a small status dict (never raises, so the master cannot hang
    waiting for this worker).
    """
    import copy
    import datetime
    import sys
    import traceback

    # Imported here to avoid a circular import (run_scenario imports benders)
    from gridpath.common_functions import (
        Logging,
        create_logs_directory_if_not_exists,
    )
    from gridpath.run_scenario import save_results

    stdout_original = sys.stdout
    stderr_original = sys.stderr
    logger = None
    try:
        if parsed_arguments.log:
            logs_directory = create_logs_directory_if_not_exists(
                scenario_directory,
                key.weather,
                key.hydro,
                key.availability,
                key.subproblem,
                key.stage,
            )
            logger = Logging(
                logs_dir=logs_directory,
                start_time=datetime.datetime.now(),
                e2e=False,
                process_id=None,
            )
            sys.stdout = logger
            sys.stderr = logger

        if hasattr(instance, "slack_max"):
            _set_feasibility_restoration_mode(instance, active=False)

        # x = x* through the same equality rows used for the trial points
        fix_linking_variables(instance, planning_sol, present_linking_vars)

        is_mip = bool(
            detect_active_mip_variables(instance, linking_vars=present_linking_vars)
        )

        results = optimiser.solve(
            instance,
            tee=False,
            symbolic_solver_labels=True,
        )

        export_args = parsed_arguments
        if is_mip and not parsed_arguments.skip_duals:
            # No meaningful duals from a MIP solve
            export_args = copy.copy(parsed_arguments)
            export_args.skip_duals = True

        save_results(
            scenario_directory,
            key.weather,
            key.hydro,
            key.availability,
            str(key.subproblem),
            key.stage,
            multi_stage,
            instance,
            results,
            dynamic_components,
            export_args,
        )

        return {
            "status": str(results.solver.status),
            "termination_condition": str(results.solver.termination_condition),
            "is_mip": is_mip,
            "op_npv": (
                float(pyo.value(instance.OP_NPV))
                if results.solver.termination_condition
                == TerminationCondition.optimal
                else None
            ),
        }
    except Exception as e:  # noqa: BLE001 - report, don't kill the worker
        traceback.print_exc()
        return {"status": "ERROR", "error": repr(e)}
    finally:
        if logger is not None:
            logger.close()
            sys.stdout = stdout_original
            sys.stderr = stderr_original


class _StartupGuard:
    """Idempotent holder for the startup semaphore permit."""

    def __init__(self, semaphore):
        self.semaphore = semaphore
        self.held = False

    def acquire(self):
        if self.semaphore is not None:
            self.semaphore.acquire()
            self.held = True

    def release(self):
        if self.held:
            self.semaphore.release()
            self.held = False


def process_memory_rss():
    return psutil.Process(os.getpid()).memory_info().rss


def benders_subproblem_worker(
    key,
    planning_constraint_names,
    linking_vars,
    input_queue,
    output_queue,
    scenario_directory,
    multi_stage,
    parsed_arguments,
    solver_options=None,
    startup_semaphore=None,
    linking_var_bounds=None,
    calculate_global_recourse_bound=True,
    memory_coordinator=None,
    worker_index=None,
):
    """
    Entry point of a persistent Benders worker process.

    Holds `startup_semaphore` (if given) while the model is built and the LP
    upper bound is solved, and releases it before reporting READY, even if
    startup fails. That lets the parent start all workers at once while
    bounding how many build/solve at the same time.
    """
    guard = _StartupGuard(startup_semaphore)
    process = psutil.Process(os.getpid())
    initial_persistent = int(float(os.getenv('GRIDPATH_BENDERS_INITIAL_PERSISTENT_GB', '4')) * 1024**3)
    initial_temporary = int(float(os.getenv('GRIDPATH_BENDERS_INITIAL_TEMPORARY_GB', '4')) * 1024**3)
    safety_factor = float(os.getenv('GRIDPATH_BENDERS_MEMORY_SAFETY_FACTOR', '1.25'))
    if safety_factor < 1:
        raise ValueError('GRIDPATH_BENDERS_MEMORY_SAFETY_FACTOR must be at least 1.')
    try:
        guard.acquire()
        if memory_coordinator is not None:
            memory_coordinator.set_reservation(worker_index, persistent=initial_persistent, temporary=initial_temporary)
        _benders_subproblem_worker_impl(
            key,
            planning_constraint_names,
            linking_vars,
            input_queue,
            output_queue,
            scenario_directory,
            multi_stage,
            parsed_arguments,
            solver_options=solver_options,
            startup_guard=guard,
            linking_var_bounds=linking_var_bounds,
            calculate_global_recourse_bound=calculate_global_recourse_bound,
            memory_coordinator=memory_coordinator,
            worker_index=worker_index,
            safety_factor=safety_factor,
        )
    finally:
        guard.release()
        if memory_coordinator is not None:
            memory_coordinator.release(worker_index)




def _new_benders_worker_model(
    scenario_directory, key, dynamic_components, multi_stage,
    parsed_arguments, planning_constraint_names, linking_vars,
):
    """Reconstruct a disposable Pyomo model (Phase-I restoration, validation
    fallback, final export).

    A FRESH ``DynamicComponents`` is always used: ``create_abstract_model``
    appends to its lists without a guard and ``OP_NPV``/load balance sum over
    them, so reusing the worker's populated object would double-count every
    cost, revenue and load-balance term.  ``dynamic_components`` is accepted
    only for call compatibility and is ignored.

    Returns ``(model, linking_vars_by_type, present_linking_vars,
    fresh_dynamic_components)``.
    """
    fresh_dynamic_components = DynamicComponents()
    model = create_operational_problem_benders(
        scenario_directory=scenario_directory, key=key,
        dynamic_components=fresh_dynamic_components, multi_stage=multi_stage,
        parsed_arguments=parsed_arguments,
    )
    for name in get_shared_constraints(model, planning_constraint_names):
        model.find_component(name).deactivate()
    by_type = classify_linking_vars(model, linking_vars)
    present = get_present_linking_vars(by_type)
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.BendersLinkingConstraints = ConstraintList()
    model.BendersLinkingConstraintMap = {}
    return model, by_type, present, fresh_dynamic_components


def _new_benders_worker_solver(parsed_arguments, solver_options=None):
    """Create the same solver adapter used by the persistent worker."""
    if str(parsed_arguments.solver).lower() == "gams":
        return create_gams_solver(
            solver_options,
            executable=parsed_arguments.solver_executable,
        )
    kwargs = {}
    if parsed_arguments.solver_executable is not None:
        kwargs["executable"] = parsed_arguments.solver_executable
    solver = SolverFactory(parsed_arguments.solver, **kwargs)
    if solver_options and hasattr(solver, "options"):
        solver.options.update(solver_options)
    return solver


_MOSEK_OPTIMIZER_VALUES = {
    "primal_simplex": "MSK_OPTIMIZER_PRIMAL_SIMPLEX",
    "automatic": "MSK_OPTIMIZER_FREE",
    "interior_point": "MSK_OPTIMIZER_INTPNT",
}


def _start_rss_monitor():
    """Sample this process's RSS every 100 ms until the returned event is set."""
    start = process_memory_rss()
    peak = [start]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.1):
            try:
                peak[0] = max(peak[0], process_memory_rss())
            except psutil.Error:
                return

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    return start, peak, stop, thread


def _native_task_trial(
    runtime, message, linking_vars_by_type, present_linking_vars, parsed_arguments
):
    """Solve one Benders trial on the native MOSEK task.

    Mirrors the legacy Pyomo worker (``pj_relax`` phase, ``fixed_z`` MIP then
    LP, plain LP) stage by stage:

    * pj_relaxed master phase: every operational integer AND every discrete
      linking column is relaxed; lambdas for all present linking variables.
    * operational ``pj_relax`` (after master integrality is restored):
      operational integers relaxed, discrete linking columns relaxed for the
      duals, lambdas for continuous linking variables only.
    * operational ``fixed_z`` with operational integers: MIP at the trial
      point (discrete linking columns stay integer, they are pinned by the
      linking rows), z* rounded and fixed, then an LP with discrete linking
      relaxed for the duals.
    * ``fixed_z`` without operational integers: plain LP.

    The task is always released/reset in ``finally`` (``runtime.end_trial``).
    """
    planning_sol = message["planning_sol"]
    present = set(present_linking_vars)
    values = {}
    for name in present_linking_vars:
        if name not in planning_sol.values:
            raise KeyError("Missing master value for linking variable {!r}".format(name))
        values[name] = runtime.clip_linking_value(name, float(planning_sol.values[name]))

    master_method = message["master_discrete_method"]
    restored = bool(message.get("master_integrality_restored", False))
    pj_relaxed = master_method == "pj_relax" and not restored
    op_method = "pj_relax" if pj_relaxed else message["operational_discrete_method"]
    active = list(present_linking_vars) if pj_relaxed else list(linking_vars_by_type["continuous"])
    discrete_linking = [
        name
        for name in (linking_vars_by_type["binary"] + linking_vars_by_type["integer"])
        if name in runtime.integer_columns
    ]
    all_discrete_linking_names = linking_vars_by_type["binary"] + linking_vars_by_type["integer"]
    op_names = [name for name in runtime.integer_columns if name not in present]
    was_operational_mip = bool(op_names)

    base_optimizer = runtime.base_params.get("MSK_IPAR_OPTIMIZER", "MSK_OPTIMIZER_FREE")
    result_common = {
        "planning_sol": planning_sol,
        "active_linking_vars": active,
        "was_mip": was_operational_mip,
        "operational_discrete_method": op_method,
        "operational_mip_variables": list(op_names),
        "pj_relaxed": pj_relaxed,
        "operational_discrete_state": None,
    }

    runtime.open()
    try:
        runtime.set_linking_values(values)
        operational_state = None
        if op_method == "fixed_z" and was_operational_mip:
            runtime.set_stage_params({"MSK_IPAR_OPTIMIZER": base_optimizer})
            if not runtime.optimize(mixed_integer=True):
                return dict(result_common, native_infeasible=True)
            z_values = runtime.primal_values(op_names)
            operational_state = {"binary": {}, "integer": {}}
            rounded = {}
            for name, z in z_values.items():
                z_int = int(round(z))
                if abs(z - z_int) > 1e-6:
                    raise ValueError(
                        "Operational discrete variable {} has non-integral value {} "
                        "after the MIP solve.".format(name, z)
                    )
                rounded[name] = z_int
                kind = "binary" if runtime.integer_columns[name]["binary"] else "integer"
                operational_state[kind][name] = z_int
            runtime.set_integrality(op_names, False)
            runtime.fix_columns(rounded)
            lp_optimizer = base_optimizer
        elif op_method == "pj_relax":
            runtime.set_integrality(op_names, False)
            lp_optimizer = _MOSEK_OPTIMIZER_VALUES.get(
                str(parsed_arguments.benders_operational_solver), base_optimizer
            )
        else:
            lp_optimizer = base_optimizer
        # Discrete linking columns must be continuous for LP duals to exist,
        # in EVERY branch (including operational pj_relax outside the
        # relaxed master phase).
        runtime.set_integrality(discrete_linking, False)

        runtime.set_stage_params({"MSK_IPAR_OPTIMIZER": lp_optimizer})
        if not runtime.optimize(mixed_integer=False):
            if operational_state is not None:
                raise RuntimeError(
                    "Fixed-z native MOSEK LP did not solve to optimality after "
                    "an optimal operational MIP."
                )
            return dict(result_common, native_infeasible=True)

        # Native replacement for restoration.primal_violations(): worst
        # relative row/column violation straight from the solved task.
        validation_ok, worst_violation = runtime.primal_violation()
        lambdas_all = runtime.linking_duals()
        lambdas = {name: lambdas_all[name] for name in active}
        op_cost = -runtime.objective_value()
        discrete_cut_phase = bool(all_discrete_linking_names) and (
            (master_method == "pj_relax" and restored)
            or master_method in ("linearized", "djc")
        )
        master_state = None
        if discrete_cut_phase:
            master_state = {
                "binary": {n: int(round(values[n])) for n in linking_vars_by_type["binary"] if n in values},
                "integer": {n: int(round(values[n])) for n in linking_vars_by_type["integer"] if n in values},
            }
        return dict(
            result_common,
            native_infeasible=False,
            native_validation_ok=validation_ok,
            native_worst_violation=worst_violation,
            feasible=True,
            op_cost=float(op_cost),
            benders_lambda=lambdas,
            continuous_lambda=dict(lambdas),
            continuous_point={name: values[name] for name in active},
            theta_coeff=1.0,
            operational_discrete_state=operational_state,
            master_discrete_state=master_state,
            is_mip=was_operational_mip,
        )
    finally:
        runtime.end_trial()


def _pyomo_recourse_fallback(
    scenario_directory, key, multi_stage, parsed_arguments,
    planning_constraint_names, linking_vars, solver_options,
    native_result, solve_failed,
):
    """Legacy Pyomo path, reconstructed on demand (rare).

    Used when (a) the native task reports infeasibility (Phase-I restoration
    needs the Pyomo model and the certificate-checked ``restore``), or (b)
    the native primal check failed (re-solve through ``solve_original``).
    The model is rebuilt with a fresh ``DynamicComponents`` and discarded
    before returning.
    """
    from gridpath.benders.restoration import solve_original

    model, types, present, _dc = _new_benders_worker_model(
        scenario_directory, key, None, multi_stage, parsed_arguments,
        planning_constraint_names, linking_vars,
    )
    solver = _new_benders_worker_solver(parsed_arguments, solver_options)
    planning_sol = native_result["planning_sol"]
    fix_linking_variables(model, planning_sol, present)
    if native_result["pj_relaxed"]:
        relax_discrete_linking_variables(model, types)
    state = native_result.get("operational_discrete_state")
    if state is not None:
        for kind in ("binary", "integer"):
            for name, val in state[kind].items():
                model.find_component(name).fix(val)
    if native_result["operational_discrete_method"] == "pj_relax":
        prepare_lp_for_duals(
            model, present_linking_vars=present,
            operational_discrete_method="pj_relax",
        )
    elif state is not None or not native_result["was_mip"]:
        prepare_lp_for_duals(
            model, present_linking_vars=present,
            operational_discrete_method="fixed_z",
        )
    results = None
    dual_scale = 1.0
    if not solve_failed:
        retry = solve_original(model, solver)
        if retry is not None:
            results, dual_scale = retry
    if results is None:
        results = type("NativeTermination", (), {})()
        results.solver = type("SolverStatus", (), {})()
        results.solver.termination_condition = TerminationCondition.infeasible
    subop_sol = process_subproblem_solution(
        model, results, planning_sol, types, solver,
        active_linking_vars=native_result["active_linking_vars"],
        present_linking_vars=present, subproblem_key=key,
        dual_scale=dual_scale,
    )
    del model, solver
    gc.collect()
    return subop_sol


def _benders_subproblem_worker_impl(
    key,
    planning_constraint_names,
    linking_vars,
    input_queue,
    output_queue,
    scenario_directory,
    multi_stage,
    parsed_arguments,
    solver_options=None,
    startup_guard=None,
    linking_var_bounds=None,
    calculate_global_recourse_bound=True,
    memory_coordinator=None,
    worker_index=None,
    safety_factor=1.25,
):
    """Build and serve one persistent operational Benders subproblem.

    For each master trial point, the worker fixes linking variables through
    equality rows, solves recourse, and returns cost and linking-row duals.
    During the P&J relaxed master phase, all operational integer variables
    are LP-relaxed so the linking duals define globally valid cuts. After
    master integrality is restored, ``OperationalDiscreteMethod`` selects
    whether non-linking operational integers are relaxed or fixed at an
    optimal MIP state. Non-optimal recourse is sent through feasibility
    restoration before the worker returns a feasibility cut.
    """

    operation_temp_reservation = int(float(os.getenv('GRIDPATH_BENDERS_OPERATION_TEMPORARY_GB', '4')) * 1024**3)
    minimum_operation_temp = int(float(os.getenv('GRIDPATH_BENDERS_MIN_OPERATION_TEMPORARY_GB', '0.5')) * 1024**3)
    dynamic_components = DynamicComponents()
    # Measure this worker's own build (+ startup recourse-bound solve) peak so
    # later Pyomo rebuilds (export / restoration fallback) can reserve it.
    _build_start_rss, build_peak_rss, build_monitor_stop, build_monitor = _start_rss_monitor()

    # ================================================================
    # Build the operational model once.
    # ================================================================

    instance = create_operational_problem_benders(
        scenario_directory=scenario_directory,
        key=key,
        dynamic_components=dynamic_components,
        multi_stage=multi_stage,
        parsed_arguments=parsed_arguments,
    )

    shared_constraints = get_shared_constraints(
        instance,
        planning_constraint_names
    )

    for name in shared_constraints:
        instance.find_component(name).deactivate()

    # ================================================================
    # Solver
    # ================================================================

    if str(parsed_arguments.solver).lower() == "gams":
        optimiser = create_gams_solver(
            solver_options,
            executable=parsed_arguments.solver_executable,
        )
        if optimiser.backend_name.startswith("mosek"):
            optimiser.options["MSK_IPAR_LOG"] = "0"
    elif parsed_arguments.solver_executable is not None:
        optimiser = SolverFactory(
            parsed_arguments.solver,
            executable=parsed_arguments.solver_executable,
        )
    else:
        optimiser = SolverFactory(parsed_arguments.solver)

    if solver_options and hasattr(optimiser, "options"):
        optimiser.options.update(solver_options)

    # ================================================================
    # Classify linking variables.
    #
    # These are master variables x, including any binary/integer
    # linking variables.
    # ================================================================

    linking_vars_by_type = classify_linking_vars(
        instance,
        linking_vars,
    )

    present_linking_vars = get_present_linking_vars(
        linking_vars_by_type
    )

    # ================================================================
    # Global recourse upper bound
    # ================================================================

    recourse_ub = None

    if calculate_global_recourse_bound:
        recourse_ub = calculate_recourse_upper_bound(
            instance,
            optimiser,
            linking_var_names=present_linking_vars,
            linking_var_bounds=linking_var_bounds,
        )

    # ================================================================
    # Dual suffix.
    # ================================================================

    instance.dual = Suffix(direction=Suffix.IMPORT)

    # ================================================================
    # Benders linking constraints.
    #
    # These impose x = xbar but do NOT fix x directly, allowing
    # duals to be extracted from these constraints.
    # ================================================================

    instance.BendersLinkingConstraints = ConstraintList()

    instance.BendersLinkingConstraintMap = {}

    # ================================================================
    # Compile a native MOSEK task (once), persist it in the subproblem
    # folder and drop the Pyomo model.
    #
    # Memory modes (GRIDPATH_BENDERS_TASK_OFFLOAD, default 1):
    #   1 = offloaded: the Task is disposed right after compilation and
    #       re-read from <subproblem dir>/benders_task/subproblem.task for
    #       every trial, so an idle worker holds almost no model memory and
    #       the coordinator can admit more concurrent workers.  The model
    #       memory is only needed (as a *temporary* reservation) while a
    #       trial is loading/solving.
    #   0 = resident: the Task stays in RAM (faster, higher memory).
    # The Pyomo model is only ever rebuilt for rare paths: Phase-I
    # restoration after an infeasible trial, a failed native primal check,
    # and the final results export.
    # ================================================================
    native_task_runtime = None
    native_task_directory = None
    configured_solver_name = str(parsed_arguments.solver).lower()
    configured_backend_name = str(getattr(optimiser, "backend_name", "")).lower()
    native_mosek_requested = (
        configured_solver_name in ("mosek", "mosek_direct", "mosek_persistent")
        or (
            configured_backend_name.startswith("mosek")
            and os.getenv("GRIDPATH_BENDERS_NATIVE_TASK_GAMS", "0") == "1"
        )
    )
    if os.getenv("GRIDPATH_BENDERS_NATIVE_TASK", "1") == "1" and native_mosek_requested:
        from gridpath.benders.mosek_task import write_pyomo_mosek_task, MosekTaskRuntime

        try:
            task_compiler = SolverFactory("mosek_persistent")
        except Exception:
            task_compiler = SolverFactory("mosek_direct")
        # Parameters re-applied with putparam after every load / stage.
        native_params = {
            str(name): str(option)
            for name, option in getattr(optimiser, "options", {}).items()
            if str(name).startswith("MSK_")
        }
        if solver_options:
            native_params.update({
                str(name): str(option) for name, option in solver_options.items()
                if str(name).startswith("MSK_")
            })
        native_params.setdefault("MSK_IPAR_LOG", "0")
        solver_threads = os.getenv("GRIDPATH_BENDERS_SOLVER_THREADS")
        if solver_threads:
            native_params["MSK_IPAR_NUM_THREADS"] = str(int(solver_threads))

        # Task files live in the subproblem's own folder (D: drive), next to
        # its inputs and results.  GRIDPATH_BENDERS_TASK_DIR relocates the
        # root while keeping the same weather/hydro/availability/subproblem/
        # stage layout.
        task_root = os.getenv("GRIDPATH_BENDERS_TASK_DIR") or scenario_directory
        native_task_directory = os.path.join(
            task_root, str(key.weather), str(key.hydro), str(key.availability),
            str(key.subproblem), str(key.stage), "benders_task",
        )
        native_task_path = os.path.join(native_task_directory, "subproblem.task")

        # One permanent equality row per linking variable; only its RHS
        # changes between Benders iterations.
        row = None
        for name in present_linking_vars:
            row = instance.BendersLinkingConstraints.add(instance.find_component(name) == 0.0)
            instance.BendersLinkingConstraintMap[name] = row
        original_objective = instance.OP_NPV.expr
        instance.OP_NPV.set_value(original_objective / BENDERS_RECOURSE_OBJECTIVE_SCALE)
        try:
            task, metadata, native_env = write_pyomo_mosek_task(
                instance, task_compiler, native_task_path,
                instance.BendersLinkingConstraintMap,
                objective_scale=BENDERS_RECOURSE_OBJECTIVE_SCALE,
            )
        finally:
            instance.OP_NPV.set_value(original_objective)
        native_offload = os.getenv("GRIDPATH_BENDERS_TASK_OFFLOAD", "1") == "1"
        native_task_runtime = MosekTaskRuntime(
            native_task_path, metadata, env=native_env, params=native_params,
            task=task, offload=native_offload,
        )
        if native_offload:
            native_task_runtime.close()
        # The Pyomo model, solver and its index maps are no longer needed.
        del original_objective, task, instance, optimiser, task_compiler, row
        gc.collect()

    # ================================================================
    # Reconcile the startup estimate with the resident model footprint.
    # Free build-time garbage (abstract model, data portal) first so the
    # measured footprint is not inflated by unreachable objects.
    gc.collect()
    build_monitor_stop.set()
    build_monitor.join()
    build_temp_reservation = max(
        minimum_operation_temp,
        int(max(0, build_peak_rss[0] - process_memory_rss()) * safety_factor),
    )
    if memory_coordinator is not None:
        resident_bytes = process_memory_rss()
        # Release the startup temporary allowance before increasing the
        # persistent reservation; this avoids workers waiting on one another
        # while all still hold their startup peak allowances.
        memory_coordinator.set_reservation(worker_index, temporary=0)
        memory_coordinator.set_reservation(
            worker_index, persistent=int(resident_bytes * safety_factor)
        )

    # Tell the parent process that this worker is ready.
    # ================================================================

    if startup_guard is not None:
        startup_guard.release()

    output_queue.put(
        {
            "status": "READY",
            "linking_vars": linking_vars,
            "recourse_upper_bound": recourse_ub,
            "linking_vars_by_type": linking_vars_by_type,
            "present_linking_vars": present_linking_vars,
        }
    )

    # ================================================================
    # Persistent Benders worker loop.
    # ================================================================

    while True:

        message = input_queue.get()

        if message == "STOP":
            if native_task_runtime is not None:
                native_task_runtime.close()
            if native_task_directory is not None and os.getenv("GRIDPATH_BENDERS_KEEP_TASKS", "0") != "1":
                shutil.rmtree(native_task_directory, ignore_errors=True)
            break

        if native_task_runtime is not None and isinstance(message, dict) and message.get("command") == "EXPORT_FINAL":
            if memory_coordinator is not None:
                memory_coordinator.set_reservation(worker_index, temporary=max(operation_temp_reservation, build_temp_reservation))
            # Final export needs the full Pyomo model (it writes the standard
            # GridPath result files); build it once, with fresh dynamic
            # components, and discard it afterwards.
            export_model, _, export_present, export_dc = _new_benders_worker_model(
                scenario_directory, key, None, multi_stage,
                parsed_arguments, planning_constraint_names, linking_vars,
            )
            export_solver = _new_benders_worker_solver(parsed_arguments, solver_options)
            export_status = _export_final_subproblem(
                instance=export_model, dynamic_components=export_dc,
                optimiser=export_solver, key=key, planning_sol=message["planning_sol"],
                present_linking_vars=export_present, scenario_directory=scenario_directory,
                multi_stage=multi_stage, parsed_arguments=parsed_arguments,
            )
            del export_model, export_solver, export_dc
            gc.collect()
            output_queue.put((key, export_status))
            if memory_coordinator is not None:
                memory_coordinator.set_reservation(worker_index, temporary=0)
            continue

        # Final-solution export: solve this subproblem once at the incumbent
        # investment decisions (operational integrality intact) and write its
        # results in the standard GridPath directory layout. The master sends
        # this once, after convergence, instead of a trial point.
        if isinstance(message, dict) and message.get("command") == "EXPORT_FINAL":
            if memory_coordinator is not None:
                memory_coordinator.set_reservation(worker_index, temporary=operation_temp_reservation)
            output_queue.put(
                (
                    key,
                    _export_final_subproblem(
                        instance=instance,
                        dynamic_components=dynamic_components,
                        optimiser=optimiser,
                        key=key,
                        planning_sol=message["planning_sol"],
                        present_linking_vars=present_linking_vars,
                        scenario_directory=scenario_directory,
                        multi_stage=multi_stage,
                        parsed_arguments=parsed_arguments,
                    ),
                )
            )
            if memory_coordinator is not None:
                memory_coordinator.set_reservation(worker_index, temporary=0)
            continue

        if native_task_runtime is not None:
            if memory_coordinator is not None:
                memory_coordinator.set_reservation(worker_index, temporary=operation_temp_reservation)
            operation_start_rss, observed_peak_rss, monitor_stop, memory_monitor = _start_rss_monitor()
            try:
                native_result = _native_task_trial(
                    native_task_runtime, message, linking_vars_by_type,
                    present_linking_vars, parsed_arguments,
                )
                if native_result["native_infeasible"] or not native_result["native_validation_ok"]:
                    if not native_result["native_infeasible"]:
                        logging.getLogger(__name__).warning(
                            "Native primal check failed for %s (worst relative "
                            "violation %.3g); re-solving through the Pyomo path.",
                            key, native_result["native_worst_violation"],
                        )
                    # Rare path: rebuild the Pyomo model (fresh components) for
                    # Phase-I restoration / certificate-checked re-solve.
                    if memory_coordinator is not None:
                        memory_coordinator.set_reservation(
                            worker_index,
                            temporary=max(operation_temp_reservation, build_temp_reservation),
                        )
                    subop_sol = _pyomo_recourse_fallback(
                        scenario_directory, key, multi_stage, parsed_arguments,
                        planning_constraint_names, linking_vars, solver_options,
                        native_result, solve_failed=native_result["native_infeasible"],
                    )
                    discrete_cut_phase_native = bool(
                        linking_vars_by_type["binary"] or linking_vars_by_type["integer"]
                    ) and (
                        (message["master_discrete_method"] == "pj_relax"
                         and message.get("master_integrality_restored", False))
                        or message["master_discrete_method"] in ("linearized", "djc")
                    )
                    fallback_master_state = None
                    if discrete_cut_phase_native:
                        fallback_master_state = {
                            kind: {
                                name: int(round(float(native_result["planning_sol"].values[name])))
                                for name in linking_vars_by_type[kind]
                                if name in native_result["planning_sol"].values
                            }
                            for kind in ("binary", "integer")
                        }
                    subop_sol.update({
                        "operational_discrete_state": native_result["operational_discrete_state"],
                        "master_discrete_state": fallback_master_state,
                        "is_mip": native_result["was_mip"],
                        "operational_mip_variables": native_result["operational_mip_variables"],
                    })
                else:
                    subop_sol = {
                        key_: native_result[key_]
                        for key_ in (
                            "feasible", "op_cost", "benders_lambda", "continuous_lambda",
                            "continuous_point", "theta_coeff", "operational_discrete_state",
                            "master_discrete_state", "is_mip", "operational_mip_variables",
                        )
                    }
            finally:
                monitor_stop.set()
                memory_monitor.join()
            gc.collect()
            current_rss = process_memory_rss()
            # Adapt the temporary (load + solve + check) reservation to what
            # this worker actually needed above its idle footprint.
            operation_temp_reservation = max(
                minimum_operation_temp,
                int(max(0, observed_peak_rss[0] - current_rss) * safety_factor),
            )
            if os.getenv("GRIDPATH_BENDERS_MEMLOG", "0") == "1":
                print(
                    f"[benders-mem] native subproblem={key.subproblem} "
                    f"rss_start={operation_start_rss / 1024**3:.2f}GiB "
                    f"rss_peak={observed_peak_rss[0] / 1024**3:.2f}GiB "
                    f"rss_end={current_rss / 1024**3:.2f}GiB",
                    flush=True,
                )
            if memory_coordinator is not None:
                memory_coordinator.set_reservation(worker_index, temporary=0)
                memory_coordinator.set_reservation(
                    worker_index, persistent=int(current_rss * safety_factor)
                )
            output_queue.put((key, subop_sol))
            continue

        if memory_coordinator is not None:
            memory_coordinator.set_reservation(worker_index, temporary=operation_temp_reservation)

        # Sample RSS during the operation. The observed peak informs this
        # worker's next reservation, so estimates adapt independently.
        operation_start_rss = process_memory_rss()
        observed_peak_rss = [operation_start_rss]
        monitor_stop = threading.Event()
        def sample_operation_memory():
            while not monitor_stop.wait(0.1):
                try:
                    observed_peak_rss[0] = max(observed_peak_rss[0], process_memory_rss())
                except psutil.Error:
                    return
        memory_monitor = threading.Thread(target=sample_operation_memory, daemon=True)
        memory_monitor.start()

        if hasattr(instance, "slack_max"):
            _set_feasibility_restoration_mode(
                instance,
                active=False,
            )

        # ============================================================
        # Current master solution.
        # ============================================================

        planning_sol = message["planning_sol"]

        master_discrete_method = message[
            "master_discrete_method"
        ]

        configured_operational_discrete_method = message[
            "operational_discrete_method"
        ]

        # ============================================================
        # Determine which master/linking phase we are in.
        #
        # IMPORTANT:
        # This concerns x only. It has nothing to do with whether
        # operational recourse is solved using pj_relax or fixed_z.
        # ============================================================

        has_discrete_linking_vars = bool(
            linking_vars_by_type["binary"]
            or linking_vars_by_type["integer"]
        )

        master_integrality_restored = message.get(
            "master_integrality_restored",
            False,
        )

        pj_relaxed_phase = (
            master_discrete_method == "pj_relax"
            and not master_integrality_restored
        )

        pj_discrete_phase = (
            master_discrete_method == "pj_relax"
            and has_discrete_linking_vars
            and master_integrality_restored
        )

        linearized_phase = (
            master_discrete_method == "linearized"
            and has_discrete_linking_vars
        )

        djc_phase = (
            master_discrete_method == "djc"
            and has_discrete_linking_vars
        )

        discrete_cut_phase = (
            pj_discrete_phase
            or linearized_phase
            or djc_phase
        )

        # Classical dual cuts are globally valid in the relaxed P&J phase
        # only for the LP relaxation of operational recourse. The configured
        # integer-recourse method is used again once master integrality is
        # restored and state-conditioned cuts are generated.
        operational_discrete_method = (
            "pj_relax"
            if pj_relaxed_phase
            else configured_operational_discrete_method
        )

        # ============================================================
        # Preserve original linking-variable domains.
        #
        # During the P&J relaxation phase, all linking variables,
        # including binary/integer linking variables, must be
        # continuous so that duals can be extracted.
        # ============================================================

        original_linking_domains = {}

        if pj_relaxed_phase:

            original_linking_domains = (
                relax_discrete_linking_variables(
                    instance,
                    linking_vars_by_type,
                )
            )

            active_linking_vars = present_linking_vars

        else:

            # Once master integrality is active, only continuous
            # linking variables participate in the classical dual
            # term of the Benders cut.
            active_linking_vars = (
                linking_vars_by_type["continuous"]
            )

        # ============================================================
        # Impose x = xbar.
        #
        # Do NOT fix the linking variables directly.
        # ============================================================

        fix_linking_variables(
            instance,
            planning_sol,
            present_linking_vars,
        )

        # ================================================================
        # Identify genuine operational MIP variables BEFORE fixed_z.
        #
        # This must happen before z is fixed. Once z is fixed, Pyomo will
        # no longer identify those variables as active MIP variables.
        # ================================================================

        initial_mip_variables = detect_active_mip_variables(
            instance,
            linking_vars=present_linking_vars,
        )

        was_operational_mip = bool(
            initial_mip_variables
        )

        # ================================================================
        # Operational discrete state.
        # ================================================================

        operational_discrete_state = None
        operational_discrete_fix_info = []
        lp_changes = []
        lp_dual_scale = 1.0

        # ================================================================
        # Solve operational problem.
        # ================================================================

        if (
                operational_discrete_method == "fixed_z"
                and was_operational_mip
        ):

            (
                mip_results,
                operational_discrete_state,
                operational_discrete_fix_info,
            ) = solve_operational_mip_and_fix_z(
                instance,
                optimiser,
                present_linking_vars=present_linking_vars,
            )

        elif operational_discrete_method == "pj_relax":

            lp_changes = prepare_lp_for_duals(
                instance,
                present_linking_vars=present_linking_vars,
                operational_discrete_method="pj_relax",
            )

            mip_results, lp_dual_scale = _solve_scaled_recourse_lp(
                instance,
                optimiser,
                optimizer_name=parsed_arguments.benders_operational_solver,
            )

        else:
            # Use fixed_z here for the case with an initial LP
            # Because this collapses to teh same LP-dual workflow
            lp_changes = prepare_lp_for_duals(
                instance,
                present_linking_vars=present_linking_vars,
                operational_discrete_method="fixed_z",
            )

            mip_results = optimiser.solve(
                instance,
                tee=False,
                symbolic_solver_labels=_use_symbolic_labels(optimiser),
            )

        mip_termination = (
            mip_results.solver.termination_condition
        )

        if mip_termination == TerminationCondition.optimal:

            # ============================================================
            # fixed_z: now convert the solved MIP into the LP used
            # for dual extraction.
            # ============================================================

            if (
                    operational_discrete_method == "fixed_z"
                    and was_operational_mip
            ):

                lp_changes = prepare_lp_for_duals(
                    instance,
                    present_linking_vars=present_linking_vars,
                    operational_discrete_method="fixed_z",
                )

                # Safety check.
                for var in instance.component_data_objects(
                        pyo.Var,
                        active=True,
                        descend_into=True,
                ):

                    if var.name in present_linking_vars:
                        continue

                    if var.is_binary() or var.is_integer():

                        if not var.fixed:

                            restore_lp_after_duals(
                                instance,
                                lp_changes,
                            )

                            if operational_discrete_fix_info:
                                restore_operational_discrete_variables(
                                    operational_discrete_fix_info
                                )

                            raise RuntimeError(
                                "Operational discrete variable "
                                f"'{var.name}' is not fixed during "
                                "fixed-z LP dual extraction."
                            )

                # Solve LP exactly once.
                lp_results, lp_dual_scale = (
                    _solve_scaled_recourse_lp(
                        instance,
                        optimiser,
                    )
                )

                lp_termination = (
                    lp_results.solver.termination_condition
                )

                if lp_termination != TerminationCondition.optimal:

                    restore_lp_after_duals(
                        instance,
                        lp_changes,
                    )

                    if operational_discrete_fix_info:
                        restore_operational_discrete_variables(
                            operational_discrete_fix_info
                        )

                    raise RuntimeError(
                        "Fixed-z LP did not solve to optimality for "
                        f"subproblem {key}. "
                        f"Termination condition: {lp_termination}"
                    )

            else:

                # The operational LP was already solved above.
                lp_results = mip_results

            # ============================================================
            # Extract Benders dual information using your existing
            # process_subproblem_solution().
            #
            # IMPORTANT:
            # active_linking_vars contains x, not z.
            # ============================================================

            subop_sol = process_subproblem_solution(
                instance,
                lp_results,
                planning_sol,
                linking_vars_by_type,
                optimiser,
                active_linking_vars=active_linking_vars,
                present_linking_vars=present_linking_vars,
                subproblem_key=key,
                dual_scale=lp_dual_scale,
            )

            # ============================================================
            # Store operational z state separately from master x state.
            # ============================================================

            subop_sol[
                "operational_discrete_state"
            ] = operational_discrete_state

            if discrete_cut_phase:

                subop_sol[
                    "master_discrete_state"
                ] = get_master_discrete_state(
                    instance,
                    linking_vars_by_type,
                )

            else:

                subop_sol[
                    "master_discrete_state"
                ] = None

            # ============================================================
            # Preserve the fact that this was originally an operational
            # MIP.
            #
            # DO NOT call detect_active_mip_variables() here because fixed_z
            # has already fixed z.
            # ============================================================

            subop_sol[
                "is_mip"
            ] = was_operational_mip

            subop_sol[
                "operational_mip_variables"
            ] = [
                var.name
                for var in initial_mip_variables
            ]

            # ============================================================
            # Restore LP modifications.
            # ============================================================

            restore_lp_after_duals(
                instance,
                lp_changes,
            )

            # ============================================================
            # CRITICAL:
            #
            # Undo z = z* before the next Benders iteration.
            #
            # Without this, the persistent worker would solve every future
            # operational MIP with the old z*.
            # ============================================================

            if operational_discrete_fix_info:
                restore_operational_discrete_variables(
                    operational_discrete_fix_info
                )

        # ============================================================
        # ============================================================
        # INFEASIBLE / NON-OPTIMAL OPERATIONAL SOLUTION
        # ============================================================
        # ============================================================

        else:

            # --------------------------------------------------------
            # There is no valid operational z* when the operational
            # MIP is infeasible, so fixed_z must NOT attempt to create
            # an operational discrete state.
            # --------------------------------------------------------

            operational_discrete_state = None

            # --------------------------------------------------------
            # Existing feasibility-restoration logic.
            #
            # This assumes your current process_subproblem_solution()
            # handles the feasibility LP exactly as in your existing
            # implementation.
            # --------------------------------------------------------

            subop_sol = process_subproblem_solution(
                instance,
                mip_results,
                planning_sol,
                linking_vars_by_type,
                optimiser,
                active_linking_vars=active_linking_vars,
                present_linking_vars=present_linking_vars,
                subproblem_key=key,
            )

            subop_sol[
                "operational_discrete_state"
            ] = None

            if discrete_cut_phase:

                subop_sol[
                    "master_discrete_state"
                ] = get_master_discrete_state(
                    instance,
                    linking_vars_by_type,
                )

            else:

                subop_sol[
                    "master_discrete_state"
                ] = None

            subop_sol["is_mip"] = was_operational_mip

            subop_sol[
                "operational_mip_variables"
            ] = [
                var.name
                for var in initial_mip_variables
            ]

            # --------------------------------------------------------
            # Defensive restoration in case anything was fixed before
            # the failure was detected.
            # --------------------------------------------------------

            if operational_discrete_fix_info:

                restore_operational_discrete_variables(
                    operational_discrete_fix_info
                )

        # ============================================================
        # Restore linking-variable domains.
        # ============================================================

        if original_linking_domains:

            restore_variable_domains(
                instance,
                original_linking_domains,
            )

        # Only the linking-row duals are used (already extracted above);
        # drop the duals Pyomo imported for every other row.
        if hasattr(instance, "dual"):
            instance.dual.clear()

        monitor_stop.set()
        memory_monitor.join()
        current_rss = process_memory_rss()
        persistent_reservation = int(current_rss * safety_factor)
        transient_peak = max(0, observed_peak_rss[0] - current_rss)
        if os.getenv("GRIDPATH_BENDERS_MEMLOG", "0") == "1":
            print(
                f"[benders-mem] subproblem={key.subproblem} "
                f"rss_start={operation_start_rss / 1024**3:.2f}GiB "
                f"rss_peak={observed_peak_rss[0] / 1024**3:.2f}GiB "
                f"rss_end={current_rss / 1024**3:.2f}GiB",
                flush=True,
            )
        operation_temp_reservation = max(
            minimum_operation_temp, int(transient_peak * safety_factor)
        )
        if memory_coordinator is not None:
            # Release the temporary reservation first so a larger persistent
            # footprint can be admitted without a circular wait between workers.
            memory_coordinator.set_reservation(worker_index, temporary=0)
            memory_coordinator.set_reservation(
                worker_index, persistent=persistent_reservation
            )

        # Send result back only after the updated reservations are admitted.
        output_queue.put(
            (
                key,
                subop_sol,
            )
        )

def relax_discrete_linking_variables(
    instance,
    linking_vars_by_type,
):
    """
    Relax binary/integer master/linking variables to continuous
    domains for the P&J relaxed master phase.

    Returns the original domains so they can be restored later.
    """

    return set_linking_variable_domains(
        instance,
        linking_vars_by_type,
        relaxed=True,
    )


def restore_variable_domains(
    instance,
    original_domains,
):
    """
    Restore the domains of master/linking variables.
    """

    for name, original_domain in original_domains.items():

        var = instance.find_component(name)

        if var is not None:
            var.domain = original_domain

def _restore_feasibility_variable_bounds(model, elastic):
    for info in getattr(model, "_feasibility_bound_info", []):
        var = info["var"]
        lower = info["lower"]
        upper = info["upper"]
        if elastic:
            var.domain = Reals
            var.setlb(None)
            var.setub(None)
        else:
            var.domain = info["domain"]
            var.setlb(lower)
            var.setub(upper)
        if info["lower_constraint"] is not None:
            info["lower_constraint"].set_value(
                var + info["scale"] * model.slack_max >= lower
            )
            if elastic:
                info["lower_constraint"].activate()
            else:
                info["lower_constraint"].deactivate()
        if info["upper_constraint"] is not None:
            info["upper_constraint"].set_value(
                var - info["scale"] * model.slack_max <= upper
            )
            if elastic:
                info["upper_constraint"].activate()
            else:
                info["upper_constraint"].deactivate()

def _set_feasibility_restoration_mode(model, active):
    """Toggle elastic rows/bound rows without leaving them in recourse LPs."""
    row_info = getattr(model, "_feasibility_slack_row_info", None)
    if row_info is None:
        return

    if active:
        for info in row_info:
            info["constraint"].set_value(info["elastic_expression"])
        _restore_feasibility_variable_bounds(model, elastic=True)
        model.slack_fix.deactivate()
    else:
        for info in row_info:
            info["constraint"].set_value(info["original_expression"])
        _restore_feasibility_variable_bounds(model, elastic=False)
        model.slack_fix.set_value(model.slack_max == 0)
        model.slack_fix.activate()

def process_subproblem_solution(
    instance,
    mip_results,
    planning_sol,
    linking_vars_by_type,
    optimiser,
    active_linking_vars=None,
    present_linking_vars=None,
    subproblem_key=None,
    dual_scale=1.0,
):
    """Return recourse cost and linking duals for one master trial point.

    An optimal recourse solve returns the operating cost and duals of the
    linking equalities. A non-optimal solve first minimizes normalized
    feasibility slack. Positive slack yields a feasibility cut from the
    restoration LP duals; near-zero slack triggers an exact-feasibility
    retry before the point is treated as feasible. ``active_linking_vars``
    must contain only linking variables present in this subproblem.
    """

    if active_linking_vars is None:
        active_linking_vars = (
            linking_vars_by_type["continuous"]
        )

    active_linking_vars = list(active_linking_vars)
    missing_linking_values = [
        name for name in active_linking_vars if name not in planning_sol.values
    ]
    if missing_linking_values:
        raise KeyError(
            "Active linking variables missing from planning solution: "
            f"{missing_linking_values}"
        )
    continuous_point = {
        name: float(planning_sol.values[name])
        for name in active_linking_vars
    }

    mip_term = (
        mip_results.solver.termination_condition
    )
    from gridpath.benders.restoration import primal_violations, solve_original

    if mip_term == TerminationCondition.optimal:
        violations = primal_violations(instance)
        if violations:
            logging.getLogger(__name__).warning(
                "Operational solution failed original-model validation for %s: %s",
                subproblem_key, violations[:5],
            )
            retry = solve_original(instance, optimiser)
            if retry is None:
                mip_term = TerminationCondition.infeasible
            else:
                mip_results, dual_scale = retry

    # =============================================================
    # Feasible / optimal operational problem
    # =============================================================

    if mip_term == TerminationCondition.optimal:

        op_cost = -value(instance.OP_NPV)

        if os.environ.get("GRIDPATH_BENDERS_CREDIT_DIAG") and hasattr(
            instance, "Project_Carbon_Credits_Generated"
        ):
            import logging as _lg

            _gen = {}
            for (_p, _prd) in instance.Project_Carbon_Credits_Generated:
                _em = sum(
                    value(instance.Project_Carbon_Emissions[_q, _t])
                    * value(instance.hrs_in_tmp[_t])
                    * value(instance.tmp_weight[_t])
                    for (_q, _t) in instance.CARBON_CREDITS_GENERATION_PRJ_OPR_TMPS
                    if _q == _p and instance.period[_t] == _prd
                )
                _gen[(_p, _prd)] = (
                    round(value(instance.Project_Carbon_Credits_Generated[_p, _prd])),
                    round(_em),
                )
            _lg.warning(
                "CREDIT DIAG sub=%s op_cost=%.6g gen(prj,prd)->(generated,"
                "emissions)=%s sell=%s",
                getattr(instance, "_benders_subproblem_id", "?"),
                op_cost,
                _gen,
                {
                    k: round(value(v))
                    for k, v in instance.Sell_Carbon_Credits.items()
                },
            )

        benders_lambda = {}

        for name in active_linking_vars:

            constraint = (
                instance.BendersLinkingConstraintMap.get(name)
            )

            if constraint is None:

                # This should now be impossible if the worker is
                # passing only present_linking_vars.
                var = instance.find_component(name)

                if var is None:
                    raise RuntimeError(
                        "Internal Benders bookkeeping error: "
                        f"linking variable '{name}' was passed as "
                        "active but does not exist in the "
                        "operational model."
                    )

                raise RuntimeError(
                    "Internal Benders bookkeeping error: "
                    f"linking variable '{name}' exists in the "
                    "operational model but has no corresponding "
                    "Benders linking constraint."
                )

            dual = instance.dual.get(
                constraint,
                None,
            )

            if dual is None:

                raise RuntimeError(
                    "No dual value was returned for Benders "
                    f"linking constraint associated with '{name}'."
                )

            benders_lambda[name] = -float(dual) * float(dual_scale)
            
        logging.info("SOC lambdas %s: %s", subproblem_key,
             {k: v for k, v in benders_lambda.items() if "SOC_ALLOCATION" in k})

        continuous_lambda = {
            name: benders_lambda[name]
            for name in active_linking_vars
            if name in benders_lambda
        }

        return {
            "feasible": True,
            "op_cost": float(op_cost),
            "benders_lambda": benders_lambda,
            "continuous_lambda": continuous_lambda,
            "continuous_point": continuous_point,
            "theta_coeff": 1.0,
        }

    # =============================================================
    # Infeasible operational problem
    # =============================================================

    present_linking_var_set = set(present_linking_vars or [])
    nonlinking_integer_vars = [
        var.name
        for var in instance.component_data_objects(
            pyo.Var,
            active=True,
            descend_into=True,
        )
        if (
            var.name not in present_linking_var_set
            and not var.fixed
            and (var.is_binary() or var.is_integer())
        )
    ]
    if nonlinking_integer_vars:
        raise NotImplementedError(
            "LP-dual feasibility cuts are not valid for infeasible "
            "operational integer recourse. Use a logic-based or "
            "combinatorial feasibility cut for variables: "
            f"{nonlinking_integer_vars[:20]}"
        )

    from gridpath.benders.restoration import restore

    restoration_result = restore(
        instance, optimiser, active_linking_vars, continuous_point, subproblem_key
    )
    if isinstance(restoration_result, tuple):
        retry_results, retry_dual_scale = restoration_result
        return process_subproblem_solution(
            instance, retry_results, planning_sol, linking_vars_by_type, optimiser,
            active_linking_vars=active_linking_vars,
            present_linking_vars=present_linking_vars,
            subproblem_key=subproblem_key, dual_scale=retry_dual_scale,
        )
    return restoration_result

def relax_operational_integrality(model):
    for var in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if var.fixed:
            continue
        if var.is_binary() or var.is_integer():
            lb, ub = var.bounds        # effective bounds, including domain bounds
            var.domain = pyo.Reals
            var.setlb(lb)
            var.setub(ub)


def apply_linking_variable_bounds(lp_instance, linking_var_names, linking_var_bounds):
    linking_var_bounds = linking_var_bounds or {}
    still_unbounded = []
    for name in linking_var_names or []:
        var = lp_instance.find_component(name)
        if var is None or var.fixed:
            continue
        master_bounds = linking_var_bounds.get(name)
        if master_bounds is not None:
            lb, ub = master_bounds
            if var.lb is None and lb is not None:
                var.setlb(lb)
            if var.ub is None and ub is not None:
                var.setub(ub)
        if var.lb is None or var.ub is None:
            still_unbounded.append(name)
    return still_unbounded


def calculate_recourse_upper_bound(
    instance, optimiser, linking_var_names=None, linking_var_bounds=None,
):
    """LP-relaxation bound on recourse value.

    Relaxes integrality and applies linking-variable bounds IN PLACE and
    restores them afterwards, instead of solving on ``instance.clone()``
    (a full second copy of the model in every worker). Set
    GRIDPATH_BENDERS_BOUND_VIA_CLONE=1 to use the old clone-based behaviour.
    """
    if os.getenv("GRIDPATH_BENDERS_BOUND_VIA_CLONE", "0") == "1":
        lp_instance = instance.clone()
        restore_state = None
    else:
        lp_instance = instance
        restore_state = _snapshot_for_bound_lp(instance, linking_var_names)

    try:
        relax_operational_integrality(lp_instance)

        still_unbounded = apply_linking_variable_bounds(
            lp_instance, linking_var_names, linking_var_bounds
        )
        if still_unbounded:
            logging.getLogger(__name__).warning(
                "Recourse bound LP: %d linking variable(s) still lack a bound, "
                "e.g. %s",
                len(still_unbounded), still_unbounded[:10],
            )

        results, _ = _solve_scaled_recourse_lp(lp_instance, optimiser)
        termination = results.solver.termination_condition

        if termination != TerminationCondition.optimal:
            raise RuntimeError(
                "LP relaxation for recourse lower bound did not solve "
                f"to optimality. Termination condition: {termination}. "
                f"{len(still_unbounded)} linking variable(s) lacked a lower "
                f"or upper bound (first 10: {still_unbounded[:10]})."
            )

        bound = value(lp_instance.OP_NPV)
    finally:
        if restore_state is not None:
            _restore_after_bound_lp(restore_state)

    return bound


def _snapshot_for_bound_lp(instance, linking_var_names):
    """Record what relax_operational_integrality / apply_linking_variable_bounds
    will change so it can be undone."""
    discrete = []
    for var in instance.component_data_objects(
        pyo.Var, active=True, descend_into=True
    ):
        if not var.fixed and (var.is_binary() or var.is_integer()):
            discrete.append((var, var.domain, var.lb, var.ub))
    linking = []
    for name in linking_var_names or []:
        var = instance.find_component(name)
        if var is not None and not var.fixed:
            linking.append((var, var.lb, var.ub))
    return discrete, linking


def _restore_after_bound_lp(state):
    discrete, linking = state
    for var, domain, lb, ub in discrete:
        var.domain = domain
        var.setlb(lb)
        var.setub(ub)
    for var, lb, ub in linking:
        var.setlb(lb)
        var.setub(ub)


def solve_operational_mip_and_fix_z(
    operational_model,
    solver,
    present_linking_vars=None,
):
    """
    Solve the actual operational MIP at the current master solution.

    The resulting operational discrete variables z* are then fixed
    at their optimal values.

    IMPORTANT:
        Master/linking binary/integer variables are excluded. Those
        variables are x and are handled separately through the
        Benders linking constraints.

    Returns
    -------
    results :
        Results from the operational MIP solve.

    operational_discrete_state :
        Dictionary containing the optimal z* values.

    fix_info :
        Information required to restore the operational discrete
        variables to their pre-MIP state after dual extraction.
    """

    if present_linking_vars is None:
        present_linking_vars = set()
    else:
        present_linking_vars = set(present_linking_vars)

    # ================================================================
    # Identify genuine operational discrete variables.
    # ================================================================

    operational_discrete_vars = []

    for var in operational_model.component_data_objects(
        pyo.Var,
        active=True,
        descend_into=True,
    ):

        # Do not treat master/linking variables as operational z.
        if var.name in present_linking_vars:
            continue

        if var.fixed:
            continue

        if var.is_binary() or var.is_integer():
            operational_discrete_vars.append(var)

    # ================================================================
    # Solve the actual operational MIP.
    # ================================================================

    results = solver.solve(
        operational_model,
        tee=False,
        symbolic_solver_labels=_use_symbolic_labels(solver),
    )

    termination = results.solver.termination_condition

    # No z* exists if the MIP did not solve optimally.
    if termination != TerminationCondition.optimal:
        return results, None, []

    # ================================================================
    # Save the optimal operational discrete state.
    # ================================================================

    operational_discrete_state = {
        "binary": {},
        "integer": {},
    }

    fix_info = []

    for var in operational_discrete_vars:

        if var.value is None:
            raise RuntimeError(
                "Operational discrete variable "
                f"'{var.name}' has no value after the MIP solve."
            )

        z_value = pyo.value(var)

        # Check that the solver returned an integer value.
        if abs(z_value - round(z_value)) > 1e-8:
            raise ValueError(
                "Operational discrete variable "
                f"'{var.name}' has non-integral value "
                f"{z_value} after the MIP solve."
            )

        z_value = int(round(z_value))

        if var.is_binary():
            operational_discrete_state["binary"][
                var.name
            ] = z_value

        else:
            operational_discrete_state["integer"][
                var.name
            ] = z_value

        # Save state needed to restore the variable later.
        fix_info.append(
            {
                "var": var,
                "original_domain": var.domain,
                "was_fixed": var.fixed,
                "original_value": var.value,
            }
        )

        # ============================================================
        # Fix z = z*
        # ============================================================

        var.fix(z_value)

    return (
        results,
        operational_discrete_state,
        fix_info,
    )


def restore_operational_discrete_variables(
    fix_info,
):
    """
    Restore operational discrete variables after fixed-z LP dual
    extraction.

    This is necessary because the operational model is persistent
    across Benders iterations.
    """

    for change in fix_info:

        var = change["var"]

        # Restore the original Pyomo domain.
        var.domain = change["original_domain"]

        # Restore the original fixed/unfixed status.
        if change["was_fixed"]:

            if change["original_value"] is not None:
                var.fix(change["original_value"])
            else:
                var.fix()

        else:
            var.unfix()

def prepare_lp_for_duals(
    model,
    present_linking_vars=None,
    operational_discrete_method="pj_relax",
):
    """
    Prepare the operational model for LP dual extraction.

    Master/linking variables:
        Binary linking variables are relaxed to [0, 1]; integer linking
        variables are relaxed to continuous domains with their original
        bounds retained. Their Benders equality constraints remain active.

    Operational variables:
        Binary/integer operational variables must already be fixed at
        their incumbent z* values for fixed_z. For pj_relax, binary
        variables are relaxed to [0, 1] and integer variables to continuous
        domains, preserving all original bounds.

    Returns
    -------
    changes : list
        Information required to restore the model after the LP solve.
    """

    if present_linking_vars is None:
        present_linking_vars = set()
    else:
        present_linking_vars = set(present_linking_vars)

    changes = []

    for var in model.component_data_objects(
        pyo.Var,
        active=True,
        descend_into=True,
    ):

        # ============================================================
        # Master/linking variable x
        # ============================================================

        if var.name in present_linking_vars:

            if var.is_binary() or var.is_integer():
                original_lb = pyo.value(var.lb, exception=False)
                original_ub = pyo.value(var.ub, exception=False)
                changes.append(
                    {
                        "var": var,
                        "original_domain": var.domain,
                        "original_lb": original_lb,
                        "original_ub": original_ub,
                        "was_fixed": var.fixed,
                        "original_value": var.value,
                        "kind": "linking",
                    }
                )

                var.domain = (
                    pyo.UnitInterval
                    if var.is_binary()
                    else Reals
                )
                var.setlb(original_lb)
                var.setub(original_ub)

            continue

        # ============================================================
        # Operational variable x
        # ============================================================

        if var.is_binary() or var.is_integer():

            if operational_discrete_method == "fixed_z":

                if not var.fixed:
                    raise RuntimeError(
                        "Operational binary/integer variable "
                        f"'{var.name}' is not fixed before LP dual "
                        "extraction."
                    )

            elif operational_discrete_method == "pj_relax":

                # No fixing: directly relax operational integrality.
                pass

            original_domain = var.domain
            original_lb = pyo.value(var.lb, exception=False)
            original_ub = pyo.value(var.ub, exception=False)
            original_is_binary = var.is_binary()

            changes.append(
                {
                    "var": var,
                    "original_domain": original_domain,
                    "original_lb": original_lb,
                    "original_ub": original_ub,
                    "was_fixed": var.fixed,
                    "original_value": var.value,
                    "kind": "operational",
                }
            )

            var.domain = (
                pyo.UnitInterval
                if original_is_binary
                else Reals
            )
            var.setlb(original_lb)
            var.setub(original_ub)

    return changes


def restore_lp_after_duals(
    model,
    changes,
):
    """
    Restore the model after the LP dual solve.

    This restores domains and fixed status. Operational z variables
    are subsequently restored to their pre-fixed MIP state by
    restore_operational_discrete_variables().
    """

    for change in changes:

        var = change["var"]

        # Restore original domain.
        var.domain = change["original_domain"]
        var.setlb(change["original_lb"])
        var.setub(change["original_ub"])

        # Restore original fixed status.
        if change["was_fixed"]:

            if change["original_value"] is not None:
                var.fix(change["original_value"])
            else:
                var.fix()

        else:
            var.unfix()

def classify_linking_vars(
    instance,
    linking_vars,
):
    """
    Classify master/linking variables that are present in this
    operational subproblem.

    Variables not present in the operational model are ignored.
    """

    linking_vars_by_type = {
        "continuous": [],
        "binary": [],
        "integer": [],
    }

    for name in linking_vars:

        var = instance.find_component(name)

        if var is None:
            continue

        if var.is_binary():
            linking_vars_by_type["binary"].append(name)

        elif var.is_integer():
            linking_vars_by_type["integer"].append(name)

        else:
            linking_vars_by_type["continuous"].append(name)

    return linking_vars_by_type


def get_present_linking_vars(
    linking_vars_by_type,
):
    """
    Return all master/linking variables present in this operational
    subproblem.
    """

    return (
        linking_vars_by_type["continuous"]
        + linking_vars_by_type["binary"]
        + linking_vars_by_type["integer"]
    )

def detect_active_mip_variables(
    model,
    linking_vars=None,
):
    """
    Return genuine operational binary/integer variables.

    Master/linking variables are excluded because they are x
    variables, even though they may themselves be binary/integer.

    Fixed variables are also excluded because they are no longer
    active discrete decisions.
    """

    if linking_vars is None:
        linking_vars = set()
    else:
        linking_vars = set(linking_vars)

    mip_variables = []

    for var in model.component_data_objects(
        pyo.Var,
        active=True,
        descend_into=True,
    ):

        # Master/linking variables are x, not operational z.
        if var.name in linking_vars:
            continue

        # Fixed variables are not active discrete decisions.
        if var.fixed:
            continue

        if var.is_binary() or var.is_integer():
            mip_variables.append(var)

    return mip_variables

def solve_local_subproblems(subproblems, planning_sol, expect_feasible, solver):

    results = {}

    for sp in subproblems:

        m = sp["model"]
        w = sp["subproblem_index"]
        linking = sp["linking_variables_sub"]

        results[w] = solve_subproblem(
            m,
            planning_sol,
            linking,
            expect_feasible,
            solver,
        )

    return results


def solve_subproblem(
    model,
    planning_sol,
    linking_vars,
    expect_feasible,
    optimiser,
):

    fix_linking_variables(
        model,
        planning_sol,
        linking_vars,
    )

    results = optimiser.solve(
        model,
        tee=False,
    )

    status = results.solver.status
    term = results.solver.termination_condition

    # ------------------------------------------------------------
    # Feasible case
    # ------------------------------------------------------------

    if term == "optimal":

        op_cost = -value(model.OP_NPV)

        duals = [
            -model.dual[
                model.find_component(name)
            ]
            for name in linking_vars
        ]

        return {
            "op_cost": op_cost,
            "lambda": duals,
            "theta_coeff": 1,
        }

    # ------------------------------------------------------------
    # Infeasible case
    # ------------------------------------------------------------

    if expect_feasible:

        raise RuntimeError(
            "Subproblem infeasible (expected feasible)."
        )

    # ------------------------------------------------------------
    # Feasibility cut
    # ------------------------------------------------------------

    model.slack_fix.deactivate()

    model.obj.deactivate()

    model.feasibility_obj = Objective(
        expr=model.slack_max
    )

    optimiser.solve(
        model
    )

    op_cost = value(
        model.feasibility_obj
    )

    duals = [
            -model.dual[
            model.find_component(name)
            ]
        for name in linking_vars
    ]

    model.feasibility_obj.deactivate()
    model.obj.activate()
    model.slack_fix.activate()

    return {
        "op_cost": op_cost,
        "lambda": duals,
        "theta_coeff": 0,
    }
# ==========================================
# 1. Define the Generic Base Function
# ==========================================
@singledispatch
def solve_subproblems(m_subproblems, planning_sol: NamedTuple, expect_feasible_subproblems: bool):
    """Base function signature that throws an error if an invalid type is passed."""
    raise TypeError(f"Unsupported type for m_subproblems: {type(m_subproblems)}")


# ==========================================
# 2. Method Variant A: For Standard Lists (Vector equivalent)
# ==========================================
@solve_subproblems.register(list)
def _(m_subproblems: List[Dict[Any, Any]], planning_sol: NamedTuple, expect_feasible_subproblems: bool,solver) -> Dict[
    Any, Any]:
    # Single-threaded, local execution
    sub_results = solve_local_subproblems(m_subproblems, planning_sol, expect_feasible_subproblems,solver)
    return sub_results


# ==========================================
# 3. Method Variant B: For Distributed Layouts (DArray equivalent)
# ==========================================
# Python doesn't have an exact built-in DArray. We define a custom class
# to mimic a distributed structure holding chunks of data for specific workers.
class DistributedArrayMock:
    def __init__(self, data_chunks: List[List[Dict[Any, Any]]], worker_ids: List[int]):
        self.data_chunks = data_chunks  # A list of lists (the chunks)
        self.worker_ids = worker_ids  # Analogous to workers()


@solve_subproblems.register(DistributedArrayMock)
def _(m_subproblems: DistributedArrayMock, planning_sol: NamedTuple, expect_feasible_subproblems: bool) -> Dict[
    Any, Any]:
    p_id = m_subproblems.worker_ids
    np_id = len(p_id)

    # Store parallel future objects
    futures = {}

    # ProcessPoolExecutor manages the worker processes pool (analogous to @sync/@async)
    with ProcessPoolExecutor(max_workers=np_id) as executor:
        for k in range(np_id):
            # Isolate the worker's specific piece of data (localpart equivalent)
            local_part = m_subproblems.data_chunks[k]

            # Submit to the worker process (equivalent to @fetchfrom / @spawnat)
            future = executor.submit(
                solve_local_subproblems,
                local_part,
                planning_sol,
                expect_feasible_subproblems
            )
            futures[future] = p_id[k]

    # Combine results as they complete (equivalent to merge(sub_results...))
    merged_results = {}
    for future in as_completed(futures):
        worker_result = future.result()  # Fetches the dictionary returned by the worker
        merged_results.update(worker_result)  # Python dictionary merging/updating

    return merged_results

def get_shared_constraints(instance, planning_constraint_names):

    from pyomo.environ import Constraint

    shared_constraints = []

    for component in instance.component_objects(
        Constraint,
        active=True,
    ):
        if component.name in planning_constraint_names:
            shared_constraints.append(component.name)

    return shared_constraints