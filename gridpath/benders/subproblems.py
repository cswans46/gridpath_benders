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

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import singledispatch
from typing import Any, Dict, List, NamedTuple
from gridpath.auxiliary.dynamic_components import DynamicComponents
from gridpath.benders.operational_model_creation import (
    create_operational_problem_benders
)
import os
import logging
class BendersState(NamedTuple):
    values: tuple

FEASIBILITY_RELATIVE_TOLERANCE = 5e-5
BENDERS_RECOURSE_OBJECTIVE_SCALE = 1e6

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
        return optimiser.solve(
            instance,
            tee=False,
            symbolic_solver_labels=True,
        )

    scoped_options = {
        optimizer_key: (
            optimizer_values[optimizer_name]
            if optimizer_name is not None
            else optimiser.options.get(
                optimizer_key, optimizer_values["automatic"]
            )
        ),
        **(extra_options or {}),
    }
    previous_options = {
        key: (key in optimiser.options, optimiser.options.get(key))
        for key in scoped_options
    }
    try:
        optimiser.options.update(scoped_options)
        return optimiser.solve(
            instance,
            tee=False,
            symbolic_solver_labels=True,
        )
    finally:
        for key, (was_present, previous_value) in previous_options.items():
            if was_present:
                optimiser.options[key] = previous_value
            else:
                optimiser.options.pop(key, None)

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
    for var in model.component_data_objects(
        pyo.Var,
        active=True,
        descend_into=True,
    ):
        if (
            var.name in linking_var_names
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
        lower = constraint.lower
        upper = constraint.upper
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
            constraint.set_value(
                (lower - scale * model.slack_max, body, upper + scale * model.slack_max)
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
def benders_subproblem_worker(
    key,
    planning_constraint_names,
    linking_vars,
    input_queue,
    output_queue,
    scenario_directory,
    multi_stage,
    parsed_arguments
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

    dynamic_components = DynamicComponents()

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

    optimiser = SolverFactory(parsed_arguments.solver)

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
    # Recourse lower bound.
    # ================================================================

    recourse_ub = calculate_recourse_upper_bound(
        instance,
        optimiser,
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
    # Tell the parent process that this worker is ready.
    # ================================================================

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
            break

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
                symbolic_solver_labels=True,
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

        # ============================================================
        # Send result back to the Benders master.
        # ============================================================

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
            var.setlb(None)
            var.setub(None)
        else:
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

def _active_primal_violations(model):
    """Return normalized violations in the model's current active rows/bounds."""
    violations = []

    for constraint in model.component_data_objects(
        Constraint,
        active=True,
        descend_into=True,
    ):
        body = pyo.value(constraint.body, exception=False)
        if body is None:
            continue

        lower = pyo.value(constraint.lower, exception=False)
        upper = pyo.value(constraint.upper, exception=False)
        violation = max(
            0.0,
            float(lower) - float(body) if lower is not None else 0.0,
            float(body) - float(upper) if upper is not None else 0.0,
        )
        scale = max(
            1.0,
            abs(float(lower)) if lower is not None else 0.0,
            abs(float(upper)) if upper is not None else 0.0,
        )
        if violation > 0.0:
            violations.append(
                (constraint.name, violation, violation / scale)
            )

    for var in model.component_data_objects(
        pyo.Var,
        active=True,
        descend_into=True,
    ):
        if var.value is None:
            continue

        lower = pyo.value(var.lb, exception=False)
        upper = pyo.value(var.ub, exception=False)
        var_value = float(var.value)
        violation = max(
            0.0,
            float(lower) - var_value if lower is not None else 0.0,
            var_value - float(upper) if upper is not None else 0.0,
        )
        scale = max(
            1.0,
            abs(float(lower)) if lower is not None else 0.0,
            abs(float(upper)) if upper is not None else 0.0,
        )
        if violation > 0.0:
            violations.append(
                (var.name, violation, violation / scale)
            )

    return sorted(violations, key=lambda item: item[2], reverse=True)

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

    if not hasattr(instance, "slack_max") or not hasattr(instance, "slack_fix"):
        add_slacks_to_subproblem(instance)

    _set_feasibility_restoration_mode(
        instance,
        active=True,
    )

    # -------------------------------------------------------------
    # Feasibility restoration problem
    # -------------------------------------------------------------

    instance.slack_fix.deactivate()

    original_objectives = list(
        instance.component_data_objects(
            Objective,
            active=True,
            descend_into=True,
        )
    )
    for objective in original_objectives:
        objective.deactivate()

    if hasattr(instance, "feasibility_obj"):
        instance.feasibility_obj.set_value(instance.slack_max)
        instance.feasibility_obj.activate()
    else:
        instance.feasibility_obj = Objective(
            expr=instance.slack_max,
            sense=minimize,
        )

    feasibility_results = _solve_with_lp_optimizer(
        instance,
        optimiser,
        "automatic",
    )

    feasibility_term = (
        feasibility_results.solver.termination_condition
    )

    # Retry alternate algorithms if the restoration solve is non-optimal.
    for retry_optimizer in (
        "primal_simplex",
        "interior_point",
    ):
        if feasibility_term == TerminationCondition.optimal:
            break
        logging.getLogger(__name__).warning(
            "Restoration solve returned %s; retrying with %s",
            feasibility_term,
            retry_optimizer,
        )
        feasibility_results = _solve_with_lp_optimizer(
            instance,
            optimiser,
            retry_optimizer,
        )
        feasibility_term = (
            feasibility_results.solver.termination_condition
        )

    if feasibility_term != TerminationCondition.optimal:
        restoration_dump_base = os.environ.get(
            "GRIDPATH_BENDERS_RESTORATION_LP"
        )
        if restoration_dump_base:
            subproblem_name = str(
                getattr(subproblem_key, "subproblem", subproblem_key)
            ).replace(os.sep, "_")
            dump_root, dump_extension = os.path.splitext(
                restoration_dump_base
            )
            restoration_dump_path = (
                f"{dump_root}_{subproblem_name}"
                f"{dump_extension or '.lp'}"
            )
            instance.write(
                restoration_dump_path,
                io_options={"symbolic_solver_labels": True},
            )

        instance.feasibility_obj.deactivate()
        _set_feasibility_restoration_mode(
            instance,
            active=False,
        )
        for objective in original_objectives:
            objective.activate()

        raise RuntimeError(
            "Feasibility problem did not solve to optimality. "
            f"Termination condition: {feasibility_term}"
        )

    restoration_slack = pyo.value(instance.slack_max)
    if restoration_slack <= FEASIBILITY_RELATIVE_TOLERANCE:
        instance.feasibility_obj.deactivate()
        for objective in original_objectives:
            objective.activate()
        restoration_values = [
            (var, var.value)
            for var in instance.component_data_objects(
                pyo.Var,
                active=True,
                descend_into=True,
            )
        ]

        for row_info in instance._feasibility_slack_row_info:
            row_info["constraint"].set_value(
                row_info["original_expression"]
            )
        _restore_feasibility_variable_bounds(instance, elastic=False)

        residuals = []
        for row_info in instance._feasibility_slack_row_info:
            body_value = pyo.value(row_info["body"], exception=False)
            if body_value is None:
                continue
            lower = row_info["lower"]
            upper = row_info["upper"]
            violation = max(
                0.0,
                lower - body_value if lower is not None else 0.0,
                body_value - upper if upper is not None else 0.0,
            )
            normalized_violation = violation / row_info["scale"]
            if normalized_violation > FEASIBILITY_RELATIVE_TOLERANCE:
                for info in instance._feasibility_slack_row_info:
                    info["constraint"].set_value(info["elastic_expression"])
                _restore_feasibility_variable_bounds(instance, elastic=True)
                instance.slack_fix.activate()
                instance.slack_fix.set_value(instance.slack_max == 0)
                raise RuntimeError(
                    "Near-zero restoration produced a row above the "
                    "normalized feasibility tolerance. "
                    f"Subproblem: {subproblem_key}; "
                    f"row: {row_info['constraint'].name}; "
                    f"normalized violation: {normalized_violation:.12g}."
                )
            if violation > 0:
                residuals.append((row_info, violation))

        for row_info, violation in residuals:
            constraint = row_info["constraint"]
            body = row_info["body"]
            lower = row_info["lower"]
            upper = row_info["upper"]
            allowance = max(
                violation,
                FEASIBILITY_RELATIVE_TOLERANCE * row_info["scale"],
            )
            if row_info["equality"] or (
                lower is not None and upper is not None
            ):
                constraint.set_value(
                    (lower - allowance, body, upper + allowance)
                )
            elif upper is not None:
                constraint.set_value(body <= upper + allowance)
            elif lower is not None:
                constraint.set_value(body >= lower - allowance)

        bound_residuals = []
        restoration_value_by_id = {
            id(var): var_value for var, var_value in restoration_values
        }
        for info in instance._feasibility_bound_info:
            var = info["var"]
            var_value = restoration_value_by_id.get(id(var))
            if var_value is None:
                continue
            lower = info["lower"]
            upper = info["upper"]
            violation = max(
                0.0,
                lower - var_value if lower is not None else 0.0,
                var_value - upper if upper is not None else 0.0,
            )
            if violation <= 0:
                continue
            scale = info["scale"]
            normalized_violation = violation / scale
            bound_residuals.append((var.name, violation, normalized_violation))
            if normalized_violation > FEASIBILITY_RELATIVE_TOLERANCE:
                for row_info in instance._feasibility_slack_row_info:
                    row_info["constraint"].set_value(
                        row_info["elastic_expression"]
                    )
                _restore_feasibility_variable_bounds(instance, elastic=True)
                instance.slack_fix.activate()
                instance.slack_fix.set_value(instance.slack_max == 0)
                raise RuntimeError(
                    "Near-zero restoration produced a variable-bound "
                    "violation above the normalized feasibility tolerance. "
                    f"Subproblem: {subproblem_key}; variable: {var.name}; "
                    f"normalized violation: {normalized_violation:.12g}."
                )

            allowance = max(
                violation,
                FEASIBILITY_RELATIVE_TOLERANCE * scale,
            )
            if lower is not None and var_value < lower:
                var.setlb(lower - allowance)
            if upper is not None and var_value > upper:
                var.setub(upper + allowance)

        instance.slack_fix.activate()
        instance.slack_fix.set_value(instance.slack_max == 0)
        retry_model_violations = _active_primal_violations(instance)
        retry_results = optimiser.solve(
            instance,
            tee=False,
            symbolic_solver_labels=True,
        )
        first_retry_termination = (
            retry_results.solver.termination_condition
        )
        if (
            first_retry_termination != TerminationCondition.optimal
            and str(getattr(optimiser, "name", "")).lower().startswith(
                ("mosek", "gurobi")
            )
        ):
            retry_results = _solve_with_lp_optimizer(
                instance,
                optimiser,
                "automatic",
            )

        if retry_results.solver.termination_condition == TerminationCondition.optimal:
            try:
                return process_subproblem_solution(
                    instance,
                    retry_results,
                    planning_sol,
                    linking_vars_by_type,
                    optimiser=optimiser,
                    active_linking_vars=active_linking_vars,
                    present_linking_vars=present_linking_vars,
                    subproblem_key=subproblem_key,
                )
            finally:
                _set_feasibility_restoration_mode(
                    instance,
                    active=False,
                )
        retry_termination = retry_results.solver.termination_condition
        _set_feasibility_restoration_mode(
            instance,
            active=False,
        )
        for var, var_value in restoration_values:
            var.set_value(var_value, skip_validation=True)
        bound_violations = []
        for info in instance._feasibility_bound_info:
            var = info["var"]
            var_value = restoration_value_by_id.get(id(var))
            if var_value is None:
                continue
            lower = info["lower"]
            upper = info["upper"]
            violation = max(
                0.0,
                lower - var_value if lower is not None else 0.0,
                var_value - upper if upper is not None else 0.0,
            )
            if violation > 1e-7:
                bound_violations.append((var.name, violation))
        raise RuntimeError(
            "The restoration LP reported a near-zero violation, but the "
            "original subproblem remained infeasible when retried. "
            f"Subproblem: {subproblem_key}; "
            f"restoration slack: {restoration_slack:.12g}; "
            f"first retry termination: {first_retry_termination}; "
            f"retry termination: {retry_termination}; "
            f"assembled retry model violations: "
            f"{retry_model_violations[:20]}; "
            f"constraint residuals: {[ (info['constraint'].name, violation) for info, violation in residuals[:20] ]}; "
            f"variable-bound residuals: {bound_violations[:20]}; "
            f"restoration bound residuals: {bound_residuals[:20]}."
        )

    # -------------------------------------------------------------
    # Fix operational integer variables before extracting duals.
    # -------------------------------------------------------------

    lp_changes = prepare_lp_for_duals(
        instance,
        present_linking_vars=present_linking_vars,
    )

    present_linking_var_set = set(present_linking_vars)

    remaining_integer_vars = [
        var.name
        for var in instance.component_data_objects(
            pyo.Var,
            descend_into=True,
        )
        if (
                (var.is_binary() or var.is_integer())
                and not var.fixed
                and var.name not in present_linking_var_set
        )
    ]

    if remaining_integer_vars:

        raise RuntimeError(
            "Feasibility LP still contains unfixed integer "
            f"variables: {remaining_integer_vars[:20]}"
        )

    # -------------------------------------------------------------
    # Re-solve after fixing integers.
    # -------------------------------------------------------------

    lp_results = _solve_with_lp_optimizer(
        instance,
        optimiser,
        "primal_simplex",
    )

    if (
        lp_results.solver.termination_condition
        != TerminationCondition.optimal
    ):

        raise RuntimeError(
            "Feasibility LP did not solve to optimality after "
            "fixing operational integer variables."
        )

    benders_lambda = {}

    for name in active_linking_vars:

        constraint = (
            instance.BendersLinkingConstraintMap.get(name)
        )

        if constraint is None:

            var = instance.find_component(name)

            if var is None:
                raise RuntimeError(
                    "Active linking variable "
                    f"'{name}' does not exist in the "
                    "operational model."
                )

            raise RuntimeError(
                "Linking variable "
                f"'{name}' exists in the operational model "
                "but has no Benders linking constraint."
            )

        dual = instance.dual.get(
            constraint,
            None,
        )

        if dual is None:

            raise RuntimeError(
                "No dual value was returned for feasibility "
                f"linking constraint '{name}'."
            )

        benders_lambda[name] = float(dual)

    continuous_lambda = {
        name: benders_lambda[name]
        for name in active_linking_vars
        if name in benders_lambda
    }

    # Duals come from the LP relaxation, so the cut intercept must too.
    op_cost = pyo.value(instance.slack_max)
    restoration_slack = op_cost
    if os.environ.get("GRIDPATH_BENDERS_INFEAS_DIAG"):
        rows = []
        for c in instance.component_data_objects(pyo.Constraint, active=True):
            d = instance.dual.get(c, None)
            if d is not None and abs(d) > 1e-14:
                rows.append((c.name[:90], float(d)))
        rows.sort(key=lambda r: -abs(r[1]))
        gens = {
            k: pyo.value(v)
            for k, v in instance.Project_Carbon_Credits_Generated.items()
        }
        ems = {}
        for (p_, t_), v in instance.Project_Carbon_Emissions.items():
            val_ = pyo.value(v) * pyo.value(instance.hrs_in_tmp[t_]) * pyo.value(instance.tmp_weight[t_])
            ems[p_] = ems.get(p_, 0.0) + val_
        cexp = {}
        for nm_c in ("Project_Carbon_Credits_Generated_Constraint", "Benders_Carbon_Credit_Generated_Total_Constraint"):
            comp = instance.find_component(nm_c)
            if comp is not None:
                for k_c, c_c in comp.items():
                    cexp[(nm_c[:12],) + (k_c if isinstance(k_c, tuple) else (k_c,))] = str(c_c.expr)[:400]
        xbar = {
            k: v for k, v in continuous_point.items() if "CARBON_CREDIT" in k
        } if isinstance(continuous_point, dict) else str(continuous_point)[:300]
        if os.environ.get("GRIDPATH_BENDERS_CUTLOG"):
            import json
            with open(os.environ["GRIDPATH_BENDERS_CUTLOG"], "a") as f_:
                f_.write(json.dumps({
                    "sub": str(getattr(subproblem_key, "subproblem", subproblem_key)),
                    "t": restoration_slack,
                    "lam": benders_lambda,
                    "x": continuous_point,
                }, default=float) + "\n")
        if os.environ.get("GRIDPATH_BENDERS_INFEAS_EXPERIMENT"):
            import re as _re
            saved = {n: c.expr for n, c in instance.BendersLinkingConstraintMap.items()}
            def _resolve(newvals, label):
                for n, v in newvals.items():
                    instance.BendersLinkingConstraintMap[n].set_value(instance.find_component(n) == v)
                _solve_with_lp_optimizer(instance, optimiser, "automatic")
                logging.getLogger("infeas_diag").warning("INFEAS EXPERIMENT %s t=%s", label, pyo.value(instance.slack_max))
                _rows = sorted(((instance.dual.get(c, 0.0) or 0.0, c.name[:80]) for c in instance.component_data_objects(pyo.Constraint, active=True)), key=lambda r: -abs(r[0]))[:14]
                logging.getLogger("infeas_diag").warning("INFEAS EXPERIMENT %s rows=%s gens=%s", label, _rows, {k: round(pyo.value(v)) for k, v in instance.Project_Carbon_Credits_Generated.items()})
                for n, e in saved.items():
                    instance.BendersLinkingConstraintMap[n].set_value(e)
            base = dict(continuous_point)
            ref = dict(base)
            for n in base:
                if "ABSOLUTE_THRESHOLD_ALLOCATION" in n:
                    ref[n] = 2e6 / 3
            zone_sum = {}
            for n in base:
                m_ = _re.match(r"BENDERS_PROJECT_CARBON_CREDIT_ABSOLUTE_THRESHOLD_ALLOCATION\[(.+?),2030", n)
                if m_ and "2030" in n:
                    zone_sum[m_.group(1)] = ref[n]
            z1 = {"Coal", "Gas_CCGT_ret", "Gas_CT_ret", "Coal_ret"}
            for n in base:
                if n.startswith("BENDERS_CARBON_CREDIT_GENERATED_TOTAL[Carbon_Credits_Zone1") or n.startswith("BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION[Carbon_Credits_Zone1"):
                    ref[n] = sum(v for k, v in zone_sum.items() if k in z1)
                elif n.startswith("BENDERS_CARBON_CREDIT_GENERATED_TOTAL[Carbon_Credits_Zone2") or n.startswith("BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION[Carbon_Credits_Zone2"):
                    ref[n] = sum(v for k, v in zone_sum.items() if k not in z1)
            sel = {n: ref[n] for n in base if "CARBON_CREDIT" in n and n in instance.BendersLinkingConstraintMap}
            _resolve(sel, "credit-ref-point")
            import json as _json
            _refrec = [_json.loads(z) for z in open(os.environ["GRIDPATH_BENDERS_INFEAS_EXPERIMENT"]) if '"ref": true' in z][-1]["x"]
            _full = {n: _refrec[n] for n in base if n in _refrec and n in instance.BendersLinkingConstraintMap}
            _resolve(_full, "FULL-ref-point")
            for _alg in ("primal_simplex", "interior_point", "automatic"):
                for n, v in _full.items():
                    instance.BendersLinkingConstraintMap[n].set_value(instance.find_component(n) == v)
                _rr = _solve_with_lp_optimizer(instance, optimiser, _alg)
                logging.getLogger("infeas_diag").warning("INFEAS EXPERIMENT FULL alg=%s t=%s term=%s", _alg, pyo.value(instance.slack_max), _rr.solver.termination_condition)
            for n, e in saved.items():
                instance.BendersLinkingConstraintMap[n].set_value(e)
            try:
                instance.slack_fix.activate()
                for n, v in _full.items():
                    instance.BendersLinkingConstraintMap[n].set_value(instance.find_component(n) == v)
                _rr = _solve_with_lp_optimizer(instance, optimiser, "automatic")
                logging.getLogger("infeas_diag").warning("INFEAS EXPERIMENT FULL-ref-point slack_fix term=%s", _rr.solver.termination_condition)
                instance.slack_fix.deactivate()
                for n, e in saved.items():
                    instance.BendersLinkingConstraintMap[n].set_value(e)
            except Exception as _e:
                logging.getLogger("infeas_diag").warning("INFEAS EXPERIMENT slack_fix error %s", _e)
            _diffs = {n: (base[n], _refrec.get(n)) for n in base if n in _refrec and abs(base[n] - _refrec[n]) > 1e-9 and "CARBON_CREDIT" not in n}
            logging.getLogger("infeas_diag").warning("INFEAS EXPERIMENT non-credit diffs=%s", _diffs)
            _resolve({}, "same-x")

        gv = {id(v): k for k, v in instance.Project_Carbon_Credits_Generated.items()}
        for c_ in instance.component_data_objects(pyo.Constraint, active=True):
            try:
                hit = [gv[id(v)] for v in _idv(c_.body, include_fixed=False) if id(v) in gv]
            except Exception:
                continue
            if hit:
                r_ = generate_standard_repn(c_.body, compute_values=True)
                logging.getLogger("infeas_diag").warning(
                    "INFEAS GROW %s lo=%s up=%s vars=%s coefs=%s",
                    c_.name, pyo.value(c_.lower) if c_.lower is not None else None,
                    pyo.value(c_.upper) if c_.upper is not None else None,
                    [(v.name, v.value) for v in r_.linear_vars][:12], list(r_.linear_coefs)[:12],
                )
        for k_, v_ in instance.Project_Carbon_Credits_Generated.items():
            logging.getLogger("infeas_diag").warning("INFEAS GVAR %s lb=%s ub=%s val=%s", k_, v_.lb, v_.ub, v_.value)
        logging.getLogger("infeas_diag").warning("INFEAS EXPR %s", cexp)
        logging.getLogger("infeas_diag").warning(
            "INFEAS DIAG t=%s lambda_nonzero=%s top_dual_rows=%s gens=%s ems=%s xbar=%s",
            restoration_slack,
            {k: v for k, v in benders_lambda.items() if abs(v) > 1e-14},
            rows[:30],
            gens,
            ems,
            xbar,
        )
    # -------------------------------------------------------------

    restore_lp_after_duals(
        instance,
        lp_changes,
    )

    instance.feasibility_obj.deactivate()
    _set_feasibility_restoration_mode(
        instance,
        active=False,
    )
    for objective in original_objectives:
        objective.activate()

    return {
        "feasible": False,
        "op_cost": float(op_cost),
        "benders_lambda": benders_lambda,
        "continuous_lambda": continuous_lambda,
        "continuous_point": continuous_point,
        "theta_coeff": 0.0,
    }

def relax_operational_integrality(model):
    """
    Relax operational integer variables for LP-based recourse
    lower-bound calculation.

    Variable bounds are preserved.
    """

    for var in model.component_data_objects(
        pyo.Var,
        active=True,
        descend_into=True,
    ):

        if var.is_binary():
            var.domain = pyo.UnitInterval

        elif var.is_integer():
            var.domain = pyo.Reals


def calculate_recourse_upper_bound(
    instance,
    optimiser,
):
    """
    Calculate a global LP-relaxation upper bound for operational NPV.

    The original model is not modified.
    """

    lp_instance = instance.clone()

    relax_operational_integrality(
        lp_instance
    )

    results, _ = _solve_scaled_recourse_lp(
        lp_instance,
        optimiser,
    )

    termination = (
        results.solver.termination_condition
    )

    if termination != TerminationCondition.optimal:

        raise RuntimeError(
            "LP relaxation for recourse lower bound did not solve "
            f"to optimality. Termination condition: {termination}"
        )

    upper_bound = value(
        lp_instance.OP_NPV
    )

    return upper_bound


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
        symbolic_solver_labels=True,
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