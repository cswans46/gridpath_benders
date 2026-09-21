import pyomo.environ as pyo
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

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import singledispatch
from typing import Any, Dict, List, NamedTuple
from gridpath.auxiliary.dynamic_components import DynamicComponents
from gridpath.benders.operational_model_creation import (
    create_operational_problem_benders,
    define_linking_vars
)
import os

class BendersState(NamedTuple):
    values: tuple

def add_slacks_to_subproblem(model):

    model.EQ_CONSTRAINTS = ConstraintList()
    model.LE_CONSTRAINTS = ConstraintList()
    model.GE_CONSTRAINTS = ConstraintList()

    model.slack_max = Var(domain=NonNegativeReals)
    model.slack_eq = Var(model.EQ_CONSTRAINTS, domain=Reals)

    # <= constraints
    for c in model.LE_CONSTRAINTS.values():
        c.set_value(c.body + model.slack_max <= c.upper)

    # >= constraints
    for c in model.GE_CONSTRAINTS.values():
        c.set_value(c.body - model.slack_max >= c.lower)

    # = constraints
    for i, c in model.EQ_CONSTRAINTS.items():
        c.set_value(c.body - model.slack_eq[i] == 0)

        model.add_component(
            f"slack_eq_bound_{i}",
            Constraint(expr=model.slack_eq[i] <= model.slack_max),
        )
        model.add_component(
            f"slack_eq_bound_neg_{i}",
            Constraint(expr=-model.slack_eq[i] <= model.slack_max),
        )

    model.slack_fix = Constraint(expr=model.slack_max == 0)

def fix_linking_variables(model, planning_sol, linking_vars):

    model.BendersLinkingConstraints.clear()
    # Make a map that maps the constraint to the variable type.
    model.BendersLinkingConstraintMap = {}

    for name in linking_vars:

        var = model.find_component(name)

        if var is None:
            continue

        constraint = model.BendersLinkingConstraints.add(
            var == planning_sol.values[name]
        )

        model.BendersLinkingConstraintMap[name] = constraint

def get_discrete_linking_vars(linking_vars_by_type):
    """
    Return all discrete linking variable names in a deterministic order.
    """
    return (
        linking_vars_by_type["binary"]
        + linking_vars_by_type["integer"]
    )

def get_discrete_state(planning_sol, linking_vars_by_type):
    """
    Return the complete discrete linking-variable configuration
    represented by the planning solution.

    The ordering must remain fixed across all Benders iterations.
    """
    discrete_vars = get_discrete_linking_vars(
        linking_vars_by_type
    )

    return tuple(
        planning_sol.values[name]
        for name in discrete_vars
    )

def get_continuous_point(planning_sol, linking_vars_by_type):
    """
    Return the continuous linking-variable values at the
    current master solution.
    """
    return {
        name: planning_sol.values[name]
        for name in linking_vars_by_type["continuous"]
    }

import multiprocessing as mp
def benders_subproblem_worker(
    key,
    planning_constraint_names,
    linking_vars,
    input_queue,
    output_queue,
    scenario_directory,
    multi_stage,
    expect_feasible,
    parsed_arguments,
):
    """
    Persistent worker for one Benders operational subproblem.

    The ConcreteModel is created ONCE and reused for every
    Benders iteration.
    """

    # ---------------------------------------------------------

    # Build the subproblem ONCE within the same worker

    dynamic_components = DynamicComponents()

    instance = create_operational_problem_benders(
        scenario_directory=scenario_directory,
        key=key,
        dynamic_components=dynamic_components,
        expect_feasible=expect_feasible,
        multi_stage=multi_stage,
        parsed_arguments=parsed_arguments,
    )

    # Solver choice INSIDE the worker.

    if parsed_arguments.solver_executable is not None:
        optimiser = SolverFactory(
            parsed_arguments.solver,
            executable=parsed_arguments.solver_executable,
        )
    else:
        optimiser = SolverFactory(parsed_arguments.solver)

    # Mute solver output for Benders subproblem iterations.
    solver_name = parsed_arguments.solver.lower()

    if "mosek" in solver_name:
        optimiser.options["MSK_IPAR_LOG"] = "0"

    elif "gurobi" in solver_name:
        optimiser.options["OutputFlag"] = 0

    linking_vars_by_type = classify_linking_vars(
        instance,
        linking_vars,
    )

    # Identify investment-related components
    shared_constraints = get_shared_constraints(instance,planning_constraint_names)

    for name in shared_constraints:
        instance.find_component(name).deactivate()

    # Obtain global LP lower bound
    recourse_lb = calculate_recourse_lower_bound(
        instance,
        optimiser,
    )

    # Create the dual variable to access the dual variables of the linking variables.
    instance.dual = Suffix(direction=Suffix.IMPORT)
    # Create the linking constraints for the operational problem (i.e. y_p = y_p^k as in pecci)
    instance.BendersLinkingConstraints = ConstraintList()

    # Tell the master that this worker is ready and provide
    # the names of the linking variables.
    output_queue.put({
        "status": "READY",
        "linking_vars": linking_vars,
        "linking_vars_by_type": linking_vars_by_type,
        "investment_components": shared_constraints,
        "recourse_lower_bound": recourse_lb,
    })

    # Remain alive for all Benders iterations

    while True:

        message = input_queue.get()

        if message == "STOP":
            break

        planning_sol = message["planning_sol"]

        fix_linking_variables(
            instance,
            planning_sol,
            linking_vars,
        )

        # =========================================================
        # 1. Solve the actual operational MIP
        # =========================================================

        mip_results = optimiser.solve(
            instance,
            tee=False,
            symbolic_solver_labels=True,
        )

        mip_term = mip_results.solver.termination_condition

        # =========================================================
        # 2. If MIP is feasible, fix operational integers and
        #    solve the resulting LP for dual information.
        # =========================================================

        if mip_term == TerminationCondition.optimal:

            op_cost = value(instance.MIN_OP_COSTS)

            fixed_integer_vars = fix_operational_integer_variables(
                instance
            )

            if fixed_integer_vars:
                remaining_integer_vars = [
                    var.name
                    for var in instance.component_data_objects(
                        pyo.Var,
                        descend_into=True,
                    )
                    if (var.is_binary() or var.is_integer()) and not var.fixed
                ]

                if remaining_integer_vars:
                    raise RuntimeError(
                        "Fixed-integer LP still contains unfixed integer "
                        f"variables: {remaining_integer_vars[:20]}"
                    )

            if fixed_integer_vars:
                lp_results = optimiser.solve(
                    instance,
                    tee=False,
                    symbolic_solver_labels=True,
                )

                if (
                        lp_results.solver.termination_condition
                        != TerminationCondition.optimal
                ):
                    raise RuntimeError(
                        "Fixed-integer LP did not solve to optimality."
                    )

            subop_sol = process_subproblem_solution(
                instance,
                mip_results,
                linking_vars_by_type,
                expect_feasible,
                optimiser,
            )
            # Add in discrete state assignment.
            discrete_state = get_discrete_state(
                planning_sol,
                linking_vars_by_type,
            )
            # Identify the continuous point at the discrete state.
            continuous_point = get_continuous_point(
                planning_sol,
                linking_vars_by_type,
            )

            subop_sol["discrete_state"] = discrete_state
            subop_sol["continuous_point"] = continuous_point

            # Restore the MIP
            for var, original_domain in fixed_integer_vars:
                var.domain = original_domain
                var.unfix()

        else:

            subop_sol = process_subproblem_solution(
                instance,
                mip_results,
                linking_vars_by_type,
                expect_feasible,
                optimiser,
            )

        output_queue.put(
            (
                key,
                subop_sol,
            )
        )

        # -----------------------------------------------------
        # Save this subproblem's solution
        # -----------------------------------------------------

        from gridpath.run_scenario import save_results

        save_results(
            scenario_directory,
            key.weather,
            key.hydro,
            key.availability,
            key.subproblem,
            key.stage,
            multi_stage,
            instance,
            mip_results,
            dynamic_components,
            parsed_arguments,
        )

def process_subproblem_solution(
    instance,
    mip_results,
    linking_vars_by_type,
    expect_feasible,
    optimiser,
):
    mip_term = mip_results.solver.termination_condition

    # ---------------------------------------------------------
    # Feasible MIP
    # ---------------------------------------------------------

    if mip_term == TerminationCondition.optimal:

        op_cost = value(instance.MIN_OP_COSTS)

        continuous_lambda = {}

        for name in linking_vars_by_type["continuous"]:

            constraint = instance.BendersLinkingConstraintMap[name]

            dual = instance.dual.get(
                constraint,
                None,
            )

            if dual is None:
                raise RuntimeError(
                    f"No dual found for continuous linking "
                    f"variable '{name}' after solving the "
                    "fixed-integer LP."
                )

            continuous_lambda[name] = dual

        return {
            "feasible": True,
            "op_cost": op_cost,
            "continuous_lambda": continuous_lambda,
            "theta_coeff": 1,
        }

    # ---------------------------------------------------------
    # Infeasible case
    # ---------------------------------------------------------

    if expect_feasible:
        raise RuntimeError(
            "Subproblem infeasible (expected feasible)."
        )

    instance.slack_fix.deactivate()
    instance.obj.deactivate()

    instance.feasibility_obj = Objective(
        expr=instance.slack_max
    )

    feasibility_results = optimiser.solve(instance)

    if (
            feasibility_results.solver.termination_condition
            != TerminationCondition.optimal
    ):
        raise RuntimeError(
            "Feasibility subproblem did not solve to optimality."
        )

    fixed_integer_vars = fix_operational_integer_variables(instance)

    if fixed_integer_vars:
        feasibility_lp_results = optimiser.solve(
            instance,
            tee=False,
            symbolic_solver_labels=True,
        )

        if (
                feasibility_lp_results.solver.termination_condition
                != TerminationCondition.optimal
        ):
            raise RuntimeError(
                "Fixed-integer feasibility LP did not solve "
                "to optimality."
            )

    continuous_lambda = {}

    for name in linking_vars_by_type["continuous"]:

        constraint = instance.BendersLinkingConstraintMap[name]

        dual = instance.dual.get(constraint, None)

        if dual is None:
            raise RuntimeError(
                f"No dual found for continuous linking "
                f"variable '{name}' in feasibility problem."
            )

        continuous_lambda[name] = dual

    op_cost = value(instance.feasibility_obj)

    for var in fixed_integer_vars:
        var.unfix()

    instance.feasibility_obj.deactivate()
    instance.obj.activate()
    instance.slack_fix.activate()

    return {
        "feasible": False,
        "op_cost": op_cost,
        "continuous_lambda": continuous_lambda,
        "theta_coeff": 0,
    }

def relax_operational_integrality(model):
    """Relax all binary and integer variables to continuous variables."""

    for var in model.component_data_objects(
        Var,
        descend_into=True,
    ):
        if var.is_binary() or var.is_integer():
            var.domain = Reals

def calculate_recourse_lower_bound(instance, optimiser):
    lp_instance = instance.clone()

    relax_operational_integrality(lp_instance)

    results = optimiser.solve(lp_instance)

    if (
        results.solver.termination_condition
        != TerminationCondition.optimal
    ):
        raise RuntimeError(
            "LP relaxation did not solve to optimality."
        )

    lower_bound = value(
        lp_instance.MIN_OP_COSTS
    )

    return lower_bound

def fix_operational_integer_variables(model):
    fixed_vars = []

    for var in model.component_data_objects(
        pyo.Var,
        descend_into=True,
    ):
        if var.is_binary() or var.is_integer():

            if var.value is None:
                raise RuntimeError(
                    f"Operational integer variable {var.name} "
                    "has no value after MIP solve."
                )

            original_domain = var.domain

            var.fix()
            var.domain = Reals

            fixed_vars.append(
                (var, original_domain)
            )

    return fixed_vars

def classify_linking_vars(model, linking_vars):
    classified = {
        "continuous": [],
        "integer": [],
        "binary": [],
    }

    for name in linking_vars:
        var = model.find_component(name)

        if var is None:
            continue

        if var.is_binary():
            classified["binary"].append(name)

        elif var.is_integer():
            classified["integer"].append(name)

        elif var.is_continuous():
            classified["continuous"].append(name)

        else:
            raise TypeError(
                f"Unsupported domain for linking variable {name}: "
                f"{var.domain}"
            )

    return classified

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