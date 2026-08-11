from pyomo.environ import value
from pyomo.environ import (
    Var,
    ConstraintList,
    Objective,
    ObjectiveList,
    value,
    minimize,
    SolverFactory,
    NonNegativeReals,
    Reals,
    Constraint,
)

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import singledispatch
from typing import Any, Dict, List, NamedTuple
from gridpath.auxiliary.dynamic_components import DynamicComponents
from gridpath.benders.operational_model_creation import (
    create_operational_problem_benders,
    define_linking_vars
)

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

    for name in linking_vars:
        var = model.find_component(name)

        var.fix(planning_sol.values[name])

        # mimic JuMP behavior
        var.domain = Reals

import multiprocessing as mp
def benders_subproblem_worker(
    key,
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

    print(
        f"Worker started for {key}. "
        f"Components: {len(list(instance.component_objects()))}"
    )

    # ---------------------------------------------------------
    # Determine the linking variables for THIS subproblem
    # ---------------------------------------------------------

    linking_vars = define_linking_vars(instance)

    # Tell the master that this worker is ready and provide
    # the names of the linking variables.
    output_queue.put(
            {
                "status": "READY",
                "linking_vars": linking_vars,
            }
    )

    # Solver choice INSIDE the worker.

    if parsed_arguments.solver_executable is not None:
        optimiser = SolverFactory(
            parsed_arguments.solver,
            executable=parsed_arguments.solver_executable,
        )
    else:
        optimiser = SolverFactory(parsed_arguments.solver)

    # Remain alive for all Benders iterations

    while True:

        message = input_queue.get()

        if message == "STOP":
            break

        # -----------------------------------------------------
        # Message contains information from the master problem
        # -----------------------------------------------------

        planning_sol = message["planning_sol"]

        # -----------------------------------------------------
        # Update existing ConcreteModel
        # -----------------------------------------------------

        fix_linking_variables(
            instance,
            planning_sol,
            linking_vars,
        )

        # -----------------------------------------------------
        # Solve existing ConcreteModel
        # -----------------------------------------------------

        results = optimiser.solve(
            instance,
            tee=False,
        )

        # -----------------------------------------------------
        # Extract ONLY the information Benders needs
        # -----------------------------------------------------

        subop_sol = process_subproblem_solution(
            instance,
            results,
            linking_vars,
            expect_feasible,
            optimiser
        )

        # -----------------------------------------------------
        # Return only small, pickleable data
        # -----------------------------------------------------

        output_queue.put(
            (
                key,
                subop_sol,
            )
        )

def process_subproblem_solution(
    instance,
    results,
    linking_vars,
    expect_feasible,
    optimiser,
):
    status = results.solver.status
    term = results.solver.termination_condition

    # ---------------------------------------------------------
    # Feasible case
    # ---------------------------------------------------------

    if term == "optimal":
        op_cost = value(instance.MIN_OP_COSTS)

        duals = [
            instance.dual[instance.find_component(name)]
            for name in linking_vars
        ]

        return {
            "op_cost": op_cost,
            "lambda": duals,
            "theta_coeff": 1,
        }

    # ---------------------------------------------------------
    # Infeasible case
    # ---------------------------------------------------------

    if expect_feasible:
        raise RuntimeError(
            "Subproblem infeasible (expected feasible)."
        )

    # ---------------------------------------------------------
    # Solve feasibility problem
    # ---------------------------------------------------------

    instance.slack_fix.deactivate()

    instance.obj.deactivate()

    instance.feasibility_obj = Objective(
        expr=instance.slack_max
    )

    optimiser.solve(instance)

    op_cost = value(instance.feasibility_obj)

    duals = [
        instance.dual[instance.find_component(name)]
        for name in linking_vars
    ]

    instance.feasibility_obj.deactivate()
    instance.obj.activate()
    instance.slack_fix.activate()

    return {
        "op_cost": op_cost,
        "lambda": duals,
        "theta_coeff": 0,
    }

def parallelise_solve(pool_data):

    # Here model will be the subproblems. We will need an index with more
    # features than this though.
    (
        key,
        model,
        planning_sol,
        linking_variables_sub,
        expect_feasible,
        optimiser,
    ) = pool_data


    subop_sol = solve_subproblem(
        model=model,
        planning_sol=planning_sol,
        linking_vars=linking_variables_sub,
        expect_feasible=expect_feasible,
        optimiser=optimiser
    )

    return subop_sol

def solve_subproblem(model,
                     planning_sol,
                     linking_vars,
                     expect_feasible,
                     optimiser):

    fix_linking_variables(model, planning_sol, linking_vars)

    results = optimiser.solve(model, tee=False)

    status = results.solver.status
    term = results.solver.termination_condition


    # feasible case

    if term == "optimal":

        op_cost = value(model.MIN_OP_COSTS)

        duals = [
            model.dual[model.find_component(name)]
            for name in linking_vars
        ]

        return {
            "op_cost": op_cost,
            "lambda": duals,
            "theta_coeff": 1,
        }


    # infeasible case

    if expect_feasible:

        raise RuntimeError("Subproblem infeasible (expected feasible).")


    # feasibility cut

    model.slack_fix.deactivate()

    model.obj.deactivate()
    model.feasibility_obj = Objective(expr=model.slack_max)

    optimiser.solve(model)

    op_cost = value(model.feasibility_obj)

    duals = [
        model.dual[model.find_component(name)]
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