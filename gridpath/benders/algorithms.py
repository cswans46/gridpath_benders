from dataclasses import dataclass
from copy import deepcopy
import math
import time
import logging
import pyomo.environ as pyo
from pyomo.environ import (
    SolverFactory
)
# from gridpath.benders.planning import add_approximate_variable_cost, solve_planning_problem
# from gridpath.benders.regularization import solve_int_level_set_problem
# from gridpath.benders.subproblems import solve_subproblem
from gridpath.benders.planning import *
from gridpath.benders.regularization import *
from gridpath.benders.subproblems import *
from multiprocessing import get_context

logger = logging.getLogger(__name__)

@dataclass
class BendersResult:
    planning_problem: object
    planning_sol: object
    subop_sol: dict
    LB_hist: list
    UB_hist: list
    gap_hist: list
    termination_status: str
    cpu_time: list
    planning_sol_hist: list

def all_variables(model) -> list:
    """Returns a flat list of all variable data objects in the Pyomo model."""
    # print("What is the type",type(model))
    # print("What is the components",model.component_data_objects(pyo.Var, descend_into=True))
    # print("What is the list",list(model.component_data_objects(pyo.Var, descend_into=True)))
    # print("here is the model: ", model.pprint())
    # Still returning an empty list of variables.

    return list(model.component_data_objects(pyo.Var, descend_into=True))

def benders(
    planning_problem,
    dynamic_components,
    scenario_directory,
    benders_subproblems,
    linking_variables_sub,
    setup,
    multi_stage,
    parsed_arguments
):
    """
    Regularized Benders decomposition.
    """
    # TODO: This needs to be altered slightly so that policy constraints have slacks by subperiod added to them.
    expect_feasible = setup["ExpectFeasibleSubproblems"]

    print("Running Benders decomposition...")
    logger.info("Running Benders decomposition")

    solver_start = time.time()
    # If we expect the subproblem to be feasible, we can skip adding slacks to the subproblem variables.
    add_approximate_variable_cost(
        planning_problem,
        benders_subproblems
    )

    max_iter = setup["MaxIter"]
    conv_tol = setup["ConvTol"]
    max_cpu = setup["MaxCpuTime"]

    gamma = setup["StabParam"]
    stab_dynamic = setup["StabDynamic"]

    stab_method = "off" if abs(gamma) < 1e-12 else "int_level_set"

    integer_investment = setup["IntegerInvestment"]

    term_status = "NONE"

    integer_routine = False

    planning_variables = all_variables(planning_problem)
    planning_variable_names = [v.name for v in planning_variables]

    if integer_investment and stab_method != "off":

        integer_vars = [v for v in planning_variables if v.is_integer()]
        binary_vars = [v for v in planning_variables if v.is_binary()]

        for v in integer_vars:
            v.domain = pyo.Reals

        for v in binary_vars:
            v.domain = pyo.PercentFraction

        integer_routine = True

    if parsed_arguments.solver_executable is not None:
        optimiser = SolverFactory(
            parsed_arguments.solver, executable=parsed_arguments.solver_executable
        )
        # Otherwise, only pass the solver name; Pyomo will look for the
        # executable in the PATH
    else:
        optimiser = SolverFactory(parsed_arguments.solver)

    # ---------------------------------------------------------
    # Initially solve planning problem
    # ---------------------------------------------------------

    planning_sol, LB = solve_planning_problem(
        planning_problem,
        planning_variable_names,
        optimiser
    )

    UB = math.inf

    LB_hist = []
    UB_hist = []
    gap_hist = []
    cpu_time = []

    planning_sol_best = deepcopy(planning_sol)
    subop_sol_best = {}

    planning_sol_hist = [
        [planning_sol.values[name] for name in planning_variable_names]
    ]

    # ---------------------------------------------------------
    # Create and Solve Subproblem in assigning processors
    # ---------------------------------------------------------

    print("Building subproblems for Benders decomposition...")

    # What we are doing here is creating the worker directories to assign.
    ctx = mp.get_context("spawn")

    workers = {}
    input_queues = {}
    output_queues = {}

    # Here, we are looping over benders subproblems to assign each worker
    # to a given subproblem. This starts the process within the loop (minimising
    # computational inefficiencies).
    for key in benders_subproblems:
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        # Here process is initialised for the current key (key corresponding to
        # each weather, hydro, subproblem, and availability iteration tuple)
        process = ctx.Process(
            target=benders_subproblem_worker,
            args=(
                key,
                input_queue,
                output_queue,
                scenario_directory,
                multi_stage,
                expect_feasible,
                parsed_arguments,
            ),
        )
        # Process is started here
        process.start()
        # Process is assigned to each worker.
        workers[key] = process
        input_queues[key] = input_queue
        output_queues[key] = output_queue

    linking_variables_sub = {}

    for key in benders_subproblems:

        result = output_queues[key].get()

        if result["status"] != "READY":
            raise RuntimeError(
                f"Worker {key} failed to initialise."
            )

        linking_variables_sub[key] = result["linking_vars"]

    for k in range(max_iter + 1):

        planning_sol_hist.append(
            [
                planning_sol.values[n]
                for n in planning_variable_names
            ]
        )

        # ---------------------------------------------------------
        # Send current planning solution to every operational worker
        # ---------------------------------------------------------

        for key in benders_subproblems:
            input_queues[key].put(
                {
                    "planning_sol": planning_sol,
                }
            )

        # ---------------------------------------------------------
        # Wait for all operational subproblems
        # ---------------------------------------------------------

        subop_sol = {}

        for key in benders_subproblems:
            result_key, result = output_queues[key].get()
            subop_sol[result_key] = result

        logger.info(
            "Subproblem solve took %.2f seconds",
            time.time() - solver_start,
        )

        # ---------------------------------------------------------
        # Update upper bound
        # ---------------------------------------------------------

        UB_new = compute_upper_bound(
            planning_sol,
            subop_sol,
        )

        if UB_new < UB:
            UB = UB_new
            planning_sol_best = deepcopy(planning_sol)
            subop_sol_best = deepcopy(subop_sol)

        # ---------------------------------------------------------
        # Generate Bender's cuts
        # ---------------------------------------------------------

        update_planning_problem_multi_cuts(
            planning_problem,
            subop_sol,
            planning_sol,
            linking_variables_sub,
        )

        # ---------------------------------------------------------
        # Resolve master
        # ---------------------------------------------------------

        unst_planning_sol, LB_new = solve_planning_problem(
            planning_problem,
            planning_variable_names,
            optimiser
        )

        print("NEW LB:", LB_new)

        for w in subop_sol:
            print(
                "THETA",
                w,
                "=",
                planning_problem.vTHETA[w].value,
            )

        print("OLD LB:", LB)
        print("NEW LB:", LB_new)

        for key in benders_subproblems:
            print(
                "theta",
                key,
                value(planning_problem.vTHETA[key])
            )

        LB = max(LB, LB_new)

        # ---------------------------------------------------------
        # Calculate Convergence
        # ---------------------------------------------------------

        gap = (UB - LB) / abs(LB)

        LB_hist.append(LB)
        UB_hist.append(UB)
        gap_hist.append(gap)
        cpu_time.append(time.time() - solver_start)

        logger.info(
            "Iter %d  LB %.3f  UB %.3f  Gap %.5f",
            k,
            LB,
            UB,
            gap,
        )

        # ---------------------------------------------------------
        # Termination Condition
        # ---------------------------------------------------------

        if gap <= conv_tol:

            if gap < 0:
                term_status = "NEGATIVE GAP"
                break

            if integer_routine:

                logger.info("Restoring integer variables")

                UB = math.inf

                for v in integer_vars:
                    set_integer(v)

                for v in binary_vars:
                    set_binary(v)

                planning_sol, LB = solve_planning_problem(
                    planning_problem,
                    planning_variable_names,
                    optimiser
                )

                planning_sol_best = deepcopy(planning_sol)
                integer_routine = False

            else:
                term_status = "OPTIMAL"
                break

        elif cpu_time[-1] >= max_cpu:
            term_status = "TIMELIMIT"
            break

        elif k == max_iter:
            term_status = "MAXITER"
            break

        elif UB == math.inf:
            planning_sol = deepcopy(unst_planning_sol)

        else:

            if stab_method == "int_level_set":

                if stab_dynamic and k >= 1:

                    gamma = update_stab_param(
                        gamma,
                        UB_hist[-1],
                        LB_hist[-1],
                        UB_hist[-2],
                        LB_hist[-2],
                    )

                if integer_investment and not integer_routine:

                    for v in integer_vars:
                        v.domain = pyo.Reals
                        v.fix(unst_planning_sol.values[v.name])

                    for v in binary_vars:
                        v.domain = pyo.PercentFraction
                        v.fix(unst_planning_sol.values[v.name])

                    planning_sol = solve_int_level_set_problem(
                        planning_problem,
                        planning_variable_names,
                        unst_planning_sol,
                        LB,
                        UB,
                        gamma,
                    )

                    for v in integer_vars:
                        v.unfix()
                        v.domain = pyo.Integers

                    for v in binary_vars:
                        v.unfix()
                        v.domain = pyo.Binary

                else:

                    planning_sol = solve_int_level_set_problem(
                        planning_problem,
                        planning_variable_names,
                        unst_planning_sol,
                        LB,
                        UB,
                        gamma,
                    )

            else:

                planning_sol = deepcopy(unst_planning_sol)

    return BendersResult(
        planning_problem=planning_problem,
        planning_sol=planning_sol_best,
        subop_sol=subop_sol_best,
        LB_hist=LB_hist,
        UB_hist=UB_hist,
        gap_hist=gap_hist,
        termination_status=term_status,
        cpu_time=cpu_time,
        planning_sol_hist=planning_sol_hist,
    )

def update_planning_problem_multi_cuts(
    planning_problem: pyo.ConcreteModel,
    subop_sol: dict,
    planning_sol: object,
    linking_variables_sub: dict,
):
    """
    Adds dynamic multi-cut Benders optimality constraints
    to the Pyomo planning model.
    """

    # Ensure a named ConstraintList exists on the model
    if not hasattr(planning_problem, "benders_cuts"):
        planning_problem.benders_cuts = pyo.ConstraintList()

    # One cut for each operational subproblem
    for w, sol_w in subop_sol.items():

        # Linking variables associated with this subproblem
        linking_vars_w = linking_variables_sub[w]

        # Build:
        #
        # op_cost_w +
        # sum_i lambda_w[i] * (x_i - xbar_i)
        #
        sum_terms = 0

        for i, var_name in enumerate(linking_vars_w):

            model_var = planning_problem.find_component(var_name)

            if model_var is None:
                raise KeyError(
                    f"Variable '{var_name}' not found "
                    "in the planning problem model."
                )

            sum_terms += (
                sol_w["lambda"][i]
                * (
                    model_var
                    - planning_sol.values[var_name]
                )
            )

        # Benders cut:
        #
        # theta_coeff_w * theta_w
        # >=
        # op_cost_w + lambda_w' (x - xbar)
        #
        lhs = (
            sol_w["theta_coeff"]
            * planning_problem.vTHETA[w]
        )

        rhs = (
            sol_w["op_cost"]
            + sum_terms
        )

        planning_problem.benders_cuts.add(
            lhs >= rhs
        )

        print(
            "CUT:",
            w,
            "theta_coeff =", sol_w["theta_coeff"],
            "op_cost =", sol_w["op_cost"],
            "lambda =", sol_w["lambda"],
        )

        print("Number of Benders cuts:", len(planning_problem.benders_cuts))

def update_stab_param(gamma, UB, LB, UB_old, LB_old):

    r = (UB_old - UB) / (UB_old - (LB + gamma * (UB_old - LB)))

    ap = UB_old - UB
    pp = UB_old - (LB + gamma * (UB_old - LB))

    if ap >= 0 and pp >= 0:

        if r <= 0.2:
            gamma = 0.9 - 0.5 * (0.9 - gamma)

        elif r >= 0.8:
            gamma *= 0.5

    return gamma

def compute_upper_bound(planning_sol,
                        subop_sol):

    print("here is the subop_sol: ", subop_sol)

    if any(sp["theta_coeff"] == 0 for sp in subop_sol.values()):
        return math.inf

    return (
        planning_sol.planning_cost
        + sum(sp["op_cost"] for sp in subop_sol.values())
    )
