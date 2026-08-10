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

    """
    Get the subproblems set up using parallelisation.
    """
    print("Building subproblems for Benders decomposition...")
    # Note that the subproblem argument is now
    # an abstractmodel().
    print("This is now the planning problem: ", planning_problem)
    # At this point there should be only an operational abstract model waiting to be instantiated.

    # TODO: This needs to be altered slightly so that policy constraints have slacks by subperiod added to them.
    expect_feasible = setup["ExpectFeasibleSubproblems"]

    print(
        "BEFORE POOL:",
        type(planning_problem),
        isinstance(planning_problem, dict)
    )

    pool_data = [
        # subproblems is just a single abstract model
        # representative of operational constraints
        (key, scenario_directory,dynamic_components, multi_stage,expect_feasible,parsed_arguments)
        for key in benders_subproblems
    ]

    from gridpath.run_scenario import create_operational_problem_benders_pool
    # We will now instantiate the subproblems here to easily revert to the alternative
    # option (which would be to instantiate each subproblem in the same routine.
    print("Generating operational Benders subproblems...")
    # This should now be a dictionary of all operational concrete models.
    # Pool must use spawn to work properly on Linux

    n_tasks = len(benders_subproblems)
    # Should be the number of subproblems including stochastic iterations.
    print("# of tasks: ", n_tasks)

    pool = get_context("spawn").Pool(n_tasks)
    subproblems = dict(pool.map(create_operational_problem_benders_pool,pool_data))

    """
    Now start the Benders algorithm.
    """

    print("Running Benders decomposition...")
    logger.info("Running Benders decomposition")
    # If we expect the subproblem to be feasible, we can skip adding slacks to the subproblem variables.
    add_approximate_variable_cost(
        planning_problem,
        len(linking_variables_sub),
    )

    solver_start = time.time()

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

    for k in range(max_iter + 1):

        planning_sol_hist.append(
            [planning_sol.values[n] for n in planning_variable_names]
        )

        start = time.time()

        # Now it's time to parallelise. Each Benders iteration will be solved separately.
        pool_data_subproblem = [
            (
                key,
                model,
                planning_sol,
                linking_variables_sub,
                expect_feasible,
                optimiser,
            )
            for key, model in subproblems.items()
        ]

        subop_sol = pool.map(
            parallelise_solve,
            pool_data_subproblem
        )

        logger.info(
            "Subproblem solve took %.2f seconds",
            time.time() - start,
        )

        UB_new = compute_upper_bound(
            planning_problem,
            planning_sol,
            subop_sol,
        )

        if UB_new < UB:
            UB = UB_new
            planning_sol_best = deepcopy(planning_sol)
            subop_sol_best = deepcopy(subop_sol)

        update_planning_problem_multi_cuts(
            planning_problem,
            subop_sol,
            planning_sol,
            linking_variables_sub,
        )

        unst_planning_sol, LB_new = solve_planning_problem(
            planning_problem,
            planning_variable_names,
            optimiser
        )

        LB = max(LB, LB_new)

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

        #
        # termination
        #

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
        linking_variables_sub: dict
):
    """
    Adds dynamic multi-cut Benders optimality constraints to the Pyomo planning model.
    """
    # 1. Ensure a named ConstraintList exists on the model to hold our cuts
    if not hasattr(planning_problem, "benders_cuts"):
        planning_problem.benders_cuts = pyo.ConstraintList()

    W = subop_sol.keys()

    # 2. Re-create the Julia comprehension mapping over all scenarios w in W
    for w in W:
        sol_w = subop_sol[w]

        # Build the dynamic sum() expression term
        sum_terms = 0
        for i in range(len(linking_variables_sub)):
            var_name = linking_variables_sub[i]

            # Find the actual Pyomo variable object inside the model by its string name
            model_var = planning_problem.find_component(var_name)

            if model_var is None:
                raise KeyError(f"Variable '{var_name}' not found in the planning problem model.")

            # Accumulate: lambda * (x_model - x_planning)
            sum_terms += sol_w.lambda_[i] * (model_var - planning_sol.values[var_name])

        # Left-hand side: subop_sol[w].theta_coeff * planning_problem.vTHETA[w]
        # (Assumes your model initialization script created 'vTHETA' as an indexed Var)
        lhs = sol_w.theta_coeff * planning_problem.vTHETA[w]
        rhs = sol_w.op_cost + sum_terms

        # 3. Append the greater-than-or-equal-to constraint directly to the model
        planning_problem.benders_cuts.add(lhs >= rhs)

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

def compute_upper_bound(model, planning_sol, subop_sol):
    print("here is the subop_sol: ", subop_sol)

    if isinstance(subop_sol, dict):
        subop_sol = [subop_sol]

    if any(sp["theta_coeff"] == 0 for sp in subop_sol):
        return math.inf

    # for w, sp in subop_sol.items():
    #
    #     expr = (
    #         sp.op_cost
    #         + quicksum(
    #             sp.lambda_[i]
    #             * (
    #                 model.getVarByName(linking_variables_sub[w][i])
    #                 - planning_sol.values[linking_variables_sub[w][i]]
    #             )
    #             for i in range(len(linking_variables_sub[w]))
    #         )
    #     )
    #
    #     model.addConstr(
    #         sp.theta_coeff * theta[w] >= expr
    #     )

    return (
        planning_sol.planning_cost
        + sum(sp["op_cost"] for sp in subop_sol)
    )
