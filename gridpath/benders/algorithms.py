from dataclasses import dataclass
from copy import deepcopy
import math
import time
import logging
from gridpath.benders.planning import add_approximate_variable_cost, solve_planning_problem
from gridpath.benders.regularization import solve_int_level_set_problem
from gridpath.benders.subproblems import solve_subproblem

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


def benders(
    planning_problem,
    subproblems,
    linking_variables_sub,
    setup,
    reformat_logging=False,
):
    """
    Regularized Benders decomposition.
    """

    logger.info("Running Benders decomposition")

    expect_feasible = setup["ExpectFeasibleSubproblems"]

    if expect_feasible:
        logger.info("Skipping feasibility cuts.")
    else:
        add_slacks_to_subproblems(subproblems)

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
            unset_integer(v)

        for v in binary_vars:
            unset_binary(v)

        integer_routine = True

    planning_sol, LB = solve_planning_problem(
        planning_problem,
        planning_variable_names,
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

        subop_sol = solve_subproblem(
            subproblems,
            planning_sol,
            expect_feasible
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
                        unset_integer(v)
                        fix(v, unst_planning_sol.values[v.name])

                    for v in binary_vars:
                        unset_binary(v)
                        fix(v, unst_planning_sol.values[v.name])

                    planning_sol = solve_int_level_set_problem(
                        planning_problem,
                        planning_variable_names,
                        unst_planning_sol,
                        LB,
                        UB,
                        gamma,
                    )

                    for v in integer_vars:
                        unfix(v)
                        set_integer(v)

                    for v in binary_vars:
                        unfix(v)
                        set_binary(v)

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

    if any(sp.theta_coeff == 0 for sp in subop_sol.values()):
        return math.inf

    return (
        planning_sol.planning_cost
        + sum(sp.op_cost for sp in subop_sol.values())
    )

    for w, sp in subop_sol.items():

        expr = (
            sp.op_cost
            + quicksum(
                sp.lambda_[i]
                * (
                    model.getVarByName(linking_variables_sub[w][i])
                    - planning_sol.values[linking_variables_sub[w][i]]
                )
                for i in range(len(linking_variables_sub[w]))
            )
        )

        model.addConstr(
            sp.theta_coeff * theta[w] >= expr
        )