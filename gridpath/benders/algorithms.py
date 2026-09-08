from dataclasses import dataclass
from copy import deepcopy
import math
import time
import logging
import pyomo.environ as pyo
from pyomo.environ import (
    SolverFactory,
    Constraint
)
from pyomo.core.expr.visitor import identify_variables
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
    planning_results: object

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

    planning_constraint_names = {
        component.name
        for component in planning_problem.component_objects(
            Constraint,
            active=True,
        )
    }

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
            parsed_arguments.solver,
            executable=parsed_arguments.solver_executable,
        )
    else:
        optimiser = SolverFactory(parsed_arguments.solver)

    # Mute solver output for Benders subproblem iterations.
    solver_name = parsed_arguments.solver.lower()

    if "mosek" in solver_name:
        optimiser.options["MSK_IPAR_LOG"] = 0

    elif "gurobi" in solver_name:
        optimiser.options["OutputFlag"] = 0

    # ---------------------------------------------------------
    # Initially solve planning problem
    # ---------------------------------------------------------

    planning_results = None

    planning_sol, master_obj, planning_results = solve_planning_problem(
        planning_problem,
        planning_variable_names,
        optimiser
    )

    UB = master_obj
    LB = -math.inf

    LB_hist = [LB]
    UB_hist = [math.inf]
    gap_hist = [math.inf]
    cpu_time = [time.time() - solver_start]

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
                planning_constraint_names,
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
    investment_components = {}

    for key in benders_subproblems:

        result = output_queues[key].get()

        if result["status"] != "READY":
            raise RuntimeError(
                f"Worker {key} failed to initialise."
            )

        linking_variables_sub[key] = result["linking_vars"]
        investment_components[key] = result["investment_components"]

    for k in range(max_iter + 1):

        # planning_sol_hist.append(
        #     [
        #         planning_sol.values[n]
        #         for n in planning_variable_names
        #     ]
        # )

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
        all_linking_diagnostics = {}
        all_ccgt_fd_diagnostics = {}

        for key in benders_subproblems:
            result_key, result, linking_diagnostics, ccgt_fd_diagnostic = (
                output_queues[key].get()
            )

            subop_sol[result_key] = result

            all_linking_diagnostics[result_key] = (
                linking_diagnostics
            )

            all_ccgt_fd_diagnostics[result_key] = (
                ccgt_fd_diagnostic
            )

        logger.info(
            "Subproblem solve took %.2f seconds",
            time.time() - solver_start,
        )

        # ---------------------------------------------------------
        # Update upper bound
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # Evaluate feasible incumbent
        # ---------------------------------------------------------

        LB_new = compute_incumbent_npv(
            planning_sol,
            subop_sol,
        )

        # print(
        #     f"\nITER {k}: "
        #     f"planning_cost={planning_sol.planning_cost:.15g}, "
        #     f"LB_new={LB_new:.15g}"
        # )

        # for key, sp in subop_sol.items():
        #     print(
        #         f"  {key}: "
        #         f"op_cost={sp['op_cost']:.15g}, "
        #         f"theta_coeff={sp['theta_coeff']}"
        #     )

        # A feasible solution gives a LOWER bound for maximization
        if LB_new > LB:
            LB = LB_new
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

        unst_planning_sol, UB_new,planning_results = solve_planning_problem(
            planning_problem,
            planning_variable_names,
            optimiser
        )

        # print("NEW LB:", LB_new)
        #
        # # for w in subop_sol:
        # #     print(
        # #         "THETA",
        # #         w,
        # #         "=",
        # #         planning_problem.vTHETA[w].value,
        # #     )
        #
        # print("OLD LB:", LB)
        # print("NEW LB:", LB_new)

        # for key in benders_subproblems:
        #     print(
        #         "theta",
        #         key,
        #         value(planning_problem.vTHETA[key])
        #     )

        print(
            f"LB DEBUG: old={LB:.6e}, "
            f"new={LB_new:.6e}, "
            f"max={max(LB, LB_new):.6e}"
        )

        # LB = LB_new

        # Master objective is an UPPER bound for maximization
        UB = min(UB, UB_new)

        # ---------------------------------------------------------
        # Calculate Convergence
        # ---------------------------------------------------------
        # Try this fix if there are no planning variables. I think it should lead to the same thing.
        if UB != math.inf:
            gap = abs(UB - LB) / max(abs(UB), 1e-10)
        else:
            gap = math.inf

        if LB > UB:
            logger.warning(
                "LB exceeds UB: LB=%.6f, UB=%.6f",
                LB,
                UB,
            )

            print(f"LB exceeds UB: LB={LB:.6f}, UB={UB:.6f}")

        LB_hist.append(LB)
        UB_hist.append(UB)
        gap_hist.append(gap)
        cpu_time.append(time.time() - solver_start)

        planning_npv = pyo.value(planning_problem.ePlanningNPV)

        theta = sum(
            pyo.value(planning_problem.vTHETA[w])
            for w in planning_problem.vTHETA
        )

        master_objective = pyo.value(planning_problem.MasterObjective)

        operational_cost = sum(
            sp["op_cost"] for sp in subop_sol.values()
        )

        npv = pyo.value(planning_problem.NPV.expr)

        print_convergence_graph(
            LB_hist,
            UB_hist,
            gap_hist,
            investment_components,
            linking_diagnostics=all_linking_diagnostics,
            subop_sol=subop_sol,
            planning_sol=planning_sol,
            unst_planning_sol=unst_planning_sol,
            linking_variables_sub=linking_variables_sub,
            planning_problem=planning_problem,
            planning_cost=planning_npv,
            operational_cost=operational_cost,
            theta=theta,
            master_objective=master_objective,
            npv=npv,
            ccgt_fd_diagnostics=all_ccgt_fd_diagnostics,
            planning_sol_hist=planning_sol_hist,
            gamma=gamma,
            planning_variable_names=planning_variable_names,
        )

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

                planning_sol, LB, planning_results = solve_planning_problem(
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
                        optimiser
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
                        optimiser
                    )

            else:

                planning_sol = deepcopy(unst_planning_sol)

    # Stop all workers
    for key in workers:
        input_queues[key].put("STOP")

    # Wait for all workers to finish
    for key in workers:
        workers[key].join()

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
        planning_results=planning_results,
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

        # print(
        #     "CUT:",
        #     w,
        #     "theta_coeff =", sol_w["theta_coeff"],
        #     "op_cost =", sol_w["op_cost"],
        #     "lambda =", sol_w["lambda"],
        # )

        # print("Number of Benders cuts:", len(planning_problem.benders_cuts))

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

def compute_incumbent_npv(planning_sol,
                        subop_sol):

    # print("here is the subop_sol: ", subop_sol)

    if any(sp["theta_coeff"] == 0 for sp in subop_sol.values()):
        return math.inf

    return (
        planning_sol.planning_cost
        - sum(sp["op_cost"] for sp in subop_sol.values())
    )


def print_convergence_graph(
    LB_hist,
    UB_hist,
    gap_hist,
    investment_components,
    linking_diagnostics,
    subop_sol,
    planning_sol,
    unst_planning_sol,
    linking_variables_sub,
    planning_problem,
    planning_cost=None,
    operational_cost=None,
    theta=None,
    master_objective=None,
    npv=None,
    ccgt_fd_diagnostics=None,
    planning_sol_hist=None,
    gamma=None,
    planning_variable_names=None,
    width=70,
    height=12,
    history_window=20,
):
    """
    Display Benders convergence directly in the Windows
    Command Prompt using ASCII/Unicode characters.

    Only the most recent `history_window` iterations are plotted,
    while the full histories are retained by the Benders algorithm.
    """

    import os
    import math
    import logging

    logger = logging.getLogger(__name__)

    if not LB_hist:
        return

    # Clear terminal
    os.system("cls")

    iteration = len(LB_hist) - 1

    current_lb = LB_hist[-1]
    current_ub = UB_hist[-1]
    current_gap = gap_hist[-1]

    planning_npv = pyo.value(planning_problem.ePlanningNPV)
    theta = sum(
        pyo.value(planning_problem.vTHETA[w])
        for w in planning_problem.vTHETA
    )
    master_objective = pyo.value(planning_problem.MasterObjective)

    print("=" * (width + 18))
    print(" BENDERS DECOMPOSITION CONVERGENCE")
    print("=" * (width + 18))

    print(f" Iteration : {iteration}")
    print()

    print(" MASTER")

    if master_objective is not None:
        print(f"   Master objective     : {master_objective:,.6g}")

    print()

    print(" SUBPROBLEMS")

    if operational_cost is not None:
        print(f"   Operational cost     : {operational_cost:,.6g}")

    if npv is not None:
        print()
        print(" ORIGINAL MODEL")
        print(f"   NPV                  : {npv:,.6g}")

    print("CONVERGENCE CRITERIA")
    print()
    print(f"   LB                   : {current_lb:,.6g}")
    print(
        f"   UB                   : "
        f"{'∞' if current_ub == math.inf else f'{current_ub:,.6g}'}"
    )
    if math.isfinite(current_gap):
        print(f" Gap                    : {current_gap:.6e}")
    else:
        print(" Gap                    : ∞")

    print()
    # ---------------------------------------------------------
    # Log detailed investment-component diagnostics
    # ---------------------------------------------------------

    logger.info("")
    logger.info("OPERATIONAL MODEL INVESTMENT COMPONENTS")

    for key, constraints in investment_components.items():

        logger.info(f"\nSubproblem: {key}")

        if constraints:
            for name in constraints:
                logger.info(f"    {name}")
        else:
            logger.info("    None")


    # =========================================================
    # SELECT PLOTTING WINDOW
    # =========================================================

    start = max(0, len(LB_hist) - history_window)

    LB_plot = LB_hist[start:]
    UB_plot = UB_hist[start:]
    gap_plot = gap_hist[start:]

    plot_start_iteration = start
    plot_end_iteration = iteration

    # =========================================================
    # OBJECTIVE BOUNDS
    # =========================================================

    print(" OBJECTIVE BOUNDS")
    print()

    finite_values = [
        x
        for x in LB_plot + UB_plot
        if math.isfinite(x)
    ]

    if finite_values:

        # ---------------------------------------------------------
        # Determine y-axis range
        # ---------------------------------------------------------

        ymin = min(finite_values)
        ymax = max(finite_values)

        if ymax == ymin:
            padding = max(abs(ymax) * 0.01, 1.0)
        else:
            padding = (ymax - ymin) * 0.05

        ymin -= padding
        ymax += padding

        # ---------------------------------------------------------
        # Force zero to be included in the range
        # ---------------------------------------------------------

        if ymin < 0 < ymax:
            # Make the range symmetric around zero.
            max_abs = max(abs(ymin), abs(ymax))

            ymin = -max_abs
            ymax = max_abs

        # -----------------------------------------------------
        # Map y value to terminal row
        # -----------------------------------------------------

        def y_to_row(value):

            return int(
                round(
                    (ymax - value)
                    / max(ymax - ymin, 1e-300)
                    * (height - 1)
                )
            )

        # -----------------------------------------------------
        # Map iteration to terminal column
        # -----------------------------------------------------

        n = len(LB_plot)

        def x_to_col(i):

            if n <= 1:
                return width // 2

            return int(
                i / (n - 1)
                * (width - 1)
            )

        # -----------------------------------------------------
        # Create graph
        # -----------------------------------------------------

        graph = [
            [" " for _ in range(width)]
            for _ in range(height)
        ]

        # -----------------------------------------------------
        # Draw line between two points
        # -----------------------------------------------------

        def draw_segment(
            x1,
            y1,
            x2,
            y2,
            symbol="·",
        ):

            steps = max(
                abs(x2 - x1),
                abs(y2 - y1),
            )

            for j in range(steps + 1):

                t = j / max(steps, 1)

                x = round(
                    x1 + t * (x2 - x1)
                )

                y = round(
                    y1 + t * (y2 - y1)
                )

                if (
                    0 <= x < width
                    and 0 <= y < height
                    and graph[y][x] == " "
                ):
                    graph[y][x] = symbol

        # -----------------------------------------------------
        # Draw LB line
        # -----------------------------------------------------

        for i in range(1, n):

            if (
                math.isfinite(LB_plot[i - 1])
                and math.isfinite(LB_plot[i])
            ):

                draw_segment(
                    x_to_col(i - 1),
                    y_to_row(LB_plot[i - 1]),
                    x_to_col(i),
                    y_to_row(LB_plot[i]),
                )

        # -----------------------------------------------------
        # Draw UB line
        # -----------------------------------------------------

        for i in range(1, n):

            if (
                math.isfinite(UB_plot[i - 1])
                and math.isfinite(UB_plot[i])
            ):

                draw_segment(
                    x_to_col(i - 1),
                    y_to_row(UB_plot[i - 1]),
                    x_to_col(i),
                    y_to_row(UB_plot[i]),
                )

        # -----------------------------------------------------
        # Draw LB observations
        # -----------------------------------------------------

        for i, value in enumerate(LB_plot):

            if math.isfinite(value):

                x = x_to_col(i)
                y = y_to_row(value)

                if (
                    0 <= x < width
                    and 0 <= y < height
                ):
                    graph[y][x] = "L"

        # -----------------------------------------------------
        # Draw UB observations
        # -----------------------------------------------------

        for i, value in enumerate(UB_plot):

            if math.isfinite(value):

                x = x_to_col(i)
                y = y_to_row(value)

                if (
                    0 <= x < width
                    and 0 <= y < height
                ):

                    if graph[y][x] == "L":
                        graph[y][x] = "◆"
                    else:
                        graph[y][x] = "U"

        # -----------------------------------------------------
        # Print graph
        # -----------------------------------------------------

        for row in range(height):

            y_value = (
                ymax
                - row / max(height - 1, 1)
                * (ymax - ymin)
            )

            print(
                f"{y_value:>12,.5g} │"
                + "".join(graph[row])
            )

        print(
            " " * 13
            + "└"
            + "─" * width
        )

        # -----------------------------------------------------
        # X axis
        # -----------------------------------------------------

        if plot_end_iteration > plot_start_iteration:

            print(
                " " * 14
                + str(plot_start_iteration)
                + " " * max(
                    width - len(str(plot_start_iteration))
                    - len(str(plot_end_iteration)),
                    1
                )
                + str(plot_end_iteration)
            )
        else:

            print(
                " " * 14
                + str(plot_start_iteration)
            )

        print()
        print(
            "   L = Lower bound"
            "    U = Upper bound"
            "    ◆ = LB = UB"
        )

    # =========================================================
    # GAP
    # =========================================================

    print()
    print(" OPTIMALITY GAP")
    print()

    # ---------------------------------------------------------
    # Check whether the current gap is exactly zero
    # ---------------------------------------------------------

    if (
        math.isfinite(current_gap)
        and current_gap == 0
    ):

        print("        CONVERGED: LB = UB")
        print()

    else:

        positive_gaps = [
            g
            for g in gap_plot
            if math.isfinite(g) and g > 0
        ]

        if positive_gaps:

            # -------------------------------------------------
            # Log10 transformation
            # -------------------------------------------------

            log_gaps = [
                math.log10(g)
                if math.isfinite(g) and g > 0
                else None
                for g in gap_plot
            ]

            finite_log_gaps = [
                g
                for g in log_gaps
                if g is not None
            ]

            log_min = min(finite_log_gaps)
            log_max = max(finite_log_gaps)

            if log_min == log_max:
                log_min -= 1

            n = len(gap_plot)

            def gap_to_row(value):

                log_value = math.log10(value)

                return int(
                    round(
                        (log_max - log_value)
                        / max(log_max - log_min, 1e-300)
                        * (height - 1)
                    )
                )

            def x_to_col(i):

                if n <= 1:
                    return width // 2

                return int(
                    i / (n - 1)
                    * (width - 1)
                )

            graph = [
                [" " for _ in range(width)]
                for _ in range(height)
            ]

            # -------------------------------------------------
            # Draw gap line
            # -------------------------------------------------

            for i in range(1, n):

                if (
                    log_gaps[i - 1] is not None
                    and log_gaps[i] is not None
                ):

                    x1 = x_to_col(i - 1)
                    y1 = gap_to_row(gap_plot[i - 1])

                    x2 = x_to_col(i)
                    y2 = gap_to_row(gap_plot[i])

                    steps = max(
                        abs(x2 - x1),
                        abs(y2 - y1),
                    )

                    for j in range(steps + 1):

                        t = j / max(steps, 1)

                        x = round(
                            x1 + t * (x2 - x1)
                        )

                        y = round(
                            y1 + t * (y2 - y1)
                        )

                        if (
                            0 <= x < width
                            and 0 <= y < height
                            and graph[y][x] == " "
                        ):
                            graph[y][x] = "·"

            # -------------------------------------------------
            # Draw gap observations
            # -------------------------------------------------

            for i, value in enumerate(gap_plot):

                if (
                    math.isfinite(value)
                    and value > 0
                ):

                    x = x_to_col(i)
                    y = gap_to_row(value)

                    if (
                        0 <= x < width
                        and 0 <= y < height
                    ):
                        graph[y][x] = "●"

            # -------------------------------------------------
            # Print gap graph
            # -------------------------------------------------

            for row in range(height):

                exponent = (
                    log_max
                    - row / max(height - 1, 1)
                    * (log_max - log_min)
                )

                print(
                    f"  10^{exponent:>5.1f} │"
                    + "".join(graph[row])
                )

            print(
                " " * 15
                + "└"
                + "─" * width
            )

            if plot_end_iteration > plot_start_iteration:

                print(
                    " " * 16
                    + str(plot_start_iteration)
                    + " " * max(
                        width
                        - len(str(plot_start_iteration))
                        - len(str(plot_end_iteration)),
                        1
                    )
                    + str(plot_end_iteration)
                )

        else:

            print("        No positive finite gaps to plot.")

    print()
    print("=" * (width + 18))