from dataclasses import dataclass
from copy import deepcopy
import math
import time
import logging
import pyomo.environ as pyo
from pyomo.environ import (
    SolverFactory,
    Constraint,
    Integers,
    Binary,
)
from pyomo.opt import TerminationCondition
import multiprocessing as mp
from pyomo.core.expr.visitor import identify_variables
from pyomo.repn.standard_repn import generate_standard_repn
# from gridpath.benders.planning import add_approximate_variable_cost, solve_planning_problem
# from gridpath.benders.regularization import solve_int_level_set_problem
# from gridpath.benders.subproblems import solve_subproblem
from gridpath.benders.planning import *
from gridpath.benders.regularization import *
from gridpath.benders.subproblems import (
    benders_subproblem_worker,
)
from gridpath.benders.gurobi_benders import (
    create_gurobi_benders_solver,
)
from gridpath.benders.mosek_benders import (
    create_mosek_benders_solver,
)

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

# ---------------------------------------------------------------------------
# GridPath constraints that can provide implied upper bounds for
# continuous linking variables.
# ---------------------------------------------------------------------------

CAPACITY_BOUND_CONSTRAINTS = (
    "Max_Build_Power_Constraint",
    "Max_Power_Constraint",
    "Max_Build_Stor_Energy_Constraint",
    "Max_Stor_Energy_Constraint",
    "Max_Build_Energy_Constraint",
    "Max_Energy_Constraint",
    "StorNewLin_Max_Duration_Constraint",
    "TxNewLin_Max_Cum_Build_Constraint"
)

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
    linking_vars,
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

    benders_states = {}
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

        binary_vars = [v for v in planning_variables if v.is_binary()]
        integer_vars = [
            v for v in planning_variables
            if v.is_integer() and not v.is_binary()
        ]

        for v in integer_vars:
            v.domain = pyo.Reals

        for v in binary_vars:
            v.domain = pyo.UnitInterval

        integer_routine = True

    if "mosek" in parsed_arguments.solver.lower():

        optimiser = create_mosek_benders_solver()

    elif "gurobi" in parsed_arguments.solver.lower():

        optimiser = create_gurobi_benders_solver()

    else:

        if parsed_arguments.solver_executable is not None:

            optimiser = SolverFactory(
                parsed_arguments.solver,
                executable=parsed_arguments.solver_executable,
            )

        else:

            optimiser = SolverFactory(
                parsed_arguments.solver
            )

    # Mute solver output for Benders subproblem iterations.
    solver_name = parsed_arguments.solver.lower()

    if "mosek" in solver_name:
        optimiser.options["MSK_IPAR_LOG"] = "0"

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
    UB_hist = [UB]
    gap_hist = [math.inf]
    cpu_time = [time.time() - solver_start]

    planning_sol_best = None
    subop_sol_best = None

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

    linking_variables_sub = {}
    linking_variables_sub_by_type = {}
    investment_components = {}
    recourse_lower_bounds = {}

    for key in benders_subproblems:
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()

        process = ctx.Process(
            target=benders_subproblem_worker,
            args=(
                key,
                planning_constraint_names,
                linking_vars,
                input_queue,
                output_queue,
                scenario_directory,
                multi_stage,
                expect_feasible,
                parsed_arguments,
            ),
        )

        process.start()

        workers[key] = process
        input_queues[key] = input_queue
        output_queues[key] = output_queue

    # Wait for all workers to initialise
    for key in benders_subproblems:

        result = output_queues[key].get()

        if result["status"] != "READY":
            raise RuntimeError(
                f"Worker {key} failed to initialise."
            )

        linking_variables_sub[key] = result["linking_vars"]

        linking_variables_sub_by_type[key] = (
            result["linking_vars_by_type"]
        )

        investment_components[key] = result["investment_components"]

        recourse_lower_bounds[key] = (result["recourse_lower_bound"])

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

        for key in benders_subproblems:

            result_key, result = (
                output_queues[key].get()
            )

            subop_sol[result_key] = result

            register_benders_state(
                benders_states,
                result_key,
                result,
            )

            if not result["feasible"]:
                binary_vars = (
                    linking_variables_sub_by_type[result_key]["binary"]
                )

                integer_vars = (
                    linking_variables_sub_by_type[result_key]["integer"]
                )

                add_combinatorial_feasibility_cut(
                    planning_problem=planning_problem,
                    binary_linking_vars=binary_vars,
                    integer_linking_vars=integer_vars,
                    planning_sol=planning_sol,
                    cut_id=f"{result_key}_{k}"
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

        use_native_mosek = (
                "mosek" in parsed_arguments.solver.lower()
        )

        update_planning_problem_multi_cuts(
            planning_problem=planning_problem,
            subop_sol=subop_sol,
            planning_sol=planning_sol,
            linking_variables_sub_by_type=linking_variables_sub_by_type,
            benders_states=benders_states,
            cut_id=k,
            use_native_mosek=use_native_mosek
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
            subop_sol=subop_sol,
            planning_sol=planning_sol,
            unst_planning_sol=unst_planning_sol,
            planning_problem=planning_problem,
            planning_cost=planning_npv,
            operational_cost=operational_cost,
            theta=theta,
            master_objective=master_objective,
            npv=npv,
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

            if integer_routine:

                logger.info("Restoring integer variables")

                for v in integer_vars:
                    v.domain = Integers

                for v in binary_vars:
                    v.domain = Binary

                planning_sol, UB, planning_results = solve_planning_problem(
                    planning_problem,
                    planning_variable_names,
                    optimiser,
                )

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

                planning_sol = solve_int_level_set_problem(
                    planning_problem,
                    planning_variable_names,
                    unst_planning_sol,
                    LB,
                    UB,
                    gamma,
                    optimiser,
                )
            else:
                planning_sol = deepcopy(unst_planning_sol)

        planning_sol_hist.append(
            [
                planning_sol.values[name]
                for name in planning_variable_names
            ]
        )

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


def register_benders_state(
    benders_states,
    subproblem_key,
    result,
):
    state = result["discrete_state"]
    state_key = (subproblem_key, state)

    if state_key not in benders_states:
        benders_states[state_key] = {
            "feasible": result["feasible"],
            "cuts": [],
        }

    if result["feasible"]:
        # print(
        #     "\nREGISTERING FEASIBLE BENDERS CUT"
        # )
        # print(
        #     f"  subproblem = {subproblem_key}"
        # )
        # print(
        #     f"  discrete_state = {result['discrete_state']}"
        # )
        # print(
        #     f"  op_cost = {result['op_cost']}"
        # )
        # print(
        #     f"  theta_coeff = {result['theta_coeff']}"
        # )
        #
        theta_coeff = result.get("theta_coeff", 1.0)

        if abs(theta_coeff) < 1e-12:
            raise RuntimeError(
                "Feasible Benders result has theta_coeff=0. "
                f"Subproblem={subproblem_key}, "
                f"discrete_state={result['discrete_state']}, "
                f"op_cost={result['op_cost']}"
            )

        benders_states[state_key]["cuts"].append({
            "op_cost": result["op_cost"],
            "continuous_lambda": dict(
                result["continuous_lambda"]
            ),
            "continuous_point": dict(
                result["continuous_point"]
            ),
            "theta_coeff": theta_coeff,
        })

def update_planning_problem_multi_cuts(
    planning_problem,
    subop_sol,
    planning_sol,
    linking_variables_sub_by_type,
    benders_states,
    cut_id,
    use_native_mosek=False,
):
    """
    Add all currently known Benders cuts.

    MOSEK:
        Cuts are stored for later insertion as native disjunctive
        constraints (DJC).

    Gurobi:
        Cuts are stored for later insertion as native indicator
        constraints.

    No explicit Big-M is used in either implementation.
    """

    if use_native_mosek:

        planning_problem._benders_native_djc_data = {
            "benders_states": benders_states,
            "linking_variables_sub_by_type": (
                linking_variables_sub_by_type
            ),
        }

        return

    # ------------------------------------------------------------
    # Gurobi native indicator formulation
    # ------------------------------------------------------------

    planning_problem._benders_native_indicator_data = {
        "benders_states": benders_states,
        "linking_variables_sub_by_type": (
            linking_variables_sub_by_type
        ),
    }

def add_state_recourse_cut(
    planning_problem,
    subproblem_key,
    state,
    continuous_vars,
    discrete_vars,
    continuous_lambda,
    continuous_point,
    op_cost,
    theta_coeff,
    cut_id,
):
    """
    Add a state-conditioned Benders optimality cut for the
    Gurobi/Pyomo formulation.

    The cut

        theta_coeff * theta
        - op_cost
        - sum(lambda_i * (x_i - xbar_i))
        >= 0

    is enforced only when the discrete linking variables equal
    the incumbent discrete state.

    A binary mismatch variable z is used:

        z = 0  <=>  discrete variables equal the incumbent state
        z = 1  <=>  discrete variables differ

    and the Benders cut is relaxed with Big-M:

        cut_expression + M*z >= 0

    Continuous linking variables therefore require finite bounds
    for the Gurobi implementation.
    """

    block_name = f"BendersStateCut_{cut_id}"

    # Avoid illegal Pyomo component names.
    block_name = (
        block_name
        .replace(",", "_")
        .replace(" ", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("'", "")
    )

    block = pyo.Block()
    setattr(planning_problem, block_name, block)

    # ------------------------------------------------------------
    # 1. State mismatch indicator
    # ------------------------------------------------------------

    block.z = pyo.Var(
        domain=pyo.Binary
    )

    mismatch_vars = []

    # ------------------------------------------------------------
    # Binary linking variables
    # ------------------------------------------------------------

    for i, var_name in enumerate(discrete_vars["binary"]):
        var = planning_problem.find_component(var_name)

        if var is None:
            raise KeyError(
                f"Could not find binary linking variable "
                f"'{var_name}' in planning problem."
            )

        target = int(round(state[i]))

        mismatch = pyo.Var(domain=pyo.Binary)
        setattr(
            block,
            f"binary_mismatch_{i}",
            mismatch,
        )

        mismatch_vars.append(mismatch)

        if target == 1:
            # mismatch = 1 when var = 0
            block.add_component(
                f"binary_mismatch_constraint_{i}",
                pyo.Constraint(
                    expr=mismatch >= 1 - var
                ),
            )

            block.add_component(
                f"binary_match_constraint_{i}",
                pyo.Constraint(
                    expr=mismatch <= 1 - var
                ),
            )

        elif target == 0:
            # mismatch = 1 when var = 1
            block.add_component(
                f"binary_mismatch_constraint_{i}",
                pyo.Constraint(
                    expr=mismatch >= var
                ),
            )

            block.add_component(
                f"binary_match_constraint_{i}",
                pyo.Constraint(
                    expr=mismatch <= var
                ),
            )

        else:
            raise ValueError(
                f"Binary state value for '{var_name}' "
                f"must be 0 or 1, got {target}."
            )

    # ------------------------------------------------------------
    # Integer linking variables
    # ------------------------------------------------------------

    integer_state_offset = len(
        discrete_vars["binary"]
    )

    for j, var_name in enumerate(discrete_vars["integer"]):
        var = planning_problem.find_component(var_name)

        if var is None:
            raise KeyError(
                f"Could not find integer linking variable "
                f"'{var_name}' in planning problem."
            )

        target = int(round(
            state[integer_state_offset + j]
        ))

        lb = var.lb
        ub = var.ub

        if lb is None or ub is None:
            raise ValueError(
                f"Integer linking variable '{var_name}' "
                f"requires finite bounds for the Gurobi "
                f"combinatorial cut. "
                f"Found lb={lb}, ub={ub}."
            )

        lb = int(lb)
        ub = int(ub)

        if target < lb or target > ub:
            raise ValueError(
                f"Incumbent value {target} for '{var_name}' "
                f"is outside its bounds [{lb}, {ub}]."
            )

        below = pyo.Var(domain=pyo.Binary)
        above = pyo.Var(domain=pyo.Binary)

        setattr(
            block,
            f"integer_below_{j}",
            below,
        )
        setattr(
            block,
            f"integer_above_{j}",
            above,
        )

        mismatch = pyo.Var(domain=pyo.Binary)

        setattr(
            block,
            f"integer_mismatch_{j}",
            mismatch,
        )

        mismatch_vars.append(mismatch)

        # var <= target - 1 if below = 1
        #
        # var <= target - 1 + M * (1-below)
        block.add_component(
            f"integer_below_constraint_{j}",
            pyo.Constraint(
                expr=var
                <= target - 1
                + (ub - target + 1) * (1 - below)
            ),
        )

        # var >= target + 1 if above = 1
        #
        # var >= target + 1 - M * (1-above)
        block.add_component(
            f"integer_above_constraint_{j}",
            pyo.Constraint(
                expr=var
                >= target + 1
                - (target - lb + 1) * (1 - above)
            ),
        )

        # At most one direction can be active.
        block.add_component(
            f"integer_direction_constraint_{j}",
            pyo.Constraint(
                expr=below + above == mismatch
            ),
        )

        # If var differs from target, one of the directions
        # must be active.
        block.add_component(
            f"integer_mismatch_lower_{j}",
            pyo.Constraint(
                expr=var - target
                <= (ub - target) * mismatch
            ),
        )

        block.add_component(
            f"integer_mismatch_upper_{j}",
            pyo.Constraint(
                expr=target - var
                <= (target - lb) * mismatch
            ),
        )

    # ------------------------------------------------------------
    # Global mismatch indicator
    # ------------------------------------------------------------

    if mismatch_vars:

        # z >= every individual mismatch
        for i, mismatch in enumerate(mismatch_vars):
            block.add_component(
                f"z_lower_{i}",
                pyo.Constraint(
                    expr=block.z >= mismatch
                ),
            )

        # z <= sum of individual mismatches
        block.z_upper = pyo.Constraint(
            expr=block.z <= sum(mismatch_vars)
        )

    else:
        # No discrete variables: the cut is globally valid.
        block.z_fix = pyo.Constraint(
            expr=block.z == 0
        )

    # ------------------------------------------------------------
    # 2. Construct the continuous Benders expression
    # ------------------------------------------------------------

    theta = planning_problem.vTHETA[subproblem_key]

    cut_expression = (
        theta_coeff * theta
        - op_cost
    )

    for name in continuous_vars:
        var = planning_problem.find_component(name)

        if var is None:
            raise KeyError(
                f"Could not find continuous linking variable "
                f"'{name}' in planning problem."
            )

        lam = float(continuous_lambda[name])
        xbar = float(continuous_point[name])

        cut_expression += (
            -lam * (var - xbar)
        )

    # ------------------------------------------------------------
    # 3. Compute a valid Big-M for the continuous cut
    # ------------------------------------------------------------

    if mismatch_vars:
        M = continuous_dual_term_upper_bound(
            planning_problem=planning_problem,
            continuous_vars=continuous_vars,
            continuous_lambda=continuous_lambda,
            continuous_point=continuous_point,
            op_cost=op_cost,
            theta=theta,
            theta_coeff=theta_coeff,
        )
    else:
        M = 0.0

    # ------------------------------------------------------------
    # 4. Add conditional Benders cut
    # ------------------------------------------------------------

    #
    # z = 0:
    #
    #     cut_expression >= 0
    #
    # z = 1:
    #
    #     cut_expression + M >= 0
    #
    # so M must be large enough to deactivate the cut.
    #

    block.benders_cut = pyo.Constraint(
        expr=cut_expression + M * block.z >= 0
    )

    return block

def add_discrete_pattern_indicator(
    planning_problem,
    binary_linking_vars,
    integer_linking_vars,
    planning_sol,
    cut_id,
):
    """
    Create a binary indicator for the incumbent discrete pattern.

    delta = 0 is feasible only when all discrete linking
    variables equal their incumbent values.

    If any discrete linking variable differs from its incumbent,
    delta = 1 is required.
    """

    block = pyo.Block(concrete=True)

    block.delta = pyo.Var(
        domain=pyo.Binary
    )

    block.binary_pattern_constraints = pyo.ConstraintList()
    block.integer_pattern_constraints = pyo.ConstraintList()

    add_binary_pattern_constraints(
        block,
        planning_problem,
        binary_linking_vars,
        planning_sol,
    )

    add_integer_pattern_constraints(
        block,
        planning_problem,
        integer_linking_vars,
        planning_sol,
    )

    component_name = f"BendersDiscretePattern_{cut_id}"

    planning_problem.add_component(
        component_name,
        block,
    )

    return block

def get_continuous_variable_bounds(
    planning_problem,
    var,
    optimiser,
):
    if not hasattr(
        planning_problem,
        "_benders_variable_bounds",
    ):
        planning_problem._benders_variable_bounds = {}

    cache = planning_problem._benders_variable_bounds

    if var.name in cache:
        return cache[var.name]

    lb = pyo.value(var.lb) if var.has_lb() else None
    ub = pyo.value(var.ub) if var.has_ub() else None

    # ---------------------------------------------------------
    # 1. Explicit Pyomo bounds
    # ---------------------------------------------------------

    if ub is not None:

        if lb is None:
            if var.is_nonnegative():
                lb = 0.0
            else:
                raise ValueError(
                    f"No finite lower bound found for "
                    f"continuous linking variable '{var.name}'."
                )

        cache[var.name] = (lb, ub)
        return lb, ub

    # ---------------------------------------------------------
    # 2. GridPath capacity/potential constraints
    # ---------------------------------------------------------

    ub = find_gridpath_capacity_upper_bound(
        planning_problem,
        var,
    )

    target_var = planning_problem.find_component(
        "StorNewLin_Build_MW[Battery,2030]"
    )

    print("\n" + "=" * 80)
    print("ALL CONSTRAINTS INVOLVING BATTERY MW")
    print("=" * 80)

    for constraint in planning_problem.component_data_objects(
            pyo.Constraint,
            active=True,
            descend_into=True,
    ):
        variables = identify_variables(
            constraint.body,
            include_fixed=True,
        )

        if any(v is target_var for v in variables):
            print("\nConstraint:", constraint.name)
            print("  body :", constraint.body)
            print("  lower:", constraint.lower)
            print("  upper:", constraint.upper)

    if ub is not None:

        if lb is None:
            if var.is_nonnegative():
                lb = 0.0
            else:
                raise ValueError(
                    f"No finite lower bound found for "
                    f"continuous linking variable '{var.name}'."
                )

        cache[var.name] = (lb, ub)
        return lb, ub

    from pyomo.gdp import Disjunct, Disjunction
    from pyomo.core import TransformationFactory

    model = pyo.ConcreteModel()

    model.x = pyo.Var(bounds=(0, 10))
    model.theta = pyo.Var()

    # Binary variable representing the discrete pattern.
    model.y = pyo.Var(domain=Binary)

    # Case 1: y = 1 -> Benders cut is active
    model.match = Disjunct()

    model.match.y_constraint = Constraint(
        expr=model.y == 1
    )

    model.match.benders_cut = Constraint(
        expr=model.theta >= 100 + 5 * model.x
    )

    # Case 2: y = 0 -> no Benders cut
    model.no_match = Disjunct()

    model.no_match.y_constraint = Constraint(
        expr=model.y == 0
    )

    model.disjunction = Disjunction(
        expr=[model.match, model.no_match]
    )

    model.obj = pyo.Objective(
        expr=model.theta,
        sense=pyo.minimize,
    )

    solver = SolverFactory("mosek")

    print("Solver available:", solver.available())

    # First test: ordinary GDP Big-M transformation.
    TransformationFactory("gdp.bigm").apply_to(model)

    results = solver.solve(model, tee=True)

    print(results.solver.termination_condition)
    print("x =", value(model.x))
    print("y =", value(model.y))
    print("theta =", value(model.theta))

    # ---------------------------------------------------------
    # 3. OBBT fallback
    # ---------------------------------------------------------

    print(
        f"\n[Benders] No explicit/GridPath upper bound found "
        f"for {var.name}."
    )
    print(
        f"[Benders] Running OBBT for {var.name}..."
    )

    ub = obbt_upper_bound(
        planning_problem,
        var,
        optimiser,
    )

    if ub is None:
        raise ValueError(
            f"Unable to determine a finite upper bound for "
            f"continuous linking variable '{var.name}'."
        )

    if lb is None:
        if var.is_nonnegative():
            lb = 0.0
        else:
            raise ValueError(
                f"No finite lower bound found for "
                f"continuous linking variable '{var.name}'."
            )

    print(
        f"[Benders] OBBT upper bound for {var.name}: {ub}"
    )

    cache[var.name] = (lb, ub)

    return lb, ub

def obbt_upper_bound(
    planning_problem,
    target_var,
    optimiser,
):
    """
    Compute an upper bound for target_var using OBBT.

    The current master objective is temporarily replaced with
    an objective maximizing target_var. Integer and binary
    variables are temporarily relaxed so that we can determine
    whether the continuous relaxation has a finite upper bound.
    The original objective and variable domains are restored
    afterward.
    """

    if target_var.fixed:
        return pyo.value(target_var)

    active_objectives = list(
        planning_problem.component_data_objects(
            pyo.Objective,
            active=True,
            descend_into=True,
        )
    )

    if len(active_objectives) != 1:
        raise RuntimeError(
            "OBBT expected exactly one active planning "
            f"objective, found {len(active_objectives)}."
        )

    objective = active_objectives[0]

    original_expr = objective.expr
    original_sense = objective.sense

    integer_vars = []

    try:

        # -----------------------------------------------------
        # Maximize the target variable.
        # -----------------------------------------------------

        objective.set_value(target_var)
        objective.sense = pyo.maximize

        # -----------------------------------------------------
        # Temporarily relax integer/binary variables.
        # -----------------------------------------------------

        for var in planning_problem.component_data_objects(
            pyo.Var,
            active=True,
            descend_into=True,
        ):
            if var.is_binary() or var.is_integer():

                original_domain = var.domain

                if var.is_binary():
                    var.domain = pyo.UnitInterval
                else:
                    var.domain = pyo.Reals

                integer_vars.append(
                    (var, original_domain)
                )

        # -----------------------------------------------------
        # Solve relaxed OBBT problem.
        # -----------------------------------------------------

        results = optimiser.solve(
            planning_problem,
            tee=True,
        )

        termination = (
            results.solver.termination_condition
        )

        print("\n" + "=" * 80)
        print("OBBT DIAGNOSTIC")
        print("=" * 80)
        print("Variable:", target_var.name)
        print("Termination:", termination)
        print("Solver status:", results.solver.status)

        if target_var.value is not None:
            print(
                "Variable value:",
                pyo.value(target_var),
            )
        else:
            print("Variable value: None")

        print("Variable LB:", target_var.lb)
        print("Variable UB:", target_var.ub)

        if termination != TerminationCondition.optimal:

            print(
                f"[Benders] OBBT for {target_var.name} "
                "failed to obtain an optimal bound. "
                f"Termination condition: {termination}"
            )

            return None

        ub = pyo.value(target_var)

        return ub

    finally:

        # -----------------------------------------------------
        # Restore original objective.
        # -----------------------------------------------------

        objective.set_value(original_expr)
        objective.sense = original_sense

        # -----------------------------------------------------
        # Restore original integer/binary domains.
        # -----------------------------------------------------

        for var, original_domain in integer_vars:
            var.domain = original_domain

def find_gridpath_capacity_upper_bound(
    planning_problem,
    target_var,
):
    """
    Search GridPath's capacity/potential constraints for a finite
    upper bound on target_var.

    Returns None if no valid bound can be identified.
    """

    best_ub = None
    best_source = None

    for constraint_name in CAPACITY_BOUND_CONSTRAINTS:

        constraint_component = planning_problem.find_component(
            constraint_name
        )

        if constraint_component is None:
            continue

        for constraint in constraint_component.values():

            if not constraint.active:
                continue

            repn = generate_standard_repn(constraint.body)

            if not repn.is_linear():
                continue

            # Check whether target_var occurs in this constraint.
            target_indices = [
                i
                for i, var in enumerate(repn.linear_vars)
                if var is target_var
            ]

            if not target_indices:
                continue

            # ----------------------------------------------------
            # We only use constraints that provide an upper bound.
            # ----------------------------------------------------

            if not constraint.has_ub():
                continue

            # There should normally be one occurrence of a variable
            # in these capacity constraints.
            #
            # If it occurs multiple times, aggregate its coefficient.
            target_coef = sum(
                repn.linear_coefs[i]
                for i in target_indices
            )

            if target_coef <= 0:
                continue

            # ----------------------------------------------------
            # Calculate the contribution of all OTHER variables.
            #
            # For a safe bound, every other variable must have a
            # known lower bound because we're calculating:
            #
            #   a*x + sum(a_j*x_j) <= RHS
            #
            # so:
            #
            #   x <= (RHS - constant - min(other terms)) / a
            # ----------------------------------------------------

            other_term_lower_bound = (
                repn.constant if repn.constant is not None else 0.0
            )

            valid = True

            for i, other_var in enumerate(repn.linear_vars):

                if other_var is target_var:
                    continue

                coef = repn.linear_coefs[i]

                if coef >= 0:

                    if other_var.has_lb():
                        other_lb = pyo.value(other_var.lb)
                    elif other_var.is_nonnegative():
                        other_lb = 0.0
                    else:
                        valid = False
                        break

                    other_term_lower_bound += coef * other_lb

                else:

                    # For a negative coefficient, the minimum
                    # contribution requires an upper bound.
                    if other_var.has_ub():
                        other_ub = pyo.value(other_var.ub)
                    else:
                        valid = False
                        break

                    other_term_lower_bound += coef * other_ub

            if not valid:
                continue

            rhs = pyo.value(constraint.upper)

            candidate_ub = (
                rhs - other_term_lower_bound
            ) / target_coef

            if candidate_ub < 0:
                continue

            if best_ub is None or candidate_ub < best_ub:
                best_ub = candidate_ub
                best_source = constraint.name

    if best_ub is not None:
        print(
            f"[Benders] Derived upper bound for {target_var.name}: "
            f"{best_ub} from {best_source}"
        )

    return best_ub

def continuous_dual_term_upper_bound(
    planning_problem,
    continuous_linking_vars,
    continuous_lambda,
    planning_sol,
    optimiser
):
    """
    Upper bound on

        sum_i lambda_i * (x_i - xbar_i)

    using finite bounds on the master linking variables.

    Bounds are obtained in the following order:

        1. Explicit Pyomo variable bounds.
        2. GridPath capacity/potential constraints.
        3. OBBT.

    The bounds must be valid over the planning problem's
    feasible region because they are used to construct the
    Big-M for the conditional Benders cut.
    """

    bound = 0.0

    for name in continuous_linking_vars:

        var = planning_problem.find_component(name)

        if var is None:
            raise KeyError(
                f"Continuous linking variable '{name}' "
                "not found in planning problem."
            )

        lb, ub = get_continuous_variable_bounds(
            planning_problem,
            var,
            optimiser
        )

        xbar = planning_sol.values[name]
        lam = continuous_lambda[name]

        if lam >= 0:
            bound += lam * (ub - xbar)
        else:
            bound += lam * (lb - xbar)

    return bound

def add_binary_pattern_constraints(
    block,
    planning_problem,
    binary_linking_vars,
    planning_sol,
):
    """
    delta = 0  => all binary linking variables equal
                 their incumbent values.

    delta = 1  => binary pattern constraint is relaxed.
    """

    for name in binary_linking_vars:

        var = planning_problem.find_component(name)

        if var is None:
            raise KeyError(
                f"Binary linking variable '{name}' not found "
                "in planning problem."
            )

        incumbent = planning_sol.values[name]

        if abs(incumbent - 1) < 1e-6:
            block.binary_pattern_constraints.add(
                var >= 1 - block.delta
            )

        elif abs(incumbent) < 1e-6:
            block.binary_pattern_constraints.add(
                var <= block.delta
            )

        else:
            raise ValueError(
                f"Expected binary incumbent for '{name}', "
                f"got {incumbent}."
            )

def add_integer_pattern_constraints(
    block,
    planning_problem,
    integer_linking_vars,
    planning_sol,
):
    """
    delta = 0  => all integer linking variables equal
                 their incumbent values.

    delta = 1  => integer variables retain only their
                  original bounds.
    """

    for name in integer_linking_vars:

        var = planning_problem.find_component(name)

        if var is None:
            raise KeyError(
                f"Integer linking variable '{name}' not found "
                "in planning problem."
            )

        incumbent = planning_sol.values[name]

        lb = var.lb
        ub = var.ub

        if lb is None or ub is None:
            raise ValueError(
                f"Integer linking variable '{name}' must have "
                "finite bounds for combinatorial Benders cuts. "
                f"Current bounds: ({lb}, {ub})."
            )

        if incumbent < lb or incumbent > ub:
            raise ValueError(
                f"Incumbent value {incumbent} for '{name}' "
                f"is outside its bounds ({lb}, {ub})."
            )

        block.integer_pattern_constraints.add(
            var - incumbent
            <= (ub - incumbent) * block.delta
        )

        block.integer_pattern_constraints.add(
            incumbent - var
            <= (incumbent - lb) * block.delta
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

def compute_incumbent_npv(planning_sol,
                        subop_sol):

    # print("here is the subop_sol: ", subop_sol)

    if any(sp["theta_coeff"] == 0 for sp in subop_sol.values()):
        return -math.inf

    return (
        planning_sol.planning_cost
        - sum(sp["op_cost"] for sp in subop_sol.values())
    )


def print_convergence_graph(
    LB_hist,
    UB_hist,
    gap_hist,
    investment_components,
    subop_sol,
    planning_sol,
    unst_planning_sol,
    planning_problem,
    planning_cost=None,
    operational_cost=None,
    theta=None,
    master_objective=None,
    npv=None,
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
    # os.system("cls")

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