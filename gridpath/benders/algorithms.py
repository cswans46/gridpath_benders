from dataclasses import dataclass
from copy import deepcopy
import os
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
    "TxNewLin_Max_Cum_Build_Constraint",
)

def all_variables(model) -> list:
    """Return all variable data objects in the Pyomo model."""
    return list(model.component_data_objects(pyo.Var, descend_into=True))

def _gap_stalled(gap_hist, n_iter, gap_floor):
    """True if the gap is below gap_floor and has not improved for n_iter"""
    if len(gap_hist) <= n_iter:
        return False
    recent = gap_hist[-n_iter:]
    best_before = min(gap_hist[:-n_iter])
    return (
        all(math.isfinite(g) for g in recent)
        and min(recent) <= gap_floor
        and min(recent) >= best_before * (1 - 1e-9)
    )

def _get_from_worker(queue, process, key):
    """Blocking queue read that notices a dead worker and stays interruptible."""
    import queue as queue_module

    while True:
        try:
            return queue.get(timeout=1.0)
        except queue_module.Empty:
            if not process.is_alive():
                try:
                    return queue.get(timeout=1.0)
                except queue_module.Empty:
                    raise RuntimeError(
                        f"Benders worker {key} died "
                        f"(exit code {process.exitcode}); see its traceback above."
                    )

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
    """Run Benders decomposition over the planning master and recourse workers.

    The routine initializes the master, starts one persistent worker per
    operational subproblem, and iterates between recourse solves, cut
    generation, bound updates, and a new master solve. With ``pj_relax``,
    it first solves a continuous master relaxation, then restores master
    integrality and continues with discrete-state cuts. Stabilization only
    selects the next trial point; it does not change the master cuts.
    """
    print("Running Benders decomposition...")

    solver_start = time.time()
    # If we expect the subproblem to be feasible, we can skip adding slacks to the subproblem variables.
    add_approximate_variable_cost(
        planning_problem,
        benders_subproblems
    )

    max_iter = setup["MaxIter"]
    conv_tol = setup["ConvTol"]
    abs_gap_tol = setup.get("AbsGapTol", math.inf)
    max_cpu = setup["MaxCpuTime"]
    gamma = setup["StabParam"]
    stab_dynamic = setup["StabDynamic"]

    master_discrete_method = setup.get(
        "MasterDiscreteMethod",
        "pj_relax",
    )

    operational_discrete_method = setup.get(
        "OperationalDiscreteMethod",
        "pj_relax",
    )

    stabilization_method = setup.get(
        "StabilizationMethod",
        "none",
    )

    valid_master_discrete_methods = {
        "none",
        "pj_relax",
        "linearized",
        "djc",
    }

    valid_operational_discrete_methods = {
        "pj_relax",
        "fixed_z",
    }

    valid_stabilization_methods = {
        "none",
        "l1",
        "l2",
    }

    if master_discrete_method not in valid_master_discrete_methods:
        raise ValueError(
            f"Invalid MasterDiscreteMethod="
            f"{master_discrete_method!r}. "
            f"Expected one of "
            f"{sorted(valid_master_discrete_methods)}."
        )

    if operational_discrete_method not in valid_operational_discrete_methods:
        raise ValueError(
            f"Invalid OperationalDiscreteMethod="
            f"{operational_discrete_method!r}. "
            f"Expected one of "
            f"{sorted(valid_operational_discrete_methods)}."
        )

    if stabilization_method not in valid_stabilization_methods:
        raise ValueError(
            f"Invalid StabilizationMethod="
            f"{stabilization_method!r}. "
            f"Expected one of "
            f"{sorted(valid_stabilization_methods)}."
        )

    if (
            stabilization_method in {"l1", "l2"}
            and abs(gamma) < 1e-12
    ):
        raise ValueError(
            f"StabilizationMethod={stabilization_method!r} "
            "requires a nonzero StabParam."
        )

    term_status = "NONE"

    # linearized runs the global P&J relaxed phase and then adds big-M activation
    # cuts. djc is a separate routine: a pure MIP master with native MOSEK DJCs,
    # seeded with one cut in every binary state.
    requested_master_method = master_discrete_method
    use_native_djc = (
        requested_master_method != "linearized"
        and "mosek" in parsed_arguments.solver.lower()
    )
    if master_discrete_method == "linearized":
        master_discrete_method = "pj_relax"

    # True means the master is currently using its original
    # integer/binary domains.
    master_integrality_restored = (
            master_discrete_method != "pj_relax"
    )

    benders_states = {}
    benders_feasibility_states = {}

    planning_constraint_names = {
        component.name
        for component in planning_problem.component_objects(
            Constraint,
            active=True,
        )
    }

    # ---------------------------------------------------------
    # 2. Identify master variables
    # ---------------------------------------------------------

    planning_variables = all_variables(planning_problem)

    planning_variable_names = [
        v.name
        for v in planning_variables
    ]

    binary_vars = [
        v for v in planning_variables
        if v.is_binary()
    ]

    integer_vars = [
        v for v in planning_variables
        if v.is_integer() and not v.is_binary()
    ]

    # "none" assumes continuous linking variables. If discrete ones exist,
    # fall back to P&J (relaxation, then state cuts) instead of failing.
    if master_discrete_method == "none":
        _discrete_linking = {
            var.name
            for var in binary_vars + integer_vars
            if var.name in set(linking_vars)
        }
        if _discrete_linking:
            logger.warning(
                "MasterDiscreteMethod='none' but %d discrete linking "
                "variable(s) exist; falling back to 'pj_relax'.",
                len(_discrete_linking),
            )
            master_discrete_method = "pj_relax"
            use_native_djc = "mosek" in parsed_arguments.solver.lower()

    # ---------------------------------------------------------
    # Apply P&J relaxation to the master
    # ---------------------------------------------------------

    if master_discrete_method == "pj_relax":

        for v in integer_vars:
            v.domain = pyo.Reals

        for v in binary_vars:
            v.domain = pyo.UnitInterval

        master_integrality_restored = False

    elif master_discrete_method == "none":

        linking_variable_names = set(linking_vars)
        discrete_linking_vars = {
            var.name
            for var in binary_vars + integer_vars
            if var.name in linking_variable_names
        }
        if discrete_linking_vars:
            raise ValueError(
                "MasterDiscreteMethod='none' requires continuous "
                "linking variables. Discrete linking variables found: "
                f"{sorted(discrete_linking_vars)[:20]}"
            )

        master_integrality_restored = True

    elif master_discrete_method in {
        "linearized",
        "djc",
    }:
        master_integrality_restored = False

    else:
        raise ValueError(
            f"Unknown BendersDiscreteMethod: "
            f"{master_discrete_method!r}."
        )

    # This calls on mosek benders solver to implement
    # The correct solver for the task (this depends on the stabilisation
    # and binary method requested).

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
    # Create and Solve Subproblem in assigning processors
    # ---------------------------------------------------------

    print("Building subproblems for Benders decomposition...")

    ctx = mp.get_context("spawn")

    workers = {}
    input_queues = {}
    output_queues = {}

    linking_variables_sub = {}
    linking_variables_sub_by_type = {}
    recourse_cost_lower_bounds = {}
    worker_keys = list(benders_subproblems)
    continuous_linking_variable_names = []
    seen_continuous = set()

    if (
        requested_master_method == "djc"
        and "mosek" not in parsed_arguments.solver.lower()
    ):
        raise ValueError(
            "BendersDiscreteMethod='djc' requires the MOSEK "
            "Benders solver."
        )

    try:
        worker_batch_size = max(
            1,
            int(parsed_arguments.n_parallel_solve),
        )
    except (AttributeError, TypeError, ValueError):
        worker_batch_size = 1

    for key in worker_keys:
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
                parsed_arguments
            ),
        )

        process.daemon = True
        process.start()

        workers[key] = process
        input_queues[key] = input_queue
        output_queues[key] = output_queue

        # Each worker solves an LP upper bound for operational NPV before READY.
        # Wait here so startup cannot consume one solver license per
        # subproblem simultaneously.
        result = _get_from_worker(output_queues[key], workers[key], key)

        if result["status"] != "READY":
            raise RuntimeError(
                f"Worker {key} failed to initialise."
            )

        linking_variables_sub[key] = (
            result["present_linking_vars"]
        )
        linking_variables_sub_by_type[key] = (
            result["linking_vars_by_type"]
        )

        recourse_cost_lower_bounds[key] = -float(
            result["recourse_upper_bound"]
        )

        if requested_master_method in ("djc", "linearized") and (
            requested_master_method == "djc" or use_native_djc
        ):
            integer_linking_vars = (
                linking_variables_sub_by_type[key]["integer"]
            )

            if integer_linking_vars:
                raise NotImplementedError(
                    "Native MOSEK DJC currently supports binary "
                    "linking variables only. "
                    f"Subproblem '{key}' contains integer "
                    f"linking variables: {integer_linking_vars}"
                )

        for name in linking_variables_sub_by_type[key]["continuous"]:
            if name not in seen_continuous:
                continuous_linking_variable_names.append(name)
                seen_continuous.add(name)

    # A maximized operational NPV has an LP-relaxation upper bound. Its
    # negative is a valid lower bound for the net-operating-cost theta.
    set_recourse_cost_lower_bounds(
        planning_problem,
        recourse_cost_lower_bounds,
    )

    # ---------------------------------------------------------
    # Initially solve planning problem
    # ---------------------------------------------------------

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

    stabilization_variables = [
        v
        for v in planning_variables
        if v.name in seen_continuous
    ]
    relaxed_recourse_samples = {
        key: []
        for key in worker_keys
    }

    # DJC seeding: cuts are valid only in the state where they are generated,
    # so unvisited states are bounded only by the loose global recourse bound.
    # Evaluate every joint binary state once (continuous linking variables at
    # their lower bound) so each state starts with a cut.
    djc_seeds = []
    if requested_master_method == "djc" and setup.get("DjcSeedStates", False):
        seed_names = sorted(
            {
                name
                for key in worker_keys
                for name in linking_variables_sub_by_type[key]["binary"]
            }
        )
        seed_max = int(setup.get("DjcSeedMaxStates", 1024))
        if seed_names and 2 ** len(seed_names) <= seed_max:
            for code in range(2 ** len(seed_names)):
                seed_sol = deepcopy(planning_sol)
                for name in seen_continuous:
                    if name in seed_sol.values:
                        seed_sol.values[name] = 0.0
                for bit, name in enumerate(seed_names):
                    seed_sol.values[name] = float((code >> bit) & 1)
                djc_seeds.append(seed_sol)
            logger.info(
                "DJC: seeding %d binary states over %d binary linking "
                "variables.", len(djc_seeds), len(seed_names),
            )
        elif seed_names:
            logger.warning(
                "DJC: %d binary linking variables exceed DjcSeedMaxStates; "
                "no state seeding.", len(seed_names),
            )
    n_seeds = len(djc_seeds)
    if djc_seeds:
        planning_sol = djc_seeds.pop(0)
    evaluating_seed = bool(n_seeds)

    for k in range(max_iter + 1 + n_seeds):
        # ---------------------------------------------------------
        # Send current planning solution to every operational worker
        # ---------------------------------------------------------

        subop_sol = {}

        for batch_start in range(
            0,
            len(worker_keys),
            worker_batch_size,
        ):
            worker_batch = worker_keys[
                batch_start:batch_start + worker_batch_size
            ]

            for key in worker_batch:
                input_queues[key].put(
                    {
                        "planning_sol": planning_sol,
                        "master_discrete_method": master_discrete_method,
                        "operational_discrete_method": (
                            operational_discrete_method
                        ),
                        "master_integrality_restored": (
                            master_integrality_restored
                        ),
                    }
                )

            for key in worker_batch:
                result_key, result = _get_from_worker(output_queues[key], workers[key], key)
                subop_sol[result_key] = result

        # ---------------------------------------------------------
        # Determine Benders phases (from setup dictionary)
        # ---------------------------------------------------------
        # Master-discrete handling concerns investment/linking variables;
        # OperationalDiscreteMethod independently controls recourse vars.

        has_discrete_linking_vars = any(
            linking_variables_sub_by_type[w]["binary"]
            or linking_variables_sub_by_type[w]["integer"]
            for w in linking_variables_sub_by_type
        )

        pj_relaxed_phase = (
                master_discrete_method == "pj_relax"
                and not master_integrality_restored
        )

        pj_discrete_phase = (
                master_discrete_method == "pj_relax"
                and master_integrality_restored
                and has_discrete_linking_vars
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

        # ---------------------------------------------------------
        # Process subproblem results
        # ---------------------------------------------------------

        for result_key, result in subop_sol.items():

            # -----------------------------------------------------
            # Feasibility cuts
            # -----------------------------------------------------

            add_benders_feasibility_cut(
                planning_problem=planning_problem,
                result_key=result_key,
                result=result,
                linking_variables_sub_by_type=(
                    linking_variables_sub_by_type
                ),
                planning_sol=planning_sol,
                master_discrete_method=master_discrete_method,
                master_integrality_restored=(
                    master_integrality_restored
                ),
                cut_id=k,
                benders_feasibility_states=benders_feasibility_states,
                native_state_cuts=(
                    master_discrete_method == "pj_relax"
                    and master_integrality_restored
                    and use_native_djc
                ),
            )

            # -----------------------------------------------------
            # P&J relaxed phase
            #
            # Discrete master variables have been relaxed to
            # continuous variables.
            #
            # Therefore we use ordinary Benders cuts over the
            # relaxed linking-variable space.
            # -----------------------------------------------------

            if pj_relaxed_phase:

                if not result["feasible"]:
                    for previous_sample in relaxed_recourse_samples[
                        result_key
                    ]:
                        cut_at_sample = float(result["op_cost"])
                        cut_scale = max(
                            1.0,
                            abs(float(result["op_cost"])),
                            abs(float(previous_sample["op_cost"])),
                        )

                        for name, dual in result[
                            "continuous_lambda"
                        ].items():
                            previous_point = float(
                                previous_sample["point"][name]
                            )
                            current_point = float(
                                result["continuous_point"][name]
                            )
                            term = float(dual) * (
                                previous_point - current_point
                            )
                            cut_at_sample += term
                            cut_scale += abs(term)

                        violation = max(0.0, cut_at_sample)
                        tolerance = min(
                            abs_gap_tol,
                            5e-4 * cut_scale,
                        )
                        if violation > tolerance:
                            raise RuntimeError(
                                "Relaxed Benders feasibility cut excludes "
                                "a previously feasible LP sample. "
                                f"Subproblem: {result_key}; "
                                f"cut violation: {violation:.12g}; "
                                f"tolerance: {tolerance:.12g}; "
                                f"restoration slack: "
                                f"{result['op_cost']:.12g}."
                            )
                add_relaxed_benders_cut(
                    planning_problem=planning_problem,
                    subproblem_key=result_key,
                    linking_vars=(
                        result["continuous_lambda"].keys()
                    ),
                    benders_lambda=(
                        result["continuous_lambda"]
                    ),
                    benders_point=(
                        result["continuous_point"]
                    ),
                    op_cost=float(result["op_cost"]),
                    theta_coeff=result.get(
                        "theta_coeff",
                        1.0,
                    ),
                )

                if result["feasible"]:
                    for previous_sample in relaxed_recourse_samples[
                        result_key
                    ]:
                        cut_at_sample = float(result["op_cost"]) + sum(
                            float(dual)
                            * (
                                float(previous_sample["point"][name])
                                - float(result["continuous_point"][name])
                            )
                            for name, dual in result[
                                "continuous_lambda"
                            ].items()
                        )
                        over = cut_at_sample - float(
                            previous_sample["op_cost"]
                        )
                        if over > max(
                            abs_gap_tol,
                            1e-6 * abs(float(previous_sample["op_cost"])),
                        ):
                            logging.warning(
                                "Benders cut diagnostic: cut for %s "
                                "overestimates a previous sample by %.6g "
                                "(sample recourse %.6g).",
                                result_key,
                                over,
                                float(previous_sample["op_cost"]),
                            )
                    logging.info(
                        "Benders credit linking point for %s: %s",
                        result_key,
                        {
                            k: v
                            for k, v in result["continuous_point"].items()
                            if "CARBON_CREDIT" in k
                        },
                    )
                    relaxed_recourse_samples[result_key].append(
                        {
                            "point": dict(result["continuous_point"]),
                            "op_cost": float(result["op_cost"]),
                        }
                    )

            # -----------------------------------------------------
            # P&J discrete phase
            #
            # Integer/binary master variables have now been
            # restored.
            #
            # Register the discrete state. The actual state-
            # conditioned Benders cuts are added below, after ALL
            # subproblems have been processed.
            # -----------------------------------------------------

            elif pj_discrete_phase:
                discrete_linking_vars = (
                    linking_variables_sub_by_type[result_key]["binary"]
                    + linking_variables_sub_by_type[result_key]["integer"]
                )
                if discrete_linking_vars:
                    register_benders_state(
                        benders_states=benders_states,
                        subproblem_key=result_key,
                        result=result,
                    )
                elif result["feasible"]:
                    add_relaxed_benders_cut(
                        planning_problem=planning_problem,
                        subproblem_key=result_key,
                        linking_vars=(
                            result["continuous_lambda"].keys()
                        ),
                        benders_lambda=(
                            result["continuous_lambda"]
                        ),
                        benders_point=(
                            result["continuous_point"]
                        ),
                        op_cost=result["op_cost"],
                        theta_coeff=result.get("theta_coeff", 1.0),
                    )

            # -----------------------------------------------------
            # Linearized combinatorial Benders
            # -----------------------------------------------------

            elif linearized_phase:

                register_benders_state(
                    benders_states=benders_states,
                    subproblem_key=result_key,
                    result=result,
                )

            # -----------------------------------------------------
            # Native DJC
            # -----------------------------------------------------

            elif djc_phase:

                register_benders_state(
                    benders_states=benders_states,
                    subproblem_key=result_key,
                    result=result,
                )

            # -----------------------------------------------------
            # Ordinary continuous Benders
            #
            # This occurs when there are no discrete linking
            # variables.
            # -----------------------------------------------------

            else:

                add_relaxed_benders_cut(
                    planning_problem=planning_problem,
                    subproblem_key=result_key,
                    linking_vars=(
                        result["continuous_lambda"].keys()
                    ),
                    benders_lambda=(
                        result["continuous_lambda"]
                    ),
                    benders_point=(
                        result["continuous_point"]
                    ),
                    op_cost=result["op_cost"],
                    theta_coeff=result.get(
                        "theta_coeff",
                        1.0,
                    ),
                )

        # ---------------------------------------------------------
        # Add discrete-state Benders cuts AFTER all subproblems
        # have been processed.
        # ---------------------------------------------------------

        if discrete_cut_phase:
            update_planning_problem_multi_cuts(
                planning_problem=planning_problem,
                benders_states=benders_states,
                benders_feasibility_states=(
                    benders_feasibility_states
                ),
                linking_variables_sub_by_type=(
                    linking_variables_sub_by_type
                ),
                master_discrete_method=master_discrete_method,
                master_integrality_restored=(
                    master_integrality_restored
                ),
                recourse_lower_bounds=recourse_cost_lower_bounds,
                optimiser=optimiser,
                native_djc=use_native_djc,
            )

        # ---------------------------------------------------------
        # Evaluate feasible incumbent
        # ---------------------------------------------------------

        if all(
            result["feasible"]
            for result in subop_sol.values()
        ):
            LB_new = compute_incumbent_npv(
                planning_sol,
                subop_sol,
            )

            # A complete feasible recourse evaluation gives a valid lower
            # bound for this maximization problem.
            if LB_new > LB:
                LB = LB_new
                planning_sol_best = deepcopy(planning_sol)
                subop_sol_best = deepcopy(subop_sol)


        # ---------------------------------------------------------
        # Resolve master
        # ---------------------------------------------------------

        if evaluating_seed:
            if djc_seeds:
                planning_sol = djc_seeds.pop(0)
                continue
            evaluating_seed = False

        unst_planning_sol, UB_new,planning_results = solve_planning_problem(
            planning_problem,
            planning_variable_names,
            optimiser
        )

        # Master objective is an UPPER bound for maximization
        UB = min(UB, UB_new)

        # ---------------------------------------------------------
        # Calculate Convergence
        # ---------------------------------------------------------
        # Try this fix if there are no planning variables. I think it should lead to the same thing.
        if UB != math.inf and LB != -math.inf:
            gap = abs(UB - LB) / max(abs(UB), 1e-10)
            absolute_gap = abs(UB - LB)
        else:
            gap = math.inf
            absolute_gap = math.inf

        # Convergence is governed by the relative gap (ConvTol). The absolute
        # tolerance is not required: at large objectives it is below double
        # precision. LB may exceed UB only by the relative tolerance.
        effective_abs_gap_tol = abs_gap_tol
        converged = (
            gap <= conv_tol
            and LB <= UB + conv_tol * abs(UB)
        )
        if LB > UB:
            logger.warning(
                "LB exceeds UB: LB=%.6f, UB=%.6f",
                LB,
                UB,
            )

        LB_hist.append(LB)
        UB_hist.append(UB)
        gap_hist.append(gap)
        cpu_time.append(time.time() - solver_start)

        master_objective = pyo.value(planning_problem.MasterObjective)
        operational_cost = sum(
            sp["op_cost"] for sp in subop_sol.values()
        )
        npv = pyo.value(planning_problem.NPV.expr)

        print_convergence_graph(
            LB_hist,
            UB_hist,
            gap_hist,
            operational_cost=operational_cost,
            master_objective=master_objective,
            npv=npv,
            absolute_gap=absolute_gap,
            absolute_gap_tolerance=abs_gap_tol,
        )

        # ---------------------------------------------------------
        # Termination Condition
        # ---------------------------------------------------------

        if converged:

            if (
                    master_discrete_method == "pj_relax"
                    and not master_integrality_restored
            ):

                # P&J is two-stage: once the relaxed master gap closes,
                # restore investment integrality and restart the loop so
                # workers evaluate that integer plan and add state cuts.

                logger.info(
                    "P&J relaxed master has converged. "
                    "Restoring integer and binary investment variables."
                )

                fractional = [
                    (v.name, v.value)
                    for v in integer_vars + binary_vars
                    if v.value is not None
                    and abs(v.value - round(v.value)) > 1e-6
                ]
                logger.info(
                    "P&J relaxed incumbent has %d fractional discrete "
                    "variable(s): %s",
                    len(fractional),
                    fractional[:10],
                )
                if not fractional:
                    # The relaxed optimum is already integer feasible, and
                    # workers have evaluated it, so it is optimal for the
                    # integer master as well.
                    term_status = "OPTIMAL"
                    break

                for v in integer_vars:
                    v.domain = Integers

                for v in binary_vars:
                    v.domain = Binary

                # The relaxed-phase incumbent belongs to a different
                # feasible region and is not an LB for the integer model.
                LB = -math.inf
                planning_sol_best = None
                subop_sol_best = None
                LB_hist[-1] = LB
                gap_hist[-1] = math.inf

                if hasattr(
                        planning_problem,
                        "_benders_variable_bounds"
                ):
                    planning_problem._benders_variable_bounds.clear()

                planning_sol, UB, planning_results = (
                    solve_planning_problem(
                        planning_problem,
                        planning_variable_names,
                        optimiser,
                    )
                )

                master_integrality_restored = True

                continue

            else:

                term_status = "OPTIMAL"
                break

        elif (
            master_integrality_restored
            or master_discrete_method != "pj_relax"
        ) and _gap_stalled(gap_hist, setup.get("StallIter", 20),
                           setup.get("StallGap", 1e-6)):
            logger.warning(
                "Benders gap stalled at %.3e (below StallGap) for %d "
                "iterations; this is numerical noise floor, stopping.",
                gap, setup.get("StallIter", 20),
            )
            term_status = "OPTIMAL"
            break
        elif cpu_time[-1] >= max_cpu:
            term_status = "TIMELIMIT"
            break

        elif k == max_iter + n_seeds:
            term_status = "MAXITER"
            break

        elif UB == math.inf:
            planning_sol = deepcopy(unst_planning_sol)

        else:

            use_level_set = (
                    gamma != 0
                    and LB is not None
                    and UB is not None
                    and UB > LB
            )

            # Near the noise floor the stabilised point stops improving the
            # incumbent; alternate with the plain master optimum (always valid).
            if (use_level_set and len(gap_hist) > 6 and gap < 1e-3
                    and gap_hist[-1] > 0.5 * gap_hist[-6]
                    and len(gap_hist) % 2 == 0):
                use_level_set = False

            if use_level_set:
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
                    # Stabilize from the point just evaluated by recourse;
                    # centering on the new master optimum would make its
                    # L1/L2 deviation objective zero at that same optimum.
                    planning_sol,
                    LB,
                    UB,
                    gamma,
                    optimiser,
                    stabilization_method=stabilization_method,
                    stabilization_variables=stabilization_variables,
                    stabilization_weights=None,
                    master_discrete_method=master_discrete_method,
                    fallback_planning_sol=unst_planning_sol,
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

    if planning_sol_best is not None and subop_sol_best is not None:
        for name, incumbent_value in planning_sol_best.values.items():
            var = planning_problem.find_component(name)
            if var is not None:
                var.set_value(incumbent_value, skip_validation=True)

        for subproblem_key, result in subop_sol_best.items():
            if result.get("feasible", False):
                planning_problem.vTHETA[subproblem_key].set_value(
                    result["op_cost"] / BENDERS_THETA_SCALE
                )

        planning_problem._benders_best_feasible_objective = LB
        planning_problem._benders_best_feasible_planning_cost = (
            planning_sol_best.planning_cost
        )
        planning_problem._benders_best_feasible_recourse_cost = sum(
            result["op_cost"]
            for result in subop_sol_best.values()
        )

    planning_results.solver.termination_condition = {
        "OPTIMAL": TerminationCondition.optimal,
        "TIMELIMIT": TerminationCondition.maxTimeLimit,
        "MAXITER": TerminationCondition.maxIterations,
    }.get(
        term_status,
        planning_results.solver.termination_condition,
    )

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

def add_benders_feasibility_cut(
    planning_problem,
    result_key,
    result,
    linking_variables_sub_by_type,
    planning_sol,
    master_discrete_method,
    master_integrality_restored,
    cut_id,
    benders_feasibility_states=None,
    native_state_cuts=False,
):
    """
    Add a Benders feasibility cut when appropriate.

    P&J relaxed phase:
        No discrete combinatorial feasibility cut is generated,
        because the master discrete variables are temporarily
        continuous.

    Linearized:
        Add an explicit combinatorial feasibility cut.

    DJC:
        Register a restoration-dual feasibility cut conditioned on the
        current binary state. This is not a global no-good cut because a
        different continuous-linking allocation may make the same binary
        state feasible.

    Parameters
    ----------
    planning_problem : Pyomo model
        Investment/master problem.

    result_key : hashable
        Subproblem identifier.

    result : dict
        Subproblem result.

    linking_variables_sub_by_type : dict
        Linking-variable metadata by subproblem.

    planning_sol : PlanningSolution
        Current master solution.

    master_discrete_method : str
        "pj_relax", "linearized", or "djc".

    master_integrality_restored : bool
        True during the P&J relaxed phase.

    cut_id : hashable
        Unique identifier for the feasibility cut.
    """

    # ---------------------------------------------------------
    # Only act on infeasible subproblems.
    # ---------------------------------------------------------

    if result["feasible"]:
        return False

    # ---------------------------------------------------------
    # P&J relaxed phase
    # ---------------------------------------------------------
    #
    # The current master point may contain fractional binary
    # or integer variables. Therefore we cannot interpret the
    # point as a discrete state and exclude that state.
    #
    if (
        master_discrete_method == "pj_relax"
        and not master_integrality_restored
    ):
        return False

    # ---------------------------------------------------------
    # Discrete phase
    # ---------------------------------------------------------

    binary_vars = (
        linking_variables_sub_by_type[result_key]["binary"]
    )

    integer_vars = (
        linking_variables_sub_by_type[result_key]["integer"]
    )

    if (
        master_discrete_method == "pj_relax"
        and master_integrality_restored
        and native_state_cuts
        and (binary_vars or integer_vars)
    ):
        if integer_vars:
            raise NotImplementedError(
                "The MOSEK P&J state-cut path supports binary discrete "
                "linking variables only. Integer variables found for "
                f"'{result_key}': {integer_vars}"
            )
        if benders_feasibility_states is None:
            raise RuntimeError(
                "P&J feasibility-state storage was not initialized."
            )
        if result.get("master_discrete_state") is None:
            raise ValueError(
                f"Infeasible P&J subproblem '{result_key}' returned no "
                "master discrete state."
            )
        benders_feasibility_states.setdefault(result_key, []).append(
            {
                "master_discrete_state": result["master_discrete_state"],
                "continuous_lambda": result["continuous_lambda"],
                "continuous_point": result["continuous_point"],
                "op_cost": float(result["op_cost"]),
                "theta_coeff": 0.0,
            }
        )
        return True

    if master_discrete_method == "djc":
        if integer_vars:
            raise NotImplementedError(
                "Native MOSEK Benders feasibility DJCs currently "
                "support binary linking variables only. Integer "
                f"linking variables were found for '{result_key}': "
                f"{integer_vars}"
            )
        if benders_feasibility_states is None:
            raise RuntimeError(
                "DJC feasibility-state storage was not initialized."
            )
        if result.get("master_discrete_state") is None:
            raise ValueError(
                f"Infeasible DJC subproblem '{result_key}' returned "
                "no master discrete state."
            )
        benders_feasibility_states.setdefault(result_key, []).append(
            {
                "master_discrete_state": result["master_discrete_state"],
                "continuous_lambda": result["continuous_lambda"],
                "continuous_point": result["continuous_point"],
                "op_cost": float(result["op_cost"]),
                "theta_coeff": 0.0,
            }
        )
        return True

    # If there are no discrete linking variables, a
    # combinatorial feasibility cut is not appropriate.
    #
    # In that case the infeasibility mechanism should instead
    # be handled by the existing continuous/feasibility-cut
    # machinery.
    if not binary_vars and not integer_vars:
        return False

    add_combinatorial_feasibility_cut(
        planning_problem=planning_problem,
        binary_linking_vars=binary_vars,
        integer_linking_vars=integer_vars,
        planning_sol=planning_sol,
        cut_id=f"{result_key}_{cut_id}",
    )

    return True

def add_combinatorial_feasibility_cut(
    planning_problem,
    binary_linking_vars,
    integer_linking_vars,
    planning_sol,
    cut_id,
):
    """
    Add a combinatorial no-good feasibility cut that excludes
    the current discrete linking-variable assignment.

    The cut is:

        sum(binary mismatches)
        + sum(integer mismatches)
        >= 1

    Binary variables use their incumbent value directly.

    Integer variables use auxiliary binary variables indicating
    whether the integer variable is strictly below or strictly
    above its incumbent value.

    This cut is valid when the current discrete assignment is
    known to produce an infeasible operational subproblem.
    """

    # ---------------------------------------------------------
    # Create a unique Block for this feasibility cut.
    # ---------------------------------------------------------

    block_name = (
        f"BendersFeasibilityBlock_{cut_id}"
    )

    if hasattr(planning_problem, block_name):
        raise RuntimeError(
            f"Benders feasibility block '{block_name}' "
            f"already exists."
        )

    block = pyo.Block()

    planning_problem.add_component(
        block_name,
        block,
    )

    # ---------------------------------------------------------
    # Store mismatch expressions.
    # ---------------------------------------------------------

    mismatch_terms = []

    # ---------------------------------------------------------
    # Binary linking variables
    # ---------------------------------------------------------

    for name in binary_linking_vars:

        var = planning_problem.find_component(name)

        if var is None:
            raise KeyError(
                f"Could not find binary linking variable "
                f"'{name}' in the planning problem."
            )

        if name not in planning_sol.values:
            raise KeyError(
                f"Binary linking variable '{name}' is missing "
                f"from the planning solution."
            )

        xbar = float(
            planning_sol.values[name]
        )

        # Binary variables should be 0 or 1.
        if abs(xbar) < 1e-8:
            xbar = 0.0
        elif abs(xbar - 1.0) < 1e-8:
            xbar = 1.0
        else:
            raise ValueError(
                f"Binary linking variable '{name}' has "
                f"non-binary incumbent value {xbar}."
            )

        if xbar == 1.0:
            # Mismatch occurs when x = 0.
            mismatch_terms.append(
                1.0 - var
            )

        else:
            # Mismatch occurs when x = 1.
            mismatch_terms.append(
                var
            )

    # ---------------------------------------------------------
    # Integer linking variables
    # ---------------------------------------------------------

    integer_mismatch_vars = {}

    for name in integer_linking_vars:

        var = planning_problem.find_component(name)

        if var is None:
            raise KeyError(
                f"Could not find integer linking variable "
                f"'{name}' in the planning problem."
            )

        if name not in planning_sol.values:
            raise KeyError(
                f"Integer linking variable '{name}' is missing "
                f"from the planning solution."
            )

        xbar = float(
            planning_sol.values[name]
        )

        # The incumbent must be integral because this function
        # is only called in the discrete phase.
        xbar_int = int(round(xbar))

        if abs(xbar - xbar_int) > 1e-8:
            raise ValueError(
                f"Integer linking variable '{name}' has "
                f"non-integer incumbent value {xbar}."
            )

        # -----------------------------------------------------
        # Integer variable bounds are required.
        # -----------------------------------------------------

        if var.lb is None or var.ub is None:
            raise ValueError(
                f"Integer linking variable '{name}' must have "
                f"finite lower and upper bounds to construct "
                f"a combinatorial feasibility cut."
            )

        lb = int(round(float(var.lb)))
        ub = int(round(float(var.ub)))

        if xbar_int < lb or xbar_int > ub:
            raise ValueError(
                f"Incumbent value {xbar_int} for integer "
                f"variable '{name}' lies outside its bounds "
                f"[{lb}, {ub}]."
            )

        # -----------------------------------------------------
        # Below / above mismatch indicators
        # -----------------------------------------------------

        below_name = (
            f"integer_below_{len(integer_mismatch_vars)}"
        )

        above_name = (
            f"integer_above_{len(integer_mismatch_vars)}"
        )

        below = pyo.Var(
            domain=pyo.Binary
        )

        above = pyo.Var(
            domain=pyo.Binary
        )

        setattr(
            block,
            below_name,
            below,
        )

        setattr(
            block,
            above_name,
            above,
        )

        integer_mismatch_vars[name] = (
            below,
            above,
        )

        # -----------------------------------------------------
        # below = 1  => x <= xbar - 1
        # below = 0  => x >= xbar
        # -----------------------------------------------------

        M_below = xbar_int - lb
        M_not_below = ub - xbar_int + 1

        block.add_component(
            f"{below_name}_lower",
            pyo.Constraint(
                expr=(
                    var
                    >=
                    xbar_int
                    - M_below * below
                )
            )
        )

        block.add_component(
            f"{below_name}_upper",
            pyo.Constraint(
                expr=(
                    var
                    <=
                    xbar_int - 1
                    + M_not_below * (1 - below)
                )
            )
        )

        # -----------------------------------------------------
        # above = 1  => x >= xbar + 1
        # above = 0  => x <= xbar
        # -----------------------------------------------------

        M_above = ub - xbar_int
        M_not_above = xbar_int - lb + 1

        block.add_component(
            f"{above_name}_upper",
            pyo.Constraint(
                expr=(
                    var
                    <=
                    xbar_int
                    + M_above * above
                )
            )
        )

        block.add_component(
            f"{above_name}_lower",
            pyo.Constraint(
                expr=(
                    var
                    >=
                    xbar_int + 1
                    - M_not_above * (1 - above)
                )
            )
        )

        # If x is different from xbar, one of these will be 1.
        mismatch_terms.append(
            below + above
        )

    # ---------------------------------------------------------
    # Make sure there is something to exclude.
    # ---------------------------------------------------------

    if not mismatch_terms:
        return False

    # ---------------------------------------------------------
    # No-good cut
    # ---------------------------------------------------------
    #
    # At least one discrete linking variable must differ from
    # the infeasible incumbent pattern.
    #

    feasibility_cut = sum(
        mismatch_terms
    ) >= 1

    # ---------------------------------------------------------
    # Also register the cut in the global feasibility list.
    # ---------------------------------------------------------

    planning_problem.BendersFeasibilityCuts.add(
        feasibility_cut
    )

    return True

def register_benders_state(
    benders_states,
    subproblem_key,
    result,
):
    """
    Register a Benders state for a discrete-state cut.

    This function should only be called when the master is operating
    with its discrete linking variables in their original domains.
    """

    if not result.get("feasible", False):
        return

    op_cost = result.get("op_cost")
    theta_coeff = result.get("theta_coeff", 1.0)

    if op_cost is None:
        raise ValueError(
            f"Feasible subproblem '{subproblem_key}' returned no "
            "net operational cost."
        )

    theta_coeff = float(theta_coeff)

    if abs(theta_coeff) <= 1e-12:
        raise ValueError(
            f"Feasible subproblem '{subproblem_key}' returned "
            f"theta_coeff={theta_coeff}. A feasible recourse cut "
            "requires a non-zero theta coefficient."
        )

    continuous_lambda = result.get("continuous_lambda")
    continuous_point = result.get("continuous_point")
    master_discrete_state = result.get(
        "master_discrete_state"
    )

    if master_discrete_state is None:
        raise ValueError(
            f"Missing master discrete state for "
            f"subproblem '{subproblem_key}'."
        )

    if continuous_lambda is None:
        raise ValueError(
            f"Subproblem '{subproblem_key}' returned no "
            "continuous_lambda."
        )

    if continuous_point is None:
        raise ValueError(
            f"Subproblem '{subproblem_key}' returned no "
            "continuous_point."
        )

    if master_discrete_state is None:
        raise ValueError(
            f"Missing master discrete state for "
            f"subproblem '{subproblem_key}'."
        )

    state = {
        "op_cost": float(op_cost),
        "theta_coeff": theta_coeff,
        "continuous_lambda": dict(continuous_lambda),
        "continuous_point": dict(continuous_point),
        "master_discrete_state": master_discrete_state,
        "recourse_lower_bound": result.get(
            "recourse_lower_bound"
        ),
    }

    benders_states.setdefault(
        subproblem_key,
        []
    ).append(state)

def add_relaxed_benders_cut(
    planning_problem,
    subproblem_key,
    linking_vars,
    benders_lambda,
    benders_point,
    op_cost,
    theta_coeff=1.0,
):
    """
    Add a standard Benders optimality cut for the current relaxed
    linking-variable space.

    This is used during the P&J relaxed phase, where binary and integer
    linking variables have temporarily been relaxed to continuous domains.

    The cut is:

        theta_coeff * theta_w >=
            op_cost + sum(
                lambda_i * (x_i - xbar_i)
            )

    Equivalently:

        theta_coeff * theta_w
        - op_cost
        - sum(lambda_i * (x_i - xbar_i))
        >= 0
    """

    theta = planning_problem.vTHETA[subproblem_key]

    cut_scale = (
        BENDERS_THETA_SCALE
        if abs(float(theta_coeff)) > 1e-12
        else 1.0
    )
    cut_expression = (
        theta_coeff * theta
        - float(op_cost) / cut_scale
    )

    for name in linking_vars:

        if name not in benders_lambda:
            raise KeyError(
                f"Missing Benders lambda coefficient for linking variable "
                f"'{name}' in subproblem '{subproblem_key}'."
            )

        if name not in benders_point:
            raise KeyError(
                f"Missing Benders incumbent point for linking variable "
                f"'{name}' in subproblem '{subproblem_key}'."
            )

        var = planning_problem.find_component(name)

        if var is None:
            raise KeyError(
                f"Could not find planning variable '{name}' in the "
                f"planning problem."
            )

        lam = float(benders_lambda[name])
        xbar = float(benders_point[name])

        cut_expression -= (lam / cut_scale) * (var - xbar)

    planning_problem.BendersOptimalityCuts.add(
        cut_expression >= 0
    )

def update_planning_problem_multi_cuts(
    planning_problem,
    benders_states,
    benders_feasibility_states,
    linking_variables_sub_by_type,
    master_discrete_method,
    master_integrality_restored,
    recourse_lower_bounds,
    optimiser,
    native_djc=True,
):
    """Dispatch accumulated discrete-state cuts by master method.

    P&J adds relaxed affine cuts directly in its first phase and
    state-conditioned cuts after integrality is restored. The linearized
    method builds linear activation formulations; DJC passes state data to
    the native MOSEK disjunction builder. ``none`` adds no discrete-state
    cuts and is valid only when linking variables are continuous.
    """

    if master_discrete_method == "pj_relax":

        # ---------------------------------------------------------
        # P&J relaxed phase
        #
        # Relaxed cuts are added directly from subop_sol, so there
        # is nothing to do here.
        # ---------------------------------------------------------

        if not master_integrality_restored:
            return

        if native_djc and str(getattr(optimiser, "name", "")).lower().startswith(
            "mosek"
        ):
            for subproblem_key, variable_types in (
                linking_variables_sub_by_type.items()
            ):
                if variable_types["integer"]:
                    raise NotImplementedError(
                        "Native MOSEK P&J state cuts support binary "
                        "discrete linking variables only. Integer "
                        f"variables found for '{subproblem_key}': "
                        f"{variable_types['integer']}"
                    )

            native_subproblems = {
                key
                for key, variable_types
                in linking_variables_sub_by_type.items()
                if variable_types["binary"]
            }
            planning_problem._benders_native_djc_data = {
                "benders_states": {
                    key: states
                    for key, states in benders_states.items()
                    if key in native_subproblems
                },
                "benders_feasibility_states": {
                    key: states
                    for key, states in benders_feasibility_states.items()
                    if key in native_subproblems
                },
                "linking_variables_sub_by_type": (
                    linking_variables_sub_by_type
                ),
            }
            return

        # ---------------------------------------------------------
        # Integer phase
        # ---------------------------------------------------------

        for subproblem_key, states in benders_states.items():

            continuous_vars = (
                linking_variables_sub_by_type[
                    subproblem_key
                ]["continuous"]
            )

            for state in states:

                add_state_recourse_cut(
                    planning_problem=planning_problem,
                    subproblem_key=subproblem_key,
                    discrete_state=state["master_discrete_state"],
                    continuous_vars=continuous_vars,
                    continuous_lambda=state["continuous_lambda"],
                    continuous_point=state["continuous_point"],
                    op_cost=state["op_cost"],
                    theta_coeff=state["theta_coeff"],
                    recourse_lower_bound=(
                        state.get("recourse_lower_bound")
                        if state.get("recourse_lower_bound") is not None
                        else recourse_lower_bounds[subproblem_key]
                    ),
                    optimiser=optimiser,
                )

    elif master_discrete_method == "linearized":

        for subproblem_key, states in benders_states.items():

            continuous_vars = (
                linking_variables_sub_by_type[
                    subproblem_key
                ]["continuous"]
            )

            for state in states:

                add_state_recourse_cut(
                    planning_problem=planning_problem,
                    subproblem_key=subproblem_key,
                    discrete_state=state["master_discrete_state"],
                    continuous_vars=continuous_vars,
                    continuous_lambda=state["continuous_lambda"],
                    continuous_point=state["continuous_point"],
                    op_cost=state["op_cost"],
                    theta_coeff=state["theta_coeff"],
                    recourse_lower_bound=(
                        state.get("recourse_lower_bound")
                        if state.get("recourse_lower_bound") is not None
                        else recourse_lower_bounds[subproblem_key]
                    ),
                    optimiser=optimiser,
                )

    elif master_discrete_method == "djc":

        planning_problem._benders_native_djc_data = {
            "benders_states": benders_states,
            "benders_feasibility_states": (
                benders_feasibility_states
            ),
            "linking_variables_sub_by_type": (
                linking_variables_sub_by_type
            ),
        }

    elif master_discrete_method == "none":
        # No master discrete-state cuts should be generated.
        #
        # If this method is selected, the caller should already have
        # verified that there are no discrete linking variables.
        return

    else:

        raise ValueError(
            f"Unknown BendersDiscreteMethod: "
            f"{master_discrete_method}"
        )

def add_state_recourse_cut(
    planning_problem,
    subproblem_key,
    discrete_state,
    continuous_vars,
    continuous_lambda,
    continuous_point,
    op_cost,
    theta_coeff,
    recourse_lower_bound,
    optimiser,
):
    """
    Add a discrete-state-conditioned Benders optimality cut.

    The cut is active when the master discrete variables exactly match
    the incumbent discrete state.

    For the matching discrete state:

        theta_coeff * theta >=
            op_cost
            + lambda^T (x - xbar)

    The cut is relaxed when the discrete state differs from the incumbent
    by means of an activation variable z.

    The resulting constraint is:

        cut_expression + M * z >= 0

    where z = 0 iff the incumbent discrete pattern is reproduced.
    """

    theta = planning_problem.vTHETA[subproblem_key]

    # ------------------------------------------------------------------
    # Create a block for this cut.
    # ------------------------------------------------------------------

    cut_id = len(planning_problem.component_map().get(
        "BendersDiscretePattern",
        []
    )) if hasattr(
        planning_problem,
        "BendersDiscretePattern"
    ) else 0

    block_name = f"BendersDiscretePattern_{subproblem_key}_{cut_id}"

    # Avoid relying on the component count above if the name already
    # exists. Your existing implementation can retain its own cut-ID
    # mechanism if you already have one.
    if hasattr(planning_problem, block_name):
        raise RuntimeError(
            f"Benders cut block '{block_name}' already exists."
        )

    block = pyo.Block()

    planning_problem.add_component(
        block_name,
        block,
    )

    # ------------------------------------------------------------------
    # Create mismatch indicators.
    #
    # z = 0 iff every discrete variable matches the incumbent.
    # ------------------------------------------------------------------

    mismatch_vars = []

    # ==============================================================
    # Binary variables
    # ==============================================================

    for name, target in discrete_state.get("binary", {}).items():

        var = planning_problem.find_component(name)

        if var is None:
            raise KeyError(
                f"Binary linking variable '{name}' was not found "
                f"in the planning problem."
            )

        target = int(round(float(target)))

        mismatch = pyo.Var(
            domain=pyo.Binary
        )

        block.add_component(
            f"Mismatch_{name}",
            mismatch,
        )

        # If target = 1:
        #
        #     mismatch = 1 - var
        #
        # If target = 0:
        #
        #     mismatch = var
        #
        if target == 1:
            block.add_component(
                f"MismatchConstraint_{name}",
                pyo.Constraint(
                    expr=mismatch == 1 - var
                ),
            )
        else:
            block.add_component(
                f"MismatchConstraint_{name}",
                pyo.Constraint(
                    expr=mismatch == var
                ),
            )

        mismatch_vars.append(mismatch)

    # ==============================================================
    # Integer variables
    # ==============================================================

    for name, target in discrete_state.get("integer", {}).items():

        var = planning_problem.find_component(name)

        if var is None:
            raise KeyError(
                f"Integer linking variable '{name}' was not found "
                f"in the planning problem."
            )

        target = int(round(float(target)))

        lb, ub = get_continuous_variable_bounds(
            planning_problem,
            name,
            optimiser,
        )

        if lb is None or ub is None:
            raise ValueError(
                f"Could not obtain finite bounds for integer linking "
                f"variable '{name}'."
            )

        lb = int(math.floor(float(lb)))
        ub = int(math.ceil(float(ub)))

        if target < lb or target > ub:
            raise ValueError(
                f"Incumbent integer value {target} for '{name}' is "
                f"outside bounds [{lb}, {ub}]."
            )

        # ----------------------------------------------------------
        # below = 1 if var <= target - 1
        # ----------------------------------------------------------

        below = pyo.Var(
            domain=pyo.Binary
        )

        # above = 1 if var >= target + 1
        # ----------------------------------------------------------

        above = pyo.Var(
            domain=pyo.Binary
        )

        block.add_component(
            f"Below_{name}",
            below,
        )

        block.add_component(
            f"Above_{name}",
            above,
        )

        # Below:
        #
        # var <= target - 1 + M_below * (1 - below)
        #
        # When below = 1:
        #
        # var <= target - 1
        #
        below_M = max(
            0,
            ub - (target - 1)
        )

        block.add_component(
            f"BelowConstraint_{name}",
            pyo.Constraint(
                expr=var <= target - 1 + below_M * (1 - below)
            ),
        )

        # Above:
        #
        # var >= target + 1 - M_above * (1 - above)
        #
        # When above = 1:
        #
        # var >= target + 1
        #
        above_M = max(
            0,
            (target + 1) - lb
        )

        block.add_component(
            f"AboveConstraint_{name}",
            pyo.Constraint(
                expr=var >= target + 1 - above_M * (1 - above)
            ),
        )

        # We need a mismatch indicator that is 1 whenever the integer
        # variable differs from the target.
        integer_mismatch = pyo.Var(
            domain=pyo.Binary
        )

        block.add_component(
            f"IntegerMismatch_{name}",
            integer_mismatch,
        )

        block.add_component(
            f"IntegerMismatchLower_{name}",
            pyo.Constraint(
                expr=integer_mismatch >= below
            ),
        )

        block.add_component(
            f"IntegerMismatchUpper_{name}",
            pyo.Constraint(
                expr=integer_mismatch >= above
            ),
        )

        block.add_component(
            f"IntegerMismatchSum_{name}",
            pyo.Constraint(
                expr=integer_mismatch <= below + above
            ),
        )

        mismatch_vars.append(integer_mismatch)

    # ------------------------------------------------------------------
    # Global activation variable z.
    #
    # z = 0 when all discrete variables match.
    # z = 1 when any discrete variable differs.
    # ------------------------------------------------------------------

    z = pyo.Var(
        domain=pyo.Binary
    )

    block.delta = z

    for i, mismatch in enumerate(mismatch_vars):

        block.add_component(
            f"DeltaLower_{i}",
            pyo.Constraint(
                expr=z >= mismatch
            ),
        )

    if mismatch_vars:
        block.add_component(
            "DeltaUpper",
            pyo.Constraint(
                expr=z <= sum(mismatch_vars)
            ),
        )
    else:
        # There are no discrete variables. The state cut should simply
        # reduce to an ordinary continuous Benders cut.
        block.add_component(
            "DeltaZero",
            pyo.Constraint(
                expr=z == 0
            ),
        )

    # ------------------------------------------------------------------
    # Construct the Benders cut.
    # ------------------------------------------------------------------

    cut_scale = (
        BENDERS_THETA_SCALE
        if abs(float(theta_coeff)) > 1e-12
        else 1.0
    )
    cut_expression = (
        float(theta_coeff) * theta
        - float(op_cost) / cut_scale
    )

    for name in continuous_vars:

        if name not in continuous_lambda:
            raise KeyError(
                f"Missing continuous lambda coefficient for '{name}'."
            )

        if name not in continuous_point:
            raise KeyError(
                f"Missing continuous incumbent point for '{name}'."
            )

        var = planning_problem.find_component(name)

        if var is None:
            raise KeyError(
                f"Continuous linking variable '{name}' was not found "
                f"in the planning problem."
            )

        lam = float(continuous_lambda[name])
        xbar = float(continuous_point[name])

        cut_expression -= (lam / cut_scale) * (var - xbar)

    # ------------------------------------------------------------------
    # Calculate a valid Big-M.
    #
    # We need:
    #
    #     cut_expression + M*z >= 0
    #
    # to be automatically satisfied when z = 1.
    #
    # Therefore:
    #
    # M >= op_cost
    #      + max lambda'(x-xbar)
    #      - theta_coeff * theta_lb
    # ------------------------------------------------------------------

    if recourse_lower_bound is None:
        raise ValueError(
            f"No recourse lower bound supplied for subproblem "
            f"'{subproblem_key}'."
        )

    recourse_lower_bound = float(recourse_lower_bound)

    if not math.isfinite(recourse_lower_bound):
        raise ValueError(
            f"Non-finite recourse lower bound for subproblem "
            f"'{subproblem_key}': {recourse_lower_bound}"
        )

    M = calculate_benders_big_m(
        planning_problem=planning_problem,
        continuous_linking_vars=continuous_vars,
        continuous_lambda=continuous_lambda,
        continuous_point=continuous_point,
        op_cost=float(op_cost),
        recourse_lower_bound=float(recourse_lower_bound),
        optimiser=optimiser,
    )
    M /= cut_scale

    block.BigM = pyo.Param(
        initialize=M,
        mutable=False,
    )

    block.ConditionalBendersCut = pyo.Constraint(
        expr=cut_expression + M * z >= 0
    )

    return M

def calculate_benders_big_m(
    planning_problem,
    continuous_linking_vars,
    continuous_lambda,
    continuous_point,
    op_cost,
    recourse_lower_bound,
    optimiser,
    theta_coeff = 1.0
):
    """
    Calculate a valid Big-M for a state-conditioned Benders
    optimality cut.

    The conditional cut is

        theta - op_cost
        - lambda' (x - xbar)
        + M z >= 0

    where z=0 activates the cut and z=1 deactivates it.

    A valid M satisfies

        M >= op_cost
             + max_x lambda'(x-xbar)
             - theta_lower_bound.
    """

    dual_term_ub = continuous_dual_term_upper_bound(
        planning_problem=planning_problem,
        continuous_linking_vars=continuous_linking_vars,
        continuous_lambda=continuous_lambda,
        continuous_point=continuous_point,
        optimiser=optimiser,
    )

    M = (
        op_cost
        + dual_term_ub
        - float(theta_coeff) * float(recourse_lower_bound)
    )

    # Numerical protection.
    M = max(0.0, M)

    return M


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

    gridpath_ub = find_gridpath_capacity_upper_bound(
        planning_problem,
        var,
    )

    if gridpath_ub is not None:

        if lb is None:
            if var.is_nonnegative():
                lb = 0.0
            else:
                raise ValueError(
                    f"No finite lower bound found for "
                    f"continuous linking variable '{var.name}'."
                )

        ub = float(gridpath_ub)

        cache[var.name] = (lb, ub)
        return lb, ub

    # ---------------------------------------------------------
    # 3. OBBT fallback
    # ---------------------------------------------------------

    ub = obbt_upper_bound(
        planning_problem,
        var,
        optimiser,
    )

    if ub is None:
        # No finite bound is derivable; use the same cap as the master.
        from gridpath.benders.planning import BENDERS_MASTER_VARIABLE_CAP

        logging.getLogger(__name__).warning(
            "No finite upper bound for linking variable %s; using %g.",
            var.name,
            BENDERS_MASTER_VARIABLE_CAP,
        )
        ub = BENDERS_MASTER_VARIABLE_CAP

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
            tee=False,
        )

        termination = (
            results.solver.termination_condition
        )

        if termination != TerminationCondition.optimal:
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

    return best_ub

def continuous_dual_term_upper_bound(
    planning_problem,
    continuous_linking_vars,
    continuous_lambda,
    continuous_point,
    optimiser,
):
    bound = 0.0

    for name in continuous_linking_vars:

        lam = continuous_lambda.get(name, 0.0)

        if abs(lam) < 1e-12:
            continue

        xbar = continuous_point[name]

        # Resolve the component name to the actual Pyomo Var
        var = planning_problem.find_component(name)

        if var is None:
            raise KeyError(
                f"Could not find continuous linking variable "
                f"'{name}' in planning problem."
            )

        lb, ub = get_continuous_variable_bounds(
            planning_problem,
            var,
            optimiser,
        )

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

def compute_incumbent_npv(planning_sol, subop_sol):

    if any(
        not sp["feasible"]
        for sp in subop_sol.values()
    ):
        return -math.inf

    if any(
        abs(sp.get("theta_coeff", 1.0)) < 1e-12
        for sp in subop_sol.values()
    ):
        raise RuntimeError(
            "Feasible subproblem has zero theta coefficient."
        )

    return (
        planning_sol.planning_cost
        - sum(
            sp["op_cost"]
            for sp in subop_sol.values()
        )
    )


def _ascii_plot(series, height, width, log_scale=False, title_fmt="{:>12.4g}"):
    """Render several (symbol, values) series as an ASCII scatter plot.

    The y-range is set from the final half of the finite data so early
    outliers do not squash the converged region; points outside the range are
    clipped to the border and drawn as '^' or 'v'.
    """
    def transform(v):
        if v is None or not math.isfinite(v):
            return None
        if log_scale:
            return math.log10(v) if v > 0 else None
        return v

    n = max(len(v) for _, v in series)
    tvals = [[transform(x) for x in vals] for _, vals in series]
    tail = [
        x for vals in tvals for x in vals[len(vals) // 2:] if x is not None
    ]
    if not tail:
        tail = [x for vals in tvals for x in vals if x is not None]
    if not tail:
        return None

    lo, hi = min(tail), max(tail)
    pad = (hi - lo) * 0.1 if hi > lo else max(abs(hi) * 0.05, 1.0)
    lo, hi = lo - pad, hi + pad

    grid = [[" "] * width for _ in range(height)]

    def col(i):
        return 0 if n <= 1 else int(round(i / (n - 1) * (width - 1)))

    for symbol, vals in zip([s for s, _ in series], tvals):
        for i, x in enumerate(vals):
            if x is None:
                continue
            if x > hi:
                row, mark = 0, "^"
            elif x < lo:
                row, mark = height - 1, "v"
            else:
                row = int(round((hi - x) / (hi - lo) * (height - 1)))
                mark = symbol
            c = col(i)
            # Where the series coincide, show '*'
            grid[row][c] = mark if grid[row][c] in (" ", mark) else "*"

    out = []
    for r in range(height):
        y = hi - r / max(height - 1, 1) * (hi - lo)
        label = f"1e{y:>6.1f}" if log_scale else title_fmt.format(y)
        out.append(f"{label:>12} |" + "".join(grid[r]))
    out.append(" " * 12 + " +" + "-" * width)
    return out


def print_convergence_graph(
    LB_hist,
    UB_hist,
    gap_hist,
    operational_cost=None,
    master_objective=None,
    npv=None,
    absolute_gap=None,
    absolute_gap_tolerance=None,
    width=None,
    height=None,
    history_window=None,
):
    """Print Benders progress and ASCII-only convergence plots."""
    import shutil

    if not LB_hist:
        return

    terminal_size = shutil.get_terminal_size(fallback=(100, 30))
    if width is None:
        width = max(20, min(80, terminal_size.columns - 20))
    if height is None:
        height = 12
    if history_window is None:
        history_window = width

    iteration = len(LB_hist) - 1
    current_lb = LB_hist[-1]
    current_ub = UB_hist[-1]
    current_gap = gap_hist[-1]
    rule = "=" * (width + 14)

    def fmt(x):
        return "inf" if x in (math.inf, -math.inf) else f"{x:,.6g}"

    print(rule)
    print(f" BENDERS CONVERGENCE - iteration {iteration}")
    print(rule)
    if master_objective is not None:
        print(f"   Master objective     : {fmt(master_objective)}")
    if operational_cost is not None:
        print(f"   Net operational cost : {fmt(operational_cost)}")
    if npv is not None:
        print(f"   Original model NPV   : {fmt(npv)}")
    print(f"   LB                   : {fmt(current_lb)}")
    print(f"   UB                   : {fmt(current_ub)}")
    print(
        "   Relative gap         : "
        + (f"{current_gap:.3e}" if math.isfinite(current_gap) else "inf")
    )
    if absolute_gap is not None:
        target = (
            f"  (target {absolute_gap_tolerance:,.0f})"
            if absolute_gap_tolerance is not None
            and math.isfinite(absolute_gap_tolerance)
            else ""
        )
        print(f"   Absolute gap         : {absolute_gap:,.2f}{target}")

    start = max(0, len(LB_hist) - history_window)
    x_label = (
        f"iterations {start} .. {iteration}"
        f"   (L = lower bound, U = upper bound, * = overlap,"
        f" ^/v = off-scale)"
    )

    print()
    print(" OBJECTIVE BOUNDS (y-range zoomed on the later iterations)")
    bounds = _ascii_plot(
        [("L", LB_hist[start:]), ("U", UB_hist[start:])], height, width
    )
    if bounds:
        print("\n".join(bounds))
        print(" " * 14 + x_label)

    print()
    print(" RELATIVE GAP (log10 scale)")
    if math.isfinite(current_gap) and current_gap == 0:
        print("   CONVERGED: LB = UB")
    else:
        gaps = _ascii_plot(
            [("o", gap_hist[start:])], height, width, log_scale=True
        )
        if gaps:
            print("\n".join(gaps))
            print(" " * 14 + f"iterations {start} .. {iteration}")
        else:
            print("   No positive finite gaps to plot.")

    print(rule)
