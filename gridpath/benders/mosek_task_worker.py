"""Experimental persistent native-MOSEK-task Benders worker.

Enable with GRIDPATH_BENDERS_WORKER_BACKEND=mosek_task. The ordinary Pyomo
worker remains the default. The task owns normal recourse iterations; Pyomo is
rebuilt only for restoration and final GridPath export.
"""
import gc
import logging
import os
import tempfile
import threading
from pathlib import Path

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.opt import SolverResults

from gridpath.auxiliary.dynamic_components import DynamicComponents
from gridpath.benders.gams_benders_solver import create_gams_solver
from gridpath.benders.operational_model_creation import create_operational_problem_benders
from gridpath.benders.mosek_task import write_pyomo_mosek_task
from gridpath.benders.subproblems import (
    BENDERS_RECOURSE_OBJECTIVE_SCALE,
    _StartupGuard,
    _export_final_subproblem,
    calculate_recourse_upper_bound,
    classify_linking_vars,
    get_present_linking_vars,
    get_shared_constraints,
    process_subproblem_solution,
    process_memory_rss,
)

LOG = logging.getLogger(__name__)


def _make_pyomo_solver(parsed_arguments, solver_options=None):
    if str(parsed_arguments.solver).lower() == "gams":
        solver = create_gams_solver(
            solver_options, executable=parsed_arguments.solver_executable
        )
        if getattr(solver, "backend_name", "").startswith("mosek"):
            solver.options["MSK_IPAR_LOG"] = "0"
        return solver
    if parsed_arguments.solver_executable is not None:
        return SolverFactory(
            parsed_arguments.solver, executable=parsed_arguments.solver_executable
        )
    return SolverFactory(parsed_arguments.solver)


def _make_task_solver(solver_options=None):
    # This is deliberately the direct interface, not the GAMS shell adapter.
    solver = SolverFactory("mosek", solver_io="direct")
    solver.options["MSK_IPAR_LOG"] = "0"
    solver.options["MSK_IPAR_OPTIMIZER"] = "MSK_OPTIMIZER_PRIMAL_SIMPLEX"
    threads = os.getenv("GRIDPATH_BENDERS_SOLVER_THREADS")
    if threads:
        solver.options["MSK_IPAR_NUM_THREADS"] = str(int(threads))
    # Only MOSEK-native options are meaningful to the direct interface.
    for key, value in (solver_options or {}).items():
        if str(key).startswith("MSK_"):
            solver.options[str(key)] = str(value)
    return solver


def _build_pyomo_instance(
    key, planning_constraint_names, scenario_directory, multi_stage,
    parsed_arguments,
):
    dynamic_components = DynamicComponents()
    instance = create_operational_problem_benders(
        scenario_directory=scenario_directory,
        key=key,
        dynamic_components=dynamic_components,
        multi_stage=multi_stage,
        parsed_arguments=parsed_arguments,
    )
    for name in get_shared_constraints(instance, planning_constraint_names):
        instance.find_component(name).deactivate()
    instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return instance, dynamic_components


def _clamped_linking_values(instance, planning_sol, names):
    values = {}
    for name in names:
        var = instance.find_component(name)
        if var is None:
            continue
        if name not in planning_sol.values:
            raise KeyError(f"Missing linking value for {name}")
        target = float(planning_sol.values[name])
        if var.lb is not None:
            target = max(target, float(pyo.value(var.lb)))
        if var.ub is not None:
            target = min(target, float(pyo.value(var.ub)))
        values[name] = target
    return values


def _add_initial_linking_rows(instance, names):
    instance.BendersLinkingConstraints = pyo.ConstraintList()
    instance.BendersLinkingConstraintMap = {}
    for name in names:
        var = instance.find_component(name)
        if var is None:
            continue
        initial = pyo.value(var, exception=False)
        if initial is None:
            initial = 0.0
        row = instance.BendersLinkingConstraints.add(var == float(initial))
        instance.BendersLinkingConstraintMap[name] = row


def _mosek_bound(mosek, lb, ub):
    if lb is None and ub is None:
        return mosek.boundkey.fr, 0.0, 0.0
    if lb is None:
        return mosek.boundkey.up, 0.0, float(ub)
    if ub is None:
        return mosek.boundkey.lo, float(lb), 0.0
    if float(lb) == float(ub):
        return mosek.boundkey.fx, float(lb), float(ub)
    return mosek.boundkey.ra, float(lb), float(ub)


class _TaskWorkerState:
    def __init__(self, task, metadata, mosek, task_solver=None):
        # Retain the Pyomo direct-solver wrapper as the owner of its MOSEK
        # environment. The Pyomo model and interface maps are released by the
        # compiler helper, but the environment must outlive the Task.
        self.task_solver = task_solver
        self.task = task
        self.metadata = metadata
        self.mosek = mosek
        self.variables = metadata["variables"]
        self.columns = metadata["variable_columns"]
        self.linking_rows = metadata["linking_rows"]
        self.scale = float(metadata["objective_scale"])
        self.linking_names = set(self.linking_rows)
        self.operational_integer_names = [
            name for name, info in self.variables.items()
            if (info["binary"] or info["integer"])
            and name not in self.linking_names and not info["fixed"]
        ]

    def _set_type(self, name, integer):
        col = self.columns[name]
        vartype = (
            self.mosek.variabletype.type_int
            if integer else self.mosek.variabletype.type_cont
        )
        self.task.putvartype([col], [vartype])

    def _restore_original_bounds(self):
        for name, info in self.variables.items():
            col = info["column"]
            if info["fixed"] and info.get("fixed_value") is not None:
                lower = upper = float(info["fixed_value"])
            else:
                lower, upper = info["lb"], info["ub"]
            key, lower, upper = _mosek_bound(self.mosek, lower, upper)
            self.task.putvarbound(col, key, lower, upper)

    def _set_integrality(self, preserve_integer):
        for name, info in self.variables.items():
            if info["binary"] or info["integer"]:
                self._set_type(name, bool(preserve_integer))

    def _set_linking_values(self, values):
        missing = self.linking_names.difference(values)
        if missing:
            raise KeyError(f"Missing task linking values: {sorted(missing)}")
        for name, row in self.linking_rows.items():
            rhs = float(values[name])
            info = self.variables[name]
            if info["lb"] is not None:
                rhs = max(rhs, float(info["lb"]))
            if info["ub"] is not None:
                rhs = min(rhs, float(info["ub"]))
            self.task.putconbound(
                int(row), self.mosek.boundkey.fx, rhs, rhs
            )

    def _optimize_mip(self):
        self.task.optimize()
        status = self.task.getsolsta(self.mosek.soltype.itg)
        if status != self.mosek.solsta.integer_optimal:
            return status, None
        values = [0.0] * self.task.getnumvar()
        self.task.getxx(self.mosek.soltype.itg, values)
        state = {"binary": {}, "integer": {}}
        for name in self.operational_integer_names:
            info = self.variables[name]
            value = int(round(float(values[self.columns[name]])))
            state["binary" if info["binary"] else "integer"][name] = value
            col = self.columns[name]
            self.task.putvarbound(
                col, self.mosek.boundkey.fx, float(value), float(value)
            )
            self._set_type(name, False)
        return status, state

    def _optimize_lp(self):
        self.task.optimize()
        status = self.task.getsolsta(self.mosek.soltype.bas)
        if status != self.mosek.solsta.optimal:
            return status, None
        primal = [0.0] * self.task.getnumvar()
        dual = [0.0] * self.task.getnumcon()
        self.task.getxx(self.mosek.soltype.bas, primal)
        self.task.gety(self.mosek.soltype.bas, dual)
        return status, (primal, dual)

    def solve(self, linking_values, active_linking_names, operational_method):
        self._restore_original_bounds()
        self._set_linking_values(linking_values)
        self._set_integrality(True)

        discrete_state = None
        if operational_method == "fixed_z" and self.operational_integer_names:
            mip_status, discrete_state = self._optimize_mip()
            if mip_status != self.mosek.solsta.integer_optimal:
                return False, mip_status, None, None, None

        # LP duals are defined only after all integer variables, including
        # discrete linking variables, have been relaxed. fixed-z operational
        # variables remain fixed at their MIP incumbent bounds.
        self._set_integrality(False)
        lp_status, solution = self._optimize_lp()
        if lp_status != self.mosek.solsta.optimal:
            return False, lp_status, None, None, discrete_state

        primal, dual = solution
        cost = -float(self.task.getprimalobj(self.mosek.soltype.bas)) * self.scale
        lambdas = {
            name: -float(dual[int(self.linking_rows[name])]) * self.scale
            for name in active_linking_names
        }
        return True, lp_status, cost, lambdas, discrete_state

    def cleanup_iteration(self):
        # Fixed-z bounds must never leak into the next master trial point.
        self._restore_original_bounds()
        self._set_integrality(True)


def _restore_with_pyomo(
    key, planning_sol, active_linking_vars, present_linking_vars,
    planning_constraint_names, scenario_directory, multi_stage,
    parsed_arguments, subproblem_key, solver_options=None,
):
    instance, _ = _build_pyomo_instance(
        key, planning_constraint_names, scenario_directory, multi_stage,
        parsed_arguments,
    )
    from gridpath.benders.subproblems import (
        fix_linking_variables, _set_feasibility_restoration_mode,
    )
    if hasattr(instance, "slack_max"):
        _set_feasibility_restoration_mode(instance, active=False)
    fix_linking_variables(instance, planning_sol, present_linking_vars)
    results = SolverResults()
    results.solver.status = SolverStatus.warning
    results.solver.termination_condition = TerminationCondition.infeasible
    solver = _make_pyomo_solver(parsed_arguments, solver_options)
    linking_types = classify_linking_vars(instance, present_linking_vars)
    return process_subproblem_solution(
        instance, results, planning_sol, linking_types, solver,
        active_linking_vars=active_linking_vars,
        present_linking_vars=present_linking_vars,
        subproblem_key=subproblem_key,
    )


def benders_subproblem_worker(
    key, planning_constraint_names, linking_vars, input_queue, output_queue,
    scenario_directory, multi_stage, parsed_arguments, solver_options=None,
    startup_semaphore=None, linking_var_bounds=None,
    calculate_global_recourse_bound=True, memory_coordinator=None,
    worker_index=None,
):
    """Persistent worker backed by a native MOSEK Task."""
    guard = _StartupGuard(startup_semaphore)
    process_rss = process_memory_rss
    initial_persistent = int(float(os.getenv(
        "GRIDPATH_BENDERS_INITIAL_PERSISTENT_GB", "4"
    )) * 1024**3)
    initial_temporary = int(float(os.getenv(
        "GRIDPATH_BENDERS_INITIAL_TEMPORARY_GB", "4"
    )) * 1024**3)
    safety_factor = float(os.getenv("GRIDPATH_BENDERS_MEMORY_SAFETY_FACTOR", "1.25"))
    if safety_factor < 1:
        raise ValueError("GRIDPATH_BENDERS_MEMORY_SAFETY_FACTOR must be at least 1.")
    task_directory = None
    try:
        guard.acquire()
        if memory_coordinator is not None:
            memory_coordinator.set_reservation(
                worker_index, persistent=initial_persistent,
                temporary=initial_temporary,
            )

        instance, dynamic_components = _build_pyomo_instance(
            key, planning_constraint_names, scenario_directory, multi_stage,
            parsed_arguments,
        )
        linking_types = classify_linking_vars(instance, linking_vars)
        present_linking_vars = get_present_linking_vars(linking_types)
        optimiser = _make_pyomo_solver(parsed_arguments, solver_options)
        recourse_ub = None
        if calculate_global_recourse_bound:
            recourse_ub = calculate_recourse_upper_bound(
                instance, optimiser, linking_var_names=present_linking_vars,
                linking_var_bounds=linking_var_bounds,
            )

        _add_initial_linking_rows(instance, present_linking_vars)
        task_solver = _make_task_solver(solver_options)
        original_objective = instance.OP_NPV.expr
        instance.OP_NPV.set_value(
            original_objective / BENDERS_RECOURSE_OBJECTIVE_SCALE
        )
        try:
            import mosek
            task_directory = tempfile.mkdtemp(prefix="gridpath_benders_task_")
            task_path = Path(task_directory) / "operational.task"
            task, metadata = write_pyomo_mosek_task(
                instance, task_solver, task_path,
                instance.BendersLinkingConstraintMap,
                objective_scale=BENDERS_RECOURSE_OBJECTIVE_SCALE,
                release_pyomo_maps=True,
            )
        finally:
            instance.OP_NPV.set_value(original_objective)

        state = _TaskWorkerState(task, metadata, mosek, task_solver=task_solver)
        del instance, optimiser, dynamic_components
        gc.collect()

        if memory_coordinator is not None:
            resident = process_rss()
            memory_coordinator.set_reservation(worker_index, temporary=0)
            memory_coordinator.set_reservation(
                worker_index, persistent=int(resident * safety_factor)
            )
        guard.release()
        output_queue.put({
            "status": "READY",
            "linking_vars": linking_vars,
            "recourse_upper_bound": recourse_ub,
            "linking_vars_by_type": linking_types,
            "present_linking_vars": present_linking_vars,
            "worker_backend": "mosek_task",
        })

        while True:
            message = input_queue.get()
            if message == "STOP":
                break
            if isinstance(message, dict) and message.get("command") == "EXPORT_FINAL":
                if memory_coordinator is not None:
                    memory_coordinator.set_reservation(
                        worker_index, temporary=initial_temporary
                    )
                export_instance, export_dynamic = _build_pyomo_instance(
                    key, planning_constraint_names, scenario_directory,
                    multi_stage, parsed_arguments,
                )
                # _export_final_subproblem applies the incumbent linking
                # equalities itself, using the same clamping and row lifecycle
                # as the original Pyomo worker.
                export_solver = _make_pyomo_solver(parsed_arguments, solver_options)
                export_result = _export_final_subproblem(
                    instance=export_instance,
                    dynamic_components=export_dynamic,
                    optimiser=export_solver,
                    key=key,
                    planning_sol=message["planning_sol"],
                    present_linking_vars=present_linking_vars,
                    scenario_directory=scenario_directory,
                    multi_stage=multi_stage,
                    parsed_arguments=parsed_arguments,
                )
                output_queue.put((key, export_result))
                del export_instance, export_dynamic, export_solver
                gc.collect()
                if memory_coordinator is not None:
                    memory_coordinator.set_reservation(worker_index, temporary=0)
                continue

            if memory_coordinator is not None:
                memory_coordinator.set_reservation(
                    worker_index, temporary=initial_temporary
                )
            planning_sol = message["planning_sol"]
            master_method = message["master_discrete_method"]
            configured_method = message["operational_discrete_method"]
            master_integrality_restored = message.get(
                "master_integrality_restored", False
            )
            pj_relaxed_phase = (
                master_method == "pj_relax" and not master_integrality_restored
            )
            operational_method = "pj_relax" if pj_relaxed_phase else configured_method
            active_linking = (
                present_linking_vars if pj_relaxed_phase
                else linking_types["continuous"]
            )
            linking_values = {}
            for name in present_linking_vars:
                if name not in planning_sol.values:
                    raise KeyError(f"Missing linking value for {name}")
                linking_values[name] = float(planning_sol.values[name])

            feasible, status, cost, lambdas, discrete_state = state.solve(
                linking_values, active_linking, operational_method
            )
            if feasible:
                continuous_point = {
                    name: float(planning_sol.values[name]) for name in active_linking
                }
                result = {
                    "feasible": True,
                    "op_cost": float(cost),
                    "benders_lambda": lambdas,
                    "continuous_lambda": dict(lambdas),
                    "continuous_point": continuous_point,
                    "theta_coeff": 1.0,
                    "operational_discrete_state": discrete_state,
                    "master_discrete_state": {
                        "binary": {
                            n: int(round(float(planning_sol.values[n])))
                            for n in linking_types["binary"]
                            if n in planning_sol.values
                        },
                        "integer": {
                            n: int(round(float(planning_sol.values[n])))
                            for n in linking_types["integer"]
                            if n in planning_sol.values
                        },
                    } if (master_method in ("linearized", "djc") or master_integrality_restored)
                    else None,
                    "is_mip": bool(state.operational_integer_names),
                    "operational_mip_variables": list(state.operational_integer_names),
                }
            else:
                result = _restore_with_pyomo(
                    key, planning_sol, active_linking, present_linking_vars,
                    planning_constraint_names, scenario_directory, multi_stage,
                    parsed_arguments, key, solver_options=solver_options,
                )
                result["operational_discrete_state"] = None
                result["master_discrete_state"] = None
                result["is_mip"] = bool(state.operational_integer_names)
                result["operational_mip_variables"] = list(state.operational_integer_names)

            state.cleanup_iteration()
            if memory_coordinator is not None:
                memory_coordinator.set_reservation(worker_index, temporary=0)
            output_queue.put((key, result))
    finally:
        guard.release()
        if memory_coordinator is not None:
            memory_coordinator.release(worker_index)
        if task_directory:
            import shutil
            shutil.rmtree(task_directory, ignore_errors=True)
