"""Native MOSEK task persistence and runtime helpers for Benders workers."""
import json
import math
import os
from pathlib import Path

METADATA_VERSION = 1

def _finite_value(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None

def _solver_index_map(solver, attribute):
    mapping = getattr(solver, attribute, None)
    if mapping is None:
        raise RuntimeError(f"Pyomo MOSEK interface lacks required map {attribute!r}.")
    return mapping

def write_pyomo_mosek_task(model, solver, task_path, linking_constraint_map, objective_scale=1.0, metadata_path=None, release_pyomo_maps=True):
    """Compile an instantiated Pyomo model and persist its native MOSEK task.

    The caller must apply objective scaling before compilation. Linking
    equality rows must already exist in linking_constraint_map. This function
    does not mutate the objective or solve the task.
    """
    if not math.isfinite(float(objective_scale)) or objective_scale <= 0:
        raise ValueError("objective_scale must be finite and positive.")
    task_path = Path(task_path)
    if task_path.suffix.lower() != ".task":
        raise ValueError(f"Expected a .task destination, got {task_path!s}.")
    metadata_path = Path(metadata_path) if metadata_path else Path(str(task_path) + ".json")
    task_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    solver.set_instance(model, symbolic_solver_labels=False)
    task = getattr(solver, "_solver_model", None)
    if task is None:
        raise RuntimeError("Pyomo MOSEK did not create a native Task object.")
    var_map = _solver_index_map(solver, "_pyomo_var_to_solver_var_map")
    con_map = _solver_index_map(solver, "_pyomo_con_to_solver_con_map")
    linking_rows = {}
    for name, constraint in linking_constraint_map.items():
        try:
            linking_rows[str(name)] = int(con_map[constraint])
        except KeyError as exc:
            raise RuntimeError(f"Linking constraint {name!r} was not compiled into MOSEK.") from exc
    from pyomo.environ import Var
    variable_columns = {}
    variable_metadata = {}
    for variable in model.component_data_objects(Var, active=True, descend_into=True):
        try:
            column = int(var_map[variable])
        except KeyError:
            continue
        variable_columns[variable.name] = column
        variable_metadata[variable.name] = {
            "column": column, "lb": _finite_value(variable.lb), "ub": _finite_value(variable.ub),
            "fixed": bool(variable.fixed),
            "fixed_value": _finite_value(variable.value) if variable.fixed else None,
            "binary": bool(variable.is_binary()),
            "integer": bool(variable.is_integer()),
        }
    task.writedata(str(task_path))
    metadata = {
        "version": METADATA_VERSION, "task_file": task_path.name,
        "objective_scale": float(objective_scale),
        "num_variables": int(task.getnumvar()), "num_constraints": int(task.getnumcon()),
        "linking_rows": linking_rows, "variable_columns": variable_columns,
        "variables": variable_metadata,
    }
    temporary_metadata = Path(str(metadata_path) + ".tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)
    if release_pyomo_maps:
        for attribute in (
            "_pyomo_var_to_solver_var_map", "_solver_var_to_pyomo_var_map",
            "_pyomo_con_to_solver_con_map", "_solver_con_to_pyomo_con_map",
            "_vars_referenced_by_con", "_vars_referenced_by_obj", "_referenced_variables",
        ):
            mapping = getattr(solver, attribute, None)
            if hasattr(mapping, "clear"):
                mapping.clear()
        solver._pyomo_model = None
    return task, metadata

class MosekTaskRuntime:
    """Index-based native Task wrapper with no Pyomo component references."""
    def __init__(self, task, metadata, mosek_module=None, env=None, solution_type=None):
        if mosek_module is None:
            import mosek as mosek_module
        self.mosek = mosek_module
        self.task = task
        self.env = env
        self.metadata = metadata
        # Do not assume a basic solution exists. MOSEK's automatic LP
        # optimizer may return an interior-point solution, while the previous
        # implementation hard-coded soltype.bas and then queried an undefined
        # solution. An explicit solution_type remains available for callers
        # that deliberately configure a particular optimizer.
        self.solution_type = solution_type
        self._available_solution_types = (
            [solution_type] if solution_type is not None
            else [
                candidate
                for candidate in (
                    getattr(self.mosek.soltype, "itr", None),
                    getattr(self.mosek.soltype, "bas", None),
                )
                if candidate is not None
            ]
        )
        self.linking_rows = dict(metadata["linking_rows"])
        self.variable_columns = dict(metadata["variable_columns"])
        self.objective_scale = float(metadata["objective_scale"])

    @classmethod
    def load(cls, task_path, metadata_path=None, env=None, mosek_module=None):
        if mosek_module is None:
            import mosek as mosek_module
        task_path = Path(task_path)
        metadata_path = Path(metadata_path) if metadata_path else Path(str(task_path) + ".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("version") != METADATA_VERSION:
            raise ValueError("Unsupported MOSEK task metadata version: " + repr(metadata.get("version")))
        env = env or mosek_module.Env()
        task = env.Task()
        task.readdata(str(task_path))
        if task.getnumvar() != metadata["num_variables"] or task.getnumcon() != metadata["num_constraints"]:
            raise RuntimeError("Loaded MOSEK task dimensions do not match metadata.")
        return cls(task, metadata, mosek_module=mosek_module, env=env)

    def set_linking_values(self, values):
        missing = set(self.linking_rows).difference(values)
        if missing:
            raise KeyError(f"Missing Benders linking values: {sorted(missing)}")
        for name, row in self.linking_rows.items():
            rhs = float(values[name])
            self.task.putconbound(row, self.mosek.boundkey.fx, rhs, rhs)

    def optimize(self):
        """Optimize and select an actually available continuous LP solution.

        MOSEK can populate either the interior-point (itr) or basic (bas)
        solution depending on its optimizer. Check both rather than assuming
        that a basic solution was produced.
        """
        self.task.optimize()
        acceptable = {
            self.mosek.solsta.optimal,
            getattr(self.mosek.solsta, "near_optimal", None),
        }
        acceptable.discard(None)
        last_status = None
        for candidate in self._available_solution_types:
            status = self.task.getsolsta(candidate)
            last_status = status
            if status in acceptable:
                self.solution_type = candidate
                return status
        # Preserve a meaningful status for infeasible/unknown outcomes.
        if self._available_solution_types:
            self.solution_type = self._available_solution_types[0]
        return last_status

    def has_primal_solution(self):
        """Whether the selected solution type has a usable optimal LP point."""
        if self.solution_type is None:
            return False
        acceptable = {
            self.mosek.solsta.optimal,
            getattr(self.mosek.solsta, "near_optimal", None),
        }
        acceptable.discard(None)
        return self.task.getsolsta(self.solution_type) in acceptable

    def solve(self, values, require_optimal=True):
        """Update linking rows, solve, and return objective/primal/dual data.

        Objective and linking duals are returned in the original, unscaled
        objective units. No primal or dual values are read for an infeasible
        or otherwise non-optimal solution.
        """
        self.set_linking_values(values)
        status = self.optimize()
        result = {
            "solution_status": str(status),
            "has_primal_solution": self.has_primal_solution(),
        }
        if not result["has_primal_solution"]:
            if require_optimal:
                raise RuntimeError(
                    "Native MOSEK task did not produce an optimal primal "
                    f"solution (solution status: {status})."
                )
            return result
        result["objective_value"] = self.objective_value()
        result["primal_values"] = self.primal_values()
        result["linking_duals"] = self.linking_duals()
        return result

    def primal_values(self, names=None):
        names = list(self.variable_columns) if names is None else list(names)
        columns = [self.variable_columns[name] for name in names]
        values = [0.0] * self.task.getnumvar()
        if columns:
            self.task.getxx(self.solution_type, values)
        return {name: float(values[column]) for name, column in zip(names, columns)}

    def linking_duals(self):
        values = [0.0] * self.task.getnumcon()
        self.task.gety(self.solution_type, values)
        return {name: -float(values[row]) * self.objective_scale for name, row in self.linking_rows.items()}

    def objective_value(self):
        return float(self.task.getprimalobj(self.solution_type)) * self.objective_scale
