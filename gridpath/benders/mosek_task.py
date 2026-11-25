"""Native MOSEK task persistence and runtime helpers for Benders workers.

Design summary
--------------
* The operational Pyomo model is compiled ONCE into a MOSEK ``Task`` (through
  Pyomo's MOSEK interface, so the rows/columns are exactly those the Pyomo
  path would send) and written to ``<subproblem dir>/benders_task/subproblem.task``.
* Only a small metadata dictionary stays in RAM: linking rows/columns, the
  integer/binary columns (the only columns whose type or bounds Benders ever
  changes) and their original bounds.  No Pyomo component or name map for the
  (very large) set of ordinary operational variables is retained.
* ``MosekTaskRuntime`` can run in two modes:
    - resident (``offload=False``): the Task stays in RAM between iterations and
      only the touched columns are reset after each trial;
    - offloaded (``offload=True``): the Task is disposed after every trial and
      re-read from the .task file on the next one, so an idle worker holds
      (almost) no model memory.  Because every trial starts from a freshly
      read task, no bounds/types ever need restoring.
* Solver parameters are NOT stored reliably in a task file by every code path,
  so they are re-applied with ``putparam`` after every load and before every
  stage (MIP stage / LP stage).
"""
import gc
import json
import math
import os
from pathlib import Path

import numpy as np

METADATA_VERSION = 2

_OPTIMAL_NAMES = ("optimal", "near_optimal", "integer_optimal", "near_integer_optimal")


def _finite_value(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _i32(values):
    """Contiguous int32 array: what MOSEK's putaijlist expects (no copy warning)."""
    return np.ascontiguousarray(values, dtype=np.int32)


def _mosek_enum_name(value):
    return str(value).split(".")[-1]


def write_pyomo_mosek_task(
    model,
    solver,
    task_path,
    linking_constraint_map,
    objective_scale=1.0,
    metadata_path=None,
    release_pyomo_maps=True,
):
    """Compile an instantiated Pyomo model, persist its MOSEK task, return it.

    The caller must apply objective scaling before compilation and must have
    added the linking equality rows.  ``solver`` should be a
    ``mosek_persistent`` (or ``mosek_direct``) Pyomo solver instance.

    Returns ``(task, metadata, env)``.  The caller owns ``task`` and must keep
    ``env`` alive for as long as any task created from it exists.
    """
    if not math.isfinite(float(objective_scale)) or objective_scale <= 0:
        raise ValueError("objective_scale must be finite and positive.")
    task_path = Path(task_path)
    if task_path.suffix.lower() != ".task":
        raise ValueError(f"Expected a .task destination, got {task_path!s}.")
    metadata_path = Path(metadata_path) if metadata_path else Path(str(task_path) + ".json")
    task_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # ``set_instance`` exists only on persistent Pyomo solvers; MOSEKDirect
    # exposes the same behaviour through the private ``_set_instance``.
    set_instance = getattr(solver, "set_instance", None)
    if set_instance is not None:
        set_instance(model, symbolic_solver_labels=False)
    else:
        solver._set_instance(model, {"symbolic_solver_labels": False})
    task = getattr(solver, "_solver_model", None)
    if task is None:
        raise RuntimeError("Pyomo MOSEK did not create a native Task object.")
    env = getattr(solver, "_mosek_env", None)
    var_map = getattr(solver, "_pyomo_var_to_solver_var_map", None)
    con_map = getattr(solver, "_pyomo_con_to_solver_con_map", None)
    if var_map is None or con_map is None:
        raise RuntimeError("Pyomo MOSEK interface lacks the required index maps.")

    linking_rows = {}
    for name, constraint in linking_constraint_map.items():
        try:
            linking_rows[str(name)] = int(con_map[constraint])
        except KeyError as exc:
            raise RuntimeError(
                f"Linking constraint {name!r} was not compiled into MOSEK."
            ) from exc

    from pyomo.environ import Var
    import mosek

    linking_columns = {}
    linking_bounds = {}
    integer_columns = {}
    for variable in model.component_data_objects(Var, active=True, descend_into=True):
        try:
            column = int(var_map[variable])
        except KeyError:
            continue
        if variable.name in linking_rows:
            linking_columns[variable.name] = column
            lb, ub = variable.bounds
            linking_bounds[variable.name] = [_finite_value(lb), _finite_value(ub)]
        if not variable.fixed and (variable.is_binary() or variable.is_integer()):
            integer_columns[variable.name] = {
                "column": column,
                "binary": bool(variable.is_binary()),
            }
    missing = set(linking_rows).difference(linking_columns)
    if missing:
        raise RuntimeError(
            f"Linking variables not compiled into MOSEK: {sorted(missing)[:5]}"
        )

    # Original native bounds of the columns Benders may modify.
    touched = {info["column"] for info in integer_columns.values()}
    touched.update(linking_columns.values())
    native_bounds = {}
    for column in sorted(touched):
        key, lower, upper = task.getvarbound(column)
        native_bounds[str(column)] = [_mosek_enum_name(key), float(lower), float(upper)]

    # Row -> constraint-group map (compressed to row ranges) so the native
    # Phase-I diagnostic can report which constraint families are violated.
    row_groups = {}
    try:
        by_group = {}
        for constraint, row in con_map.items():
            by_group.setdefault(
                constraint.parent_component().local_name, []
            ).append(int(row))
        for group, rows in by_group.items():
            rows.sort()
            ranges = []
            start = previous = rows[0]
            for row in rows[1:]:
                if row != previous + 1:
                    ranges.append([start, previous + 1])
                    start = row
                previous = row
            ranges.append([start, previous + 1])
            row_groups[group] = ranges
        del by_group
    except Exception:
        row_groups = {}

    # Column -> variable-group map, same idea as the row groups: it lets the
    # unbounded-ray diagnostic name the variable families involved.
    column_groups = {}
    try:
        by_group = {}
        for variable, column in var_map.items():
            by_group.setdefault(
                variable.parent_component().local_name, []
            ).append(int(column))
        for group, columns in by_group.items():
            columns.sort()
            ranges = []
            start = previous = columns[0]
            for column in columns[1:]:
                if column != previous + 1:
                    ranges.append([start, previous + 1])
                    start = column
                previous = column
            ranges.append([start, previous + 1])
            column_groups[group] = ranges
        del by_group
    except Exception:
        column_groups = {}

    task.writedata(str(task_path))
    metadata = {
        "version": METADATA_VERSION,
        "task_file": task_path.name,
        "objective_scale": float(objective_scale),
        "num_variables": int(task.getnumvar()),
        "num_constraints": int(task.getnumcon()),
        "linking_rows": linking_rows,
        "linking_columns": linking_columns,
        "linking_bounds": linking_bounds,
        "integer_columns": integer_columns,
        "native_bounds": native_bounds,
        "row_groups": row_groups,
        "column_groups": column_groups,
    }
    temporary_metadata = Path(str(metadata_path) + ".tmp")
    temporary_metadata.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)

    if release_pyomo_maps:
        for attribute in (
            "_pyomo_var_to_solver_var_map",
            "_solver_var_to_pyomo_var_map",
            "_pyomo_con_to_solver_con_map",
            "_solver_con_to_pyomo_con_map",
            "_vars_referenced_by_con",
            "_vars_referenced_by_obj",
            "_referenced_variables",
        ):
            mapping = getattr(solver, attribute, None)
            if hasattr(mapping, "clear"):
                mapping.clear()
        solver._pyomo_model = None
        solver._solver_model = None
    del mosek
    return task, metadata, env


class MosekTaskRuntime:
    """Index-based native Task wrapper with no Pyomo component references."""

    def __init__(
        self,
        task_path,
        metadata,
        env=None,
        mosek_module=None,
        params=None,
        task=None,
        offload=False,
    ):
        if mosek_module is None:
            import mosek as mosek_module
        self.mosek = mosek_module
        self.task_path = str(task_path)
        self.metadata = metadata
        # The Env must outlive every Task created from it.
        self.env = env if env is not None else mosek_module.Env()
        self.task = task
        self.offload = bool(offload)
        # MOSEK's Python binding requires string parameter values.
        self.base_params = {str(k): str(v) for k, v in (params or {}).items()}
        self.solution_type = None

        self.objective_scale = float(metadata["objective_scale"])
        self.linking_rows = dict(metadata["linking_rows"])
        self.linking_columns = dict(metadata["linking_columns"])
        self.linking_bounds = dict(metadata["linking_bounds"])
        self.integer_columns = dict(metadata["integer_columns"])
        self.native_bounds = dict(metadata["native_bounds"])
        self.row_groups = dict(metadata.get("row_groups") or {})
        self.column_groups = dict(metadata.get("column_groups") or {})
        self._elastic_first = None
        self._elastic_weights = None
        self._touched_columns = sorted(int(c) for c in self.native_bounds)
        if self.task is not None:
            self._apply_params({})

    # ------------------------------------------------------------------
    # Lifetime
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, task_path, metadata_path=None, env=None, mosek_module=None,
             params=None, offload=False):
        task_path = Path(task_path)
        metadata_path = Path(metadata_path) if metadata_path else Path(str(task_path) + ".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("version") != METADATA_VERSION:
            raise ValueError(
                "Unsupported MOSEK task metadata version: " + repr(metadata.get("version"))
            )
        runtime = cls(task_path, metadata, env=env, mosek_module=mosek_module,
                      params=params, offload=offload)
        runtime.open()
        return runtime

    def open(self):
        """Read the task from disk if it is not resident."""
        if self.task is not None:
            return
        task = self.env.Task()
        task.readdata(self.task_path)
        if (task.getnumvar() != self.metadata["num_variables"]
                or task.getnumcon() != self.metadata["num_constraints"]):
            raise RuntimeError("Loaded MOSEK task dimensions do not match metadata.")
        self.task = task
        self._apply_params({})

    def close(self):
        """Dispose the resident task (no-op if already released)."""
        task, self.task = self.task, None
        self.solution_type = None
        if task is None:
            return
        try:
            task.__exit__(None, None, None)
        except Exception:
            pass
        del task
        gc.collect()

    def end_trial(self):
        """Called after every trial: dispose (offload) or reset touched columns."""
        if self.offload or self._elastic_first is not None:
            # An elastic trial adds columns; never reuse that task.
            self._elastic_first = None
            self.close()
        else:
            self.reset_touched()

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def _apply_params(self, extra):
        params = dict(self.base_params)
        params.update({str(k): str(v) for k, v in (extra or {}).items()})
        for name, value in params.items():
            self.task.putparam(name, value)

    def set_stage_params(self, extra=None):
        """Re-apply base parameters plus stage-specific overrides."""
        self._apply_params(extra)

    # ------------------------------------------------------------------
    # Model edits (all bulk)
    # ------------------------------------------------------------------
    def clip_linking_value(self, name, target):
        lb, ub = self.linking_bounds[name]
        if lb is not None:
            target = max(target, lb)
        if ub is not None:
            target = min(target, ub)
        return target

    def set_linking_values(self, values):
        missing = set(self.linking_rows).difference(values)
        if missing:
            raise KeyError(f"Missing Benders linking values: {sorted(missing)[:5]}")
        names = list(self.linking_rows)
        rows = [self.linking_rows[n] for n in names]
        rhs = [float(values[n]) for n in names]
        if rows:
            self.task.putconboundlist(
                rows, [self.mosek.boundkey.fx] * len(rows), rhs, rhs
            )

    def set_integrality(self, names, integer):
        columns = [self.integer_columns[n]["column"] for n in names]
        if not columns:
            return
        target = (
            self.mosek.variabletype.type_int if integer
            else self.mosek.variabletype.type_cont
        )
        self.task.putvartypelist(columns, [target] * len(columns))

    def fix_columns(self, values):
        """Fix integer columns at (already rounded) values."""
        names = list(values)
        columns = [self.integer_columns[n]["column"] for n in names]
        vals = [float(values[n]) for n in names]
        if columns:
            self.task.putvarboundlist(
                columns, [self.mosek.boundkey.fx] * len(columns), vals, vals
            )

    def reset_touched(self):
        """Restore original bounds and integrality of every touched column."""
        if self.task is None or not self._touched_columns:
            return
        keys, lowers, uppers = [], [], []
        for column in self._touched_columns:
            key, lower, upper = self.native_bounds[str(column)]
            keys.append(getattr(self.mosek.boundkey, key))
            lowers.append(lower)
            uppers.append(upper)
        self.task.putvarboundlist(self._touched_columns, keys, lowers, uppers)
        int_names = list(self.integer_columns)
        self.set_integrality(int_names, True)

    # ------------------------------------------------------------------
    # Solve and read
    # ------------------------------------------------------------------
    def _is_optimal(self, status):
        acceptable = {
            getattr(self.mosek.solsta, name)
            for name in _OPTIMAL_NAMES
            if hasattr(self.mosek.solsta, name)
        }
        return status in acceptable

    def optimize(self, mixed_integer=False):
        """Optimize; select the solution type that actually holds an optimum.

        Returns True when an optimal (or MOSEK ``near_optimal``) solution of
        the requested kind is available.  For LPs the basic solution is
        preferred, matching Pyomo's MOSEKDirect (bas for LP).
        """
        self.task.optimize()
        if mixed_integer:
            candidates = [self.mosek.soltype.itg]
        else:
            candidates = [self.mosek.soltype.bas, self.mosek.soltype.itr]
        self.solution_type = None
        for candidate in candidates:
            if self._is_optimal(self.task.getsolsta(candidate)):
                self.solution_type = candidate
                return True
        return False

    def status_summary(self):
        """Human-readable MOSEK problem/solution status for every solution type."""
        parts = []
        for label, candidate in (("bas", self.mosek.soltype.bas),
                                 ("itr", self.mosek.soltype.itr)):
            try:
                parts.append("{}: solsta={} prosta={}".format(
                    label,
                    _mosek_enum_name(self.task.getsolsta(candidate)),
                    _mosek_enum_name(self.task.getprosta(candidate)),
                ))
            except Exception as exc:  # a solution type may not exist
                parts.append("{}: unavailable ({})".format(label, type(exc).__name__))
        return "; ".join(parts)

    def proved_infeasible(self):
        """True when MOSEK holds a primal infeasibility certificate."""
        for candidate in (self.mosek.soltype.bas, self.mosek.soltype.itr):
            try:
                if self.task.getsolsta(candidate) == self.mosek.solsta.prim_infeas_cer:
                    return True
            except Exception:
                continue
        return False

    def dual_infeasible(self):
        """True when MOSEK holds a dual-infeasibility certificate (primal ray)."""
        for candidate in (self.mosek.soltype.bas, self.mosek.soltype.itr):
            try:
                if self.task.getsolsta(candidate) == self.mosek.solsta.dual_infeas_cer:
                    return candidate
            except Exception:
                continue
        return None

    def unbounded_ray_report(self, top=8):
        """Describe the primal ray of an unbounded LP, by variable family.

        The ray's scale is arbitrary, so everything is judged relative to the
        largest ray entry and to the total cost along the ray, never against
        an absolute tolerance.
        """
        which = self.dual_infeasible()
        if which is None:
            return "no dual-infeasibility certificate"
        x = np.asarray(self.task.getxx(which), dtype=float)
        n_var = len(x)
        c = np.asarray(self.task.getc(), dtype=float)
        contribution = c * x
        total = float(contribution.sum())
        xmax = float(np.abs(x).max()) if n_var else 0.0
        parts = [
            "total cost along ray {:.6g}, max|ray| {:.3g}, "
            "{} of {} columns nonzero".format(
                total, xmax, int((np.abs(x) > 1e-12 * max(xmax, 1e-300)).sum()), n_var
            )
        ]
        if self.column_groups:
            ranked = []
            covered = np.zeros(n_var, dtype=bool)
            for group, ranges in self.column_groups.items():
                cost = 0.0
                count = 0
                weight = 0.0
                for start, end in ranges:
                    end = min(end, n_var)
                    if end <= start:
                        continue
                    covered[start:end] = True
                    cost += float(contribution[start:end].sum())
                    weight += float(np.abs(x[start:end]).sum())
                    count += int((np.abs(x[start:end]) > 1e-9 * max(xmax, 1e-300)).sum())
                if count:
                    ranked.append((abs(cost), group, count, weight, cost))
            ranked.sort(reverse=True)
            parts.append("by family (largest |cost along ray| first): " + "; ".join(
                "{} ({} cols, sum|ray| {:.3g}, cost {:.3g})".format(g, k, w, cst)
                for _, g, k, w, cst in ranked[:top]
            ))
            extra = float(contribution[~covered].sum()) if (~covered).any() else 0.0
            if abs(extra) > 0:
                parts.append("columns outside any family: cost {:.3g}".format(extra))
        else:
            parts.append("column groups unavailable (re-export the task)")
        order = np.argsort(-np.abs(contribution))[:top]
        try:
            bk, bl, bu = self.task.getvarboundslice(0, n_var)
        except Exception:
            bl = bu = None
        details = []
        for j in order:
            if contribution[j] == 0.0:
                continue
            family = "?"
            for group, ranges in self.column_groups.items():
                if any(st <= j < en for st, en in ranges):
                    family = group
                    break
            bound = "" if bl is None else " bounds [{:.3g}, {:.3g}]".format(bl[j], bu[j])
            details.append("col {} {} ray {:.3g} cost {:.3g}{}".format(
                int(j), family, x[j], c[j], bound))
        if details:
            parts.append("top columns: " + "; ".join(details))
        return " | ".join(parts)

    def primal_values(self, names):
        solution = self.task.getxx(self.solution_type)
        return {
            name: float(solution[self.integer_columns[name]["column"]])
            for name in names
        }

    def linking_duals(self):
        y = self.task.gety(self.solution_type)
        return {
            name: -float(y[row]) * self.objective_scale
            for name, row in self.linking_rows.items()
        }

    def objective_value(self):
        return float(self.task.getprimalobj(self.solution_type)) * self.objective_scale

    def _rows_of_column(self, column):
        """Row indices with a non-zero in ``column`` (API-version tolerant)."""
        result = self.task.getacol(int(column))
        if len(result) == 3:      # (nzj, subj, valj)
            return [int(i) for i in result[1]]
        return [int(i) for i in result[0]]   # (subj, valj)

    def add_elastic_allocations(self, multiplier):
        """Let rows that use a policy allocation be violated at a penalty.

        Every non-linking row with a non-zero in an allocation column gets
        non-negative slack columns (``p`` for a lower bound, ``n`` for an
        upper bound), priced in the objective at
        ``multiplier * ||c||_1 * w_i`` per unit with
        ``w_i = 1 / max(1, |l_i|, |u_i|)``. The recourse problem is then
        always feasible with respect to the allocations. Returns the number
        of rows made elastic. The task is flagged so ``end_trial`` disposes
        it (the next trial re-reads the pristine task).
        """
        from gridpath.benders.allocation_names import is_allocation_name

        mosek = self.mosek
        task = self.task
        bk = mosek.boundkey
        columns = [
            int(c) for n, c in self.linking_columns.items()
            if is_allocation_name(n)
        ]
        if not columns:
            return 0
        rows = set()
        for column in columns:
            rows.update(self._rows_of_column(column))
        rows.difference_update(int(r) for r in self.linking_rows.values())
        if not rows:
            return 0
        rows = np.array(sorted(rows), dtype=np.int64)
        n_var = task.getnumvar()
        n_con = task.getnumcon()
        keys, lowers, uppers = task.getconboundslice(0, n_con)
        lower = np.array(lowers, dtype=float)[rows]
        upper = np.array(uppers, dtype=float)[rows]
        has_lower = np.array([keys[i] in (bk.lo, bk.ra, bk.fx) for i in rows])
        has_upper = np.array([keys[i] in (bk.up, bk.ra, bk.fx) for i in rows])
        del keys, lowers, uppers
        lower[~has_lower] = -np.inf
        upper[~has_upper] = np.inf
        weight = 1.0 / np.maximum.reduce([
            np.ones(len(rows)),
            np.where(np.isfinite(lower), np.abs(lower), 0.0),
            np.where(np.isfinite(upper), np.abs(upper), 0.0),
        ])
        c_norm = float(np.sum(np.abs(np.asarray(task.getc(), dtype=float))))
        if not math.isfinite(c_norm) or c_norm <= 0.0:
            c_norm = 1.0
        rows_p, w_p = rows[has_lower], weight[has_lower]
        rows_n, w_n = rows[has_upper], weight[has_upper]
        n_new = len(rows_p) + len(rows_n)
        if n_new == 0:
            return 0
        task.appendvars(n_new)
        task.putvarboundsliceconst(n_var, n_var + n_new, bk.lo, 0.0, 0.0)
        weights = np.concatenate([w_p, w_n])
        task.putcslice(n_var, n_var + n_new, float(multiplier) * c_norm * weights)
        if len(rows_p):
            task.putaijlist(_i32(rows_p), _i32(np.arange(n_var, n_var + len(rows_p))),
                            np.ones(len(rows_p), dtype=np.float64))
        if len(rows_n):
            task.putaijlist(_i32(rows_n), _i32(np.arange(n_var + len(rows_p), n_var + n_new)),
                            -np.ones(len(rows_n), dtype=np.float64))
        self._elastic_first = n_var
        self._elastic_weights = weights
        return int(len(rows))

    def elastic_violation(self):
        """Total relative violation used by the last elastic solve (0 if none)."""
        if self._elastic_first is None or self.solution_type is None:
            return 0.0
        x = np.asarray(self.task.getxx(self.solution_type), dtype=float)
        slack = x[self._elastic_first:self._elastic_first + len(self._elastic_weights)]
        return float(np.dot(self._elastic_weights, np.maximum(slack, 0.0)))

    def phase_one(self, active_names, point, key_label="", diag_groups=8):
        """Native Phase-I on the loaded task (replaces the Pyomo restoration).

        The loaded task must be an LP whose linking rows are already fixed at
        the master's trial point (``set_linking_values``) and whose integer
        columns are relaxed. It is modified IN PLACE and disposed before
        returning, so the next trial re-reads the pristine task from disk.

        Elastic LP: every non-linking row ``l <= a x <= u`` gets non-negative
        slack columns, ``l <= a x + p - n <= u`` (``p`` only if a lower bound
        exists, ``n`` only if an upper bound exists), and the objective is
        ``min sum_i w_i (p_i + n_i)`` with ``w_i = 1 / max(1, |l_i|, |u_i|)``.
        Linking rows and column bounds are kept exact, so the optimum ``v`` is
        a convex, piecewise-linear function of the linking right-hand sides
        that is zero exactly when the subproblem is feasible. The duals of the
        linking rows are therefore a valid sub-gradient and give the
        feasibility cut ``v + sum_j lambda_j (x_j - xbar_j) <= 0``.

        Returns the cut dictionary, or ``None`` when the Phase-I optimum is
        not trustworthy (not optimal, zero violation, or a poor duality gap),
        in which case the caller falls back to the Pyomo restoration.
        """
        import logging

        mosek = self.mosek
        task = self.task
        bk = mosek.boundkey
        n_var = task.getnumvar()
        n_con = task.getnumcon()
        if n_con == 0:
            return None

        keys, lowers, uppers = task.getconboundslice(0, n_con)
        lower = np.array(lowers, dtype=float)
        upper = np.array(uppers, dtype=float)
        has_lower = np.array([k in (bk.lo, bk.ra, bk.fx) for k in keys])
        has_upper = np.array([k in (bk.up, bk.ra, bk.fx) for k in keys])
        del keys, lowers, uppers
        lower[~has_lower] = -np.inf
        upper[~has_upper] = np.inf
        is_linking = np.zeros(n_con, dtype=bool)
        for row in self.linking_rows.values():
            is_linking[int(row)] = True
        has_lower &= ~is_linking
        has_upper &= ~is_linking

        magnitude = np.maximum.reduce([
            np.ones(n_con),
            np.where(np.isfinite(lower), np.abs(lower), 0.0),
            np.where(np.isfinite(upper), np.abs(upper), 0.0),
        ])
        weight = 1.0 / magnitude

        rows_p = np.flatnonzero(has_lower)
        rows_n = np.flatnonzero(has_upper)
        n_p, n_n = len(rows_p), len(rows_n)
        if n_p + n_n == 0:
            return None

        # Replace the objective by the total weighted slack.
        task.putcslice(0, n_var, np.zeros(n_var))
        task.putcfix(0.0)
        task.putobjsense(mosek.objsense.minimize)
        task.appendvars(n_p + n_n)
        task.putvarboundsliceconst(n_var, n_var + n_p + n_n, bk.lo, 0.0, 0.0)
        cols_p = np.arange(n_var, n_var + n_p)
        cols_n = np.arange(n_var + n_p, n_var + n_p + n_n)
        cost = np.concatenate([weight[rows_p], weight[rows_n]])
        task.putcslice(n_var, n_var + n_p + n_n, cost)
        if n_p:
            task.putaijlist(_i32(rows_p), _i32(cols_p), np.ones(n_p, dtype=np.float64))
        if n_n:
            task.putaijlist(_i32(rows_n), _i32(cols_n), -np.ones(n_n, dtype=np.float64))

        self.set_stage_params({})
        try:
            if not self.optimize(mixed_integer=False):
                logging.getLogger(__name__).warning(
                    "Native Phase-I for %s did not solve to optimality (%s).",
                    key_label, self.status_summary(),
                )
                return None
            sol = self.solution_type
            primal = float(task.getprimalobj(sol))
            try:
                dual = float(task.getdualobj(sol))
            except Exception:
                dual = primal
            if not (math.isfinite(primal) and math.isfinite(dual)):
                return None
            intercept = min(primal, dual)
            gap = abs(primal - dual) / max(1.0, abs(primal))
            if intercept <= 1e-9:
                # Feasible after all (numerical trouble in the original LP).
                return None
            if gap > 1e-6:
                logging.getLogger(__name__).warning(
                    "Native Phase-I for %s: duality gap %.3g too large.",
                    key_label, gap,
                )
                return None
            y = np.asarray(task.gety(sol), dtype=float)
            lambdas = {}
            for name in active_names:
                value = float(y[int(self.linking_rows[name])])
                if not math.isfinite(value):
                    return None
                lambdas[name] = value

            if os.environ.get("GRIDPATH_BENDERS_PHASE1_DIAG", "1") != "0":
                try:
                    self._log_phase_one_diagnostic(
                        key_label, intercept, np.asarray(task.getxx(sol), dtype=float),
                        n_var, n_con, rows_p, rows_n, weight, lambdas, diag_groups,
                    )
                except Exception as error:  # diagnostic only
                    logging.getLogger(__name__).warning(
                        "Phase-I diagnostic failed for %s: %s", key_label, error
                    )
        finally:
            # The task now carries the elastic LP; never reuse it.
            self.close()

        cut_scale = max(1.0, 1.0 / intercept)
        scaled = {n: v * cut_scale for n, v in lambdas.items()}
        return {
            "feasible": False,
            "op_cost": intercept * cut_scale,
            "benders_lambda": scaled,
            "continuous_lambda": dict(scaled),
            "continuous_point": dict(point),
            "theta_coeff": 0.0,
            "restoration_slack": intercept,
            "restoration_cut_scale": cut_scale,
            "restoration_dual_gap": gap,
        }

    def _log_phase_one_diagnostic(
        self, key_label, intercept, x, n_var, n_con, rows_p, rows_n, weight,
        lambdas, top,
    ):
        """Log which constraint families (and linking variables) drive Phase-I."""
        import logging

        n_p = len(rows_p)
        slack_p = x[n_var:n_var + n_p]
        slack_n = x[n_var + n_p:n_var + n_p + len(rows_n)]
        violation = np.zeros(n_con)
        np.add.at(violation, rows_p, slack_p)
        np.add.at(violation, rows_n, slack_n)
        weighted = violation * weight
        report = []
        if self.row_groups:
            for group, ranges in self.row_groups.items():
                total = 0.0
                count = 0
                worst = 0.0
                for start, end in ranges:
                    block = weighted[start:end]
                    total += float(block.sum())
                    count += int((block > 1e-9).sum())
                    if block.size:
                        worst = max(worst, float(block.max()))
                if total > 1e-9:
                    report.append((total, group, count, worst))
            report.sort(reverse=True)
            groups_text = "; ".join(
                "{} (sum {:.3g}, {} rows, max {:.3g})".format(g, t, c, w)
                for t, g, c, w in report[:top]
            ) or "none above tolerance"
        else:
            groups_text = "row groups unavailable (re-export the task)"
        ranked = sorted(lambdas.items(), key=lambda item: -abs(item[1]))[:5]
        links_text = "; ".join("{}: {:.3g}".format(n, v) for n, v in ranked)
        logging.getLogger(__name__).warning(
            "Phase-I diagnostic %s: violation %.6g. Violated constraint groups: "
            "%s. Linking variables with largest |sensitivity|: %s",
            key_label, intercept, groups_text, links_text,
        )

    def primal_violation(self, tolerance=1e-7):
        """Native replacement for the Pyomo ``primal_violations`` check.

        Computes, with NumPy and without any Pyomo object, the worst relative
        violation of (a) every row activity against its bounds, (b) every
        column value against its bounds, and (c) integrality of columns that
        are still integer.  Returns ``(ok, worst_relative_violation)``.
        """
        sol = self.solution_type
        n_var = self.task.getnumvar()
        n_con = self.task.getnumcon()
        worst = 0.0

        def _bound_arrays(keys, lowers, uppers):
            lower = np.array(lowers, dtype=float)
            upper = np.array(uppers, dtype=float)
            bk = self.mosek.boundkey
            no_lower = np.array([k in (bk.up, bk.fr) for k in keys])
            no_upper = np.array([k in (bk.lo, bk.fr) for k in keys])
            lower[no_lower] = -np.inf
            upper[no_upper] = np.inf
            return lower, upper

        def _worst(values, lower, upper):
            scale = np.maximum.reduce([
                np.ones_like(values),
                np.where(np.isfinite(lower), np.abs(lower), 0.0),
                np.where(np.isfinite(upper), np.abs(upper), 0.0),
            ])
            below = np.where(np.isfinite(lower), lower - values, 0.0)
            above = np.where(np.isfinite(upper), values - upper, 0.0)
            violation = np.maximum(0.0, np.maximum(below, above)) / scale
            return float(violation.max()) if violation.size else 0.0

        if n_con:
            xc = self.task.getxc(sol)
            keys, lowers, uppers = self.task.getconboundslice(0, n_con)
            lower, upper = _bound_arrays(keys, lowers, uppers)
            worst = max(worst, _worst(np.asarray(xc), lower, upper))
        xx = np.asarray(self.task.getxx(sol))
        keys, lowers, uppers = self.task.getvarboundslice(0, n_var)
        lower, upper = _bound_arrays(keys, lowers, uppers)
        worst = max(worst, _worst(xx, lower, upper))
        finite = math.isfinite(worst) and bool(np.all(np.isfinite(xx)))
        return (finite and worst <= tolerance), worst
