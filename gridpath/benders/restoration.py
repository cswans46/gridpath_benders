"""Numerically checked LP solves for operational recourse and Phase I."""

import json
import logging
import math
import os

import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn

RESTORATION_OBJECTIVE_SCALE = 1e6
PRIMAL_TOLERANCE = 1e-7
CERTIFICATE_TOLERANCE = 1e-8
ZERO_SLACK_TOLERANCE = 1e-9


def _finite(value, name):
    number = pyo.value(value, exception=False)
    if number is None or not math.isfinite(float(number)):
        raise ValueError(f"Missing or non-finite value for {name}: {number}")
    return float(number)


def _bound(bound, name):
    if bound is None:
        return None
    number = float(pyo.value(bound))
    if math.isnan(number):
        raise ValueError(f"NaN bound for {name}")
    return number if math.isfinite(number) else None


def primal_violations(model, tolerance=PRIMAL_TOLERANCE, *, backward_error=False):
    """Check original rows, bounds, linking equalities, and integrality."""
    violations = []
    used_vars = {}
    for row in model.component_data_objects(pyo.Constraint, active=True):
        lower = _bound(row.lower, row.name)
        upper = _bound(row.upper, row.name)
        if lower is None and upper is None:
            continue
        repn = generate_standard_repn(row.body)
        if not repn.is_linear():
            raise ValueError(f"Benders LP validation requires a linear row: {row.name}")
        for var in repn.linear_vars:
            used_vars[id(var)] = var
        body = _finite(row.body, row.name)
        scale = max(
            1.0,
            abs(lower or 0),
            abs(upper or 0),
            sum(abs(float(coef)) for coef in repn.linear_coefs),
        )
        if backward_error:
            scale = max(
                scale,
                sum(
                    abs(float(coef) * _finite(var, var.name))
                    for var, coef in zip(repn.linear_vars, repn.linear_coefs)
                ),
            )
        violation = max(
            0.0,
            lower - body if lower is not None else 0.0,
            body - upper if upper is not None else 0.0,
        )
        if violation > tolerance * scale:
            violations.append((row.name, violation, violation / scale))
    for objective in model.component_data_objects(pyo.Objective, active=True):
        repn = generate_standard_repn(objective.expr)
        for var in repn.linear_vars:
            used_vars[id(var)] = var
        _finite(objective.expr, objective.name)
    for var in model.component_data_objects(pyo.Var):
        if var.value is None and id(var) not in used_vars and not var.fixed:
            continue
        number = _finite(var, var.name)
        lower, upper = _bound(var.lb, var.name), _bound(var.ub, var.name)
        scale = max(1.0, abs(lower or 0), abs(upper or 0))
        violation = max(
            0.0,
            lower - number if lower is not None else 0.0,
            number - upper if upper is not None else 0.0,
        )
        if var.is_integer():
            violation = max(violation, abs(number - round(number)))
            scale = 1.0
        if violation > tolerance * scale:
            violations.append((var.name, violation, violation / scale))
    return sorted(violations, key=lambda item: item[2], reverse=True)


def _scale_rows(model):
    """Use coefficient scales, never trial-point RHS magnitudes."""
    scales = {}
    for row in model.component_data_objects(pyo.Constraint, active=True):
        lower, upper = _bound(row.lower, row.name), _bound(row.upper, row.name)
        if lower is None and upper is None:
            row.deactivate()
            continue
        repn = generate_standard_repn(row.body)
        if not repn.is_linear():
            raise ValueError(f"Restoration requires a linear row: {row.name}")
        _finite(repn.constant, row.name)
        scale = max(
            [1.0] + [abs(_finite(coef, row.name)) for coef in repn.linear_coefs]
        )
        scales[row.name] = scale
        row.set_value(
            (
                lower / scale if lower is not None else None,
                row.body / scale,
                upper / scale if upper is not None else None,
            )
        )
    return scales


def _certificate(model):
    """Validate a minimization LP's primal/dual certificate in solver units."""
    violations = primal_violations(model, CERTIFICATE_TOLERANCE, backward_error=True)
    if violations:
        raise ValueError(f"LP primal residuals: {violations[:5]}")
    objective = next(model.component_data_objects(pyo.Objective, active=True))
    if objective.sense != pyo.minimize:
        raise ValueError("Restoration certificate requires a minimization objective")
    repn = generate_standard_repn(objective.expr)
    gradient = {
        id(var): float(coef) for var, coef in zip(repn.linear_vars, repn.linear_coefs)
    }
    magnitude = {key: abs(coef) for key, coef in gradient.items()}
    dual_objective = float(repn.constant or 0)
    complementarity = 0.0
    for row in model.component_data_objects(pyo.Constraint, active=True):
        dual = _finite(model.dual.get(row), f"dual of {row.name}")
        lower, upper = _bound(row.lower, row.name), _bound(row.upper, row.name)
        selected = lower if dual >= 0 else upper
        if selected is None:
            if abs(dual) > CERTIFICATE_TOLERANCE:
                raise ValueError(f"Wrong dual sign for {row.name}: {dual}")
            selected = 0.0
        repn = generate_standard_repn(row.body)
        dual_objective += dual * (selected - float(repn.constant or 0))
        complementarity += abs(dual * (_finite(row.body, row.name) - selected))
        for var, coef in zip(repn.linear_vars, repn.linear_coefs):
            term = dual * float(coef)
            gradient[id(var)] = gradient.get(id(var), 0.0) - term
            magnitude[id(var)] = magnitude.get(id(var), 0.0) + abs(term)
    for var in model.component_data_objects(pyo.Var):
        if var.fixed or id(var) not in gradient:
            continue
        rc = _finite(model.rc.get(var), f"reduced cost of {var.name}")
        residual = gradient[id(var)] - rc
        if abs(residual) > CERTIFICATE_TOLERANCE * max(
            1.0, magnitude[id(var)], abs(rc)
        ):
            raise ValueError(f"Dual stationarity residual for {var.name}: {residual}")
        selected = _bound(var.lb if rc >= 0 else var.ub, var.name)
        if selected is None:
            if abs(rc) > CERTIFICATE_TOLERANCE:
                raise ValueError(f"Wrong reduced-cost sign for {var.name}: {rc}")
        else:
            dual_objective += rc * selected
            complementarity += abs(rc * (_finite(var, var.name) - selected))
    dual_objective = _finite(dual_objective, "dual objective")
    complementarity = _finite(complementarity, "complementarity")
    primal_objective = _finite(objective.expr, objective.name)
    scale = max(1.0, abs(primal_objective), abs(dual_objective))
    gap = abs(primal_objective - dual_objective)
    if max(gap, complementarity) > CERTIFICATE_TOLERANCE * scale:
        raise ValueError(
            f"LP certificate gap={gap}, complementarity={complementarity}, scale={scale}"
        )
    return dual_objective, gap


def _options(solver):
    name = str(getattr(solver, "name", "")).lower()
    if name.startswith("mosek"):
        return {
            "MSK_DPAR_BASIS_TOL_X": "1e-9",
            "MSK_DPAR_INTPNT_TOL_PFEAS": "1e-10",
            "MSK_DPAR_INTPNT_TOL_DFEAS": "1e-10",
            "MSK_DPAR_INTPNT_TOL_REL_GAP": "1e-10",
        }
    if name.startswith("gurobi"):
        return {"FeasibilityTol": 1e-9, "OptimalityTol": 1e-9, "BarConvTol": 1e-10}
    if "highs" in name:
        return {
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        }
    return {}


def _copy_solution(source, target, scales, objective_scale):
    for var in source.component_data_objects(pyo.Var):
        if var.value is not None:
            target.find_component(var.name).set_value(var.value, skip_validation=True)
    target.dual.clear()
    for row in source.component_data_objects(pyo.Constraint, active=True):
        dual = source.dual.get(row)
        if dual is not None:
            target.dual[target.find_component(row.name)] = (
                float(dual) / scales[row.name] / objective_scale
            )


def solve_original(model, solver):
    """Retry the unchanged operational LP on scaled, disposable clones."""
    from gridpath.benders.subproblems import (
        BENDERS_RECOURSE_OBJECTIVE_SCALE,
        _solve_with_lp_optimizer,
        prepare_lp_for_duals,
    )

    failures = []
    for algorithm in ("primal_simplex", "interior_point"):
        # Release the previous attempt's clone before allocating another.
        clone = None
        clone = model.clone()
        prepare_lp_for_duals(
            clone,
            present_linking_vars=getattr(clone, "BendersLinkingConstraintMap", {}),
            operational_discrete_method="fixed_z",
        )
        scales = _scale_rows(clone)
        clone.OP_NPV.set_value(clone.OP_NPV.expr / BENDERS_RECOURSE_OBJECTIVE_SCALE)
        clone.dual.clear()
        result = _solve_with_lp_optimizer(clone, solver, algorithm, _options(solver))
        term = result.solver.termination_condition
        if term == pyo.TerminationCondition.optimal:
            violations = primal_violations(clone)
            if not violations:
                _copy_solution(clone, model, scales, 1.0)
                violations = primal_violations(model)
                if not violations:
                    return result, BENDERS_RECOURSE_OBJECTIVE_SCALE
            failures.append(f"{algorithm}: original residuals {violations[:5]}")
        elif term == pyo.TerminationCondition.infeasible:
            return None
        else:
            failures.append(f"{algorithm}: {term}")
        logging.getLogger(__name__).warning("Operational retry: %s", failures[-1])
    raise RuntimeError(f"Original operational LP retry was inconclusive: {failures}")


def restore(model, solver, active_names, point, subproblem_key):
    """Return a validated Phase-I feasibility cut, without changing the model."""
    from gridpath.benders.subproblems import (
        add_slacks_to_subproblem,
        _solve_with_lp_optimizer,
        prepare_lp_for_duals,
    )

    failures = []
    for algorithm in ("primal_simplex", "interior_point"):
        # Release the previous attempt's clone before allocating another.
        clone = None
        clone = model.clone()
        prepare_lp_for_duals(
            clone,
            present_linking_vars=getattr(clone, "BendersLinkingConstraintMap", {}),
            operational_discrete_method="fixed_z",
        )
        for objective in clone.component_data_objects(pyo.Objective, active=True):
            objective.deactivate()
        add_slacks_to_subproblem(clone)
        clone.slack_fix.deactivate()
        clone.feasibility_obj = pyo.Objective(
            expr=RESTORATION_OBJECTIVE_SCALE * clone.slack_max
        )
        scales = _scale_rows(clone)
        if not hasattr(clone, "rc"):
            clone.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        clone.dual.clear()
        clone.rc.clear()
        results = _solve_with_lp_optimizer(clone, solver, algorithm, _options(solver))
        term = results.solver.termination_condition
        if term == pyo.TerminationCondition.optimal:
            try:
                dual_objective, gap = _certificate(clone)
                slack = _finite(clone.slack_max, "restoration slack")
                intercept = min(slack, dual_objective / RESTORATION_OBJECTIVE_SCALE)
                if intercept <= ZERO_SLACK_TOLERANCE:
                    retry = solve_original(model, solver)
                    if retry is not None:
                        return retry
                    raise ValueError(
                        f"Near-zero Phase I but original LP is infeasible; slack={slack}"
                    )
                lambdas = {}
                for name in active_names:
                    row = clone.BendersLinkingConstraintMap.get(name)
                    if row is None:
                        raise ValueError(f"Missing restoration linking row for {name}")
                    lambdas[name] = (
                        _finite(clone.dual.get(row), f"linking dual for {name}")
                        / scales[row.name]
                        / RESTORATION_OBJECTIVE_SCALE
                    )
                # Scale the whole inequality, not just its duals, so the master
                # cannot discard a meaningful small Phase-I violation.
                cut_scale = max(1.0, 1.0 / intercept)
                cut = {
                    "feasible": False,
                    "op_cost": intercept * cut_scale,
                    "benders_lambda": {
                        name: dual * cut_scale for name, dual in lambdas.items()
                    },
                    "continuous_lambda": {
                        name: dual * cut_scale for name, dual in lambdas.items()
                    },
                    "continuous_point": point,
                    "theta_coeff": 0.0,
                    "restoration_slack": slack,
                    "restoration_cut_scale": cut_scale,
                    "restoration_dual_gap": gap / RESTORATION_OBJECTIVE_SCALE,
                }
                for name, dual in cut["benders_lambda"].items():
                    _finite(dual, f"scaled linking dual for {name}")
                if os.environ.get("GRIDPATH_BENDERS_INFEAS_DIAG"):
                    logging.getLogger(__name__).warning(
                        "Restoration %s: slack=%.12g, dual gap=%.12g, cut scale=%.12g",
                        subproblem_key,
                        slack,
                        cut["restoration_dual_gap"],
                        cut_scale,
                    )
                cutlog = os.environ.get("GRIDPATH_BENDERS_CUTLOG")
                if cutlog:
                    with open(cutlog, "a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "sub": str(
                                        getattr(
                                            subproblem_key, "subproblem", subproblem_key
                                        )
                                    ),
                                    "t": slack,
                                    "lam": cut["benders_lambda"],
                                    "x": point,
                                    "cut_intercept": cut["op_cost"],
                                    "cut_scale": cut_scale,
                                }
                            )
                            + "\n"
                        )
                return cut
            except ValueError as error:
                failures.append(f"{algorithm}: {error}")
        else:
            failures.append(f"{algorithm}: {term}")
        logging.getLogger(__name__).warning(
            "Restoration certificate rejected for %s: %s", subproblem_key, failures[-1]
        )
    dump = os.environ.get("GRIDPATH_BENDERS_RESTORATION_LP")
    if dump:
        root, extension = os.path.splitext(dump)
        sub = str(getattr(subproblem_key, "subproblem", subproblem_key)).replace(
            os.sep, "_"
        )
        clone.write(
            f"{root}_{sub}{extension or '.lp'}",
            io_options={"symbolic_solver_labels": True},
        )
    raise RuntimeError(
        f"No reliable restoration certificate for {subproblem_key}: {failures}"
    )
