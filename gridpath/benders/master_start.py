"""Cheaper, validated starts for the Benders master.

Three independent pieces live here (see section 15.5.18 of the Benders
documentation):

1. Scaled artificial caps (``--benders_bound_strategy scaled_caps``).  A
   continuous planning variable with no upper bound gets a cap sized from the
   bounds of its own variable family instead of the blanket 1e8.  The caps are
   *artificial*: they are recorded in a registry, and before Benders may stop,
   the master optimum is checked.  A cap that is active at the optimum is
   enlarged and the master is solved again.  A cap that is inactive at the
   optimum of a convex master cannot have changed that optimum, so the final
   bound is then valid for the uncapped master.

2. A seeded start (``--benders_seed_plan``).  A plan (every variable at its
   lower bound, or a saved plan from an earlier run) is projected onto the
   master's feasible set with one small LP and evaluated before the first
   master optimum is used.

3. Persisting the incumbent plan as JSON so a later run can seed from it.

The interior-point master option is a solver setting and is applied in
``apply_master_interior_point``.
"""

import json
import logging
import math
import os
import statistics

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

logger = logging.getLogger(__name__)

BOUND_STRATEGIES = ("legacy", "scaled_caps")
DEFAULT_CAP_FACTOR = 10.0
# A cap is never smaller than this, so a family of tiny bounds cannot produce
# a cap that is itself a binding restriction.
MIN_SCALED_CAP = 1.0
# Used only when the model offers no finite bound at all to scale from.
NO_REFERENCE_CAP = 1e6
# Bounds this large are artefacts (for example earlier 1e8 caps), not scales.
MAX_REFERENCE_BOUND = 1e7
ACTIVE_CAP_REL_TOL = 1e-6
MAX_CAP_ENLARGEMENTS = 8

PLAN_FILE_NAME = "benders_incumbent_plan.json"


# ---------------------------------------------------------------------------
# Artificial-cap registry
# ---------------------------------------------------------------------------

def _registry(instance):
    if not hasattr(instance, "_benders_artificial_caps"):
        instance._benders_artificial_caps = {"ub": {}, "lb": {}}
    return instance._benders_artificial_caps


def register_artificial_cap(instance, name, side, cap):
    """Record that ``name`` carries an artificial bound (side 'ub' or 'lb')."""
    _registry(instance)[side][name] = float(cap)


def artificial_cap_count(instance):
    reg = _registry(instance)
    return len(reg["ub"]) + len(reg["lb"])


def family_of(name):
    """Component name without its index: ``Cap[a,2030]`` -> ``Cap``."""
    return name.split("[", 1)[0]


def _day_weights(instance):
    """{(period, subproblem): days represented}; equal weights if unknown."""
    weights = getattr(instance, "_benders_subproblem_day_weight", None)
    if weights:
        return weights
    by_period = getattr(instance, "SUBPROBLEMS_BY_PERIOD", None)
    if by_period is None:
        return {}
    try:
        return {
            (int(period), str(sp)): 1.0
            for period in by_period
            for sp in by_period[period]
        }
    except Exception:
        return {}


def allocation_required_totals(instance):
    """Largest period total each allocation family must carry, {family: total}.

    An allocation variable without a natural bound is capped from a family
    reference; the fallback (1e6) can be far below the budget it has to
    carry (for example (K-1) * rate * Capacity for the performance-standard
    MW allocation), which makes the first master infeasible inside the caps.
    The totals below come from the master's own sum constraints.
    """
    totals = {}

    def note(family, value):
        try:
            value = abs(float(value))
        except (TypeError, ValueError):
            return
        if math.isfinite(value) and value > 0:
            totals[family] = max(totals.get(family, 0.0), value)

    weights = _day_weights(instance)

    def n_sub(period):
        return len([1 for (p, _sp) in weights if p == period])

    expr = getattr(instance, "Benders_Performance_Standard_Capacity_MW", None)
    if expr is not None:
        try:
            from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr

            for (zone, period) in expr:
                rate = pyo.value(
                    instance.performance_standard_tco2_per_mw[zone, period]
                )
                if not math.isfinite(rate):
                    continue
                _lb, ub = compute_bounds_on_expr(expr[zone, period].expr)
                if ub is None or not math.isfinite(ub):
                    continue
                note(
                    "PERFORMANCE_STANDARD_MW_ALLOCATION",
                    max(n_sub(period) - 1, 1) * rate * ub,
                )
        except Exception:
            pass

    for family, param_name in (
        ("PERIOD_ENERGY_TARGET_ALLOCATION", "Period_Energy_Target"),
        ("CARBON_CAP_ALLOCATION", "carbon_cap_target"),
    ):
        param = getattr(instance, param_name, None)
        if param is None:
            continue
        try:
            for index in param:
                note(family, pyo.value(param[index]))
        except Exception:
            pass
    requirement = getattr(instance, "benders_generic_policy_requirement", None)
    if requirement is not None:
        try:
            for index in requirement:
                note("GENERIC_POLICY_ALLOCATION", pyo.value(requirement[index]))
        except Exception:
            pass
    return totals


def scaled_cap_references(instance, variables):
    """Return ({family: reference}, global_reference).

    The reference of a family is the largest finite upper bound among its
    variables that carry an explicit (non-artificial) bound.  The global
    reference is the median of the family references.
    """
    reg = _registry(instance)
    family_ref = {}
    for var in variables:
        if var.name in reg["ub"] or not var.has_ub():
            continue
        ub = pyo.value(var.ub)
        if ub is None or not math.isfinite(ub) or abs(ub) > MAX_REFERENCE_BOUND:
            continue
        fam = family_of(var.name)
        family_ref[fam] = max(family_ref.get(fam, 0.0), abs(float(ub)))
    # The global reference (used for families with no reference of their
    # own) must not be inflated by the allocation budgets.
    positive = [v for v in family_ref.values() if v > 0]
    global_ref = statistics.median(positive) if positive else None
    # Allocation families must be able to carry their period budget.
    for fam, total in allocation_required_totals(instance).items():
        family_ref[fam] = max(family_ref.get(fam, 0.0), total)
    return family_ref, global_ref


def scaled_cap_value(name, family_ref, global_ref, factor):
    """Cap for one variable: factor x its family reference (else global)."""
    ref = family_ref.get(family_of(name))
    if ref is None or ref <= 0:
        ref = global_ref
    if ref is None or ref <= 0:
        return NO_REFERENCE_CAP
    return max(MIN_SCALED_CAP, float(factor) * ref)


def _is_auxiliary(name):
    """Recourse-cost and stabilisation variables are never capped."""
    fam = family_of(name)
    return fam.startswith("vTHETA") or fam.startswith("Benders")


def apply_scaled_caps(instance, variables, factor, reference_variables=None):
    """Cap continuous variables that have no upper bound (and no finite lower
    bound for variables the domain leaves free below).

    Returns the number of bounds set.  Every bound set is registered.
    """
    variables = [
        v for v in variables
        if not v.is_binary() and not v.is_integer() and not v.fixed
        and not _is_auxiliary(v.name)
    ]
    # References come from the bounded variables of the whole model, not only
    # from the ones being capped.
    family_ref, global_ref = scaled_cap_references(
        instance,
        variables if reference_variables is None else reference_variables,
    )
    n_set = 0
    for var in variables:
        if not var.has_ub():
            cap = scaled_cap_value(var.name, family_ref, global_ref, factor)
            var.setub(cap)
            register_artificial_cap(instance, var.name, "ub", cap)
            n_set += 1
        if _has_no_finite_lower_bound(var):
            cap = scaled_cap_value(var.name, family_ref, global_ref, factor)
            var.setlb(-cap)
            register_artificial_cap(instance, var.name, "lb", -cap)
            n_set += 1
    return n_set


def _has_no_finite_lower_bound(var):
    if var.has_lb():
        return False
    try:
        domain_lb, _ = var.domain.bounds()
    except (AttributeError, TypeError):
        domain_lb = None
    if domain_lb is None:
        return True
    return not math.isfinite(float(pyo.value(domain_lb)))


def active_artificial_caps(instance, values):
    """Artificial bounds that the solution ``values`` sits on.

    Returns {name: ('ub'|'lb', cap)}.  Only a solution of the master itself
    (not a stabilised trial point) answers the question "is the cap binding".
    """
    reg = _registry(instance)
    active = {}
    for name, cap in reg["ub"].items():
        val = values.get(name)
        if val is not None and val >= cap - ACTIVE_CAP_REL_TOL * max(1.0, abs(cap)):
            active[name] = ("ub", cap)
    for name, cap in reg["lb"].items():
        val = values.get(name)
        if val is not None and val <= cap + ACTIVE_CAP_REL_TOL * max(1.0, abs(cap)):
            active[name] = ("lb", cap)
    return active


def enlarge_artificial_caps(instance, active, factor):
    """Multiply the active caps by ``factor``; return {name: new cap}."""
    reg = _registry(instance)
    new_caps = {}
    for name, (side, cap) in active.items():
        var = instance.find_component(name)
        if var is None:
            continue
        new_cap = cap * float(factor)
        if side == "ub":
            var.setub(new_cap)
        else:
            var.setlb(new_cap)
        reg[side][name] = new_cap
        new_caps[name] = new_cap
    return new_caps


def validate_artificial_caps(instance, solve, solution, master_obj, factor,
                             rel_tol, max_rounds=MAX_CAP_ENLARGEMENTS):
    """Check that the artificial caps did not shape the master optimum.

    ``solve()`` re-solves the master and returns (solution, master_obj).

    The master is a concave maximisation over a convex set.  If no cap is
    active at its optimum, that optimum is optimal without the caps.  If some
    are active, enlarge them and re-solve: if the optimal value does not rise,
    the caps were not binding (an at-bound value with no benefit, as for a
    variable the objective ignores); if it rises, repeat from the new optimum.
    In both cases the argument is exact, not heuristic: a better point outside
    the enlarged box would, by concavity along the segment from the optimum,
    imply a better point inside it.

    Returns (solution, master_obj, rounds, changed, validated).
    """
    rounds = 0
    changed = False
    while True:
        active = active_artificial_caps(instance, solution.values)
        if not active:
            return solution, master_obj, rounds, changed, True
        if rounds >= max_rounds:
            return solution, master_obj, rounds, changed, False
        enlarge_artificial_caps(instance, active, factor)
        rounds += 1
        new_solution, new_obj = solve()
        if new_obj <= master_obj + rel_tol * max(1.0, abs(master_obj)):
            return new_solution, new_obj, rounds, changed, True
        changed = True
        solution, master_obj = new_solution, new_obj


# ---------------------------------------------------------------------------
# Interior-point master (an option, independent of the two start devices)
# ---------------------------------------------------------------------------

def apply_master_interior_point(optimiser, solver_name):
    """Make master solves use the interior-point method with no crossover.

    The master sees an interior point of its optimal face rather than a
    vertex, which makes consecutive trial points move less.  Cuts are still
    valid because they come from the subproblems, not from the master basis.
    """
    name = str(solver_name).lower()
    if "mosek" in name:
        optimiser.options["MSK_IPAR_OPTIMIZER"] = "MSK_OPTIMIZER_INTPNT"
        optimiser.options["MSK_IPAR_INTPNT_BASIS"] = "MSK_BI_NEVER"
        # Without crossover there is no basic solution: tell the Pyomo
        # adapter to read the interior solution (see mosek_benders.py).
        optimiser._benders_interior_point = True
        return True
    if "gurobi" in name:
        optimiser.options["Method"] = 2
        optimiser.options["Crossover"] = 0
        return True
    return False


# ---------------------------------------------------------------------------
# Seed plan
# ---------------------------------------------------------------------------

def plan_file_path(scenario_directory):
    return os.path.join(scenario_directory, PLAN_FILE_NAME)


def save_plan(path, values, objective=None):
    """Write the plan atomically; a failure only warns."""
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as stream:
            json.dump(
                {"version": 1, "objective": objective, "values": values},
                stream,
            )
        os.replace(tmp, path)
        return True
    except Exception as exc:
        logger.warning("Could not save the incumbent plan to %s (%r).", path, exc)
        return False


def load_plan(path):
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    values = payload.get("values", payload)
    return {str(k): float(v) for k, v in values.items()}


from gridpath.benders.allocation_names import (  # noqa: E402
    ALLOCATION_COMPONENTS,
    is_allocation_name as _is_allocation_name,
)


def allocation_share_groups(instance):
    """Return [(variable names, shares)] for every allocation family/key.

    Each group is the set of per-day allocations that share one period
    budget; the share of a day is its represented number of days over the
    period total (an equal split when every day weighs the same).
    """
    weights = _day_weights(instance)
    if not weights:
        return []
    groups = []
    for component in ALLOCATION_COMPONENTS:
        var = getattr(instance, component, None)
        if var is None or not var.is_indexed():
            continue
        by_key = {}
        for index, data in var.items():
            if not isinstance(index, tuple) or len(index) < 2:
                continue
            period = None
            for element in index[:-1]:
                try:
                    if (int(element), str(index[-1])) in weights:
                        period = int(element)
                        break
                except (TypeError, ValueError):
                    continue
            if period is None or data.fixed:
                continue
            by_key.setdefault(index[:-1], []).append(
                (data.name, weights[(period, str(index[-1]))])
            )
        for members in by_key.values():
            total = sum(w for _, w in members)
            if len(members) > 1 and total > 0:
                groups.append(
                    ([n for n, _ in members], [w / total for _, w in members])
                )
    return groups


def resolve_seed_targets(instance, seed_setting, scenario_directory,
                         continuous_names):
    """Return ({name: target value}, description) or ({}, reason).

    ``seed_setting``: 'none', 'lower_bounds', 'auto' or a path to a plan file.
    """
    setting = str(seed_setting or "none").strip()
    key = setting.lower()
    if key in ("", "none"):
        return {}, "disabled"

    def lower_targets():
        targets = {}
        for name in continuous_names:
            if _is_allocation_name(name):
                continue
            var = instance.find_component(name)
            if var is None or var.fixed:
                continue
            lb = pyo.value(var.lb) if var.has_lb() else None
            if lb is None or not math.isfinite(lb) or name in _registry(instance)["lb"]:
                lb = 0.0
            targets[name] = float(lb)
        return targets

    if key == "lower_bounds":
        return lower_targets(), "every continuous variable at its lower bound"
    if key == "auto":
        path = plan_file_path(scenario_directory)
        if os.path.isfile(path):
            return _file_targets(instance, path, continuous_names)
        return lower_targets(), (
            "no saved plan at %s; every continuous variable at its lower "
            "bound" % path
        )
    return _file_targets(instance, setting, continuous_names)


def _file_targets(instance, path, continuous_names):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "--benders_seed_plan file not found: %s" % path
        )
    saved = load_plan(path)
    wanted = set(continuous_names)
    targets = {n: v for n, v in saved.items() if n in wanted}
    return targets, "plan file %s (%d of %d saved values match the master)" % (
        path, len(targets), len(saved),
    )


def project_onto_master(instance, optimiser, targets, share_groups=None,
                        share_weight=1e-6):
    """Return the feasible master point closest (weighted L1) to ``targets``.

    One LP over the master's constraints and bounds (cuts included, so a
    re-projection later stays consistent).  The objective is swapped for the
    distance and restored afterwards.  Returns ({name: value}, status) where
    the dict is empty if the LP did not solve to optimality.
    """
    if not targets:
        return {}, "no targets"
    # Groups already described by the plan keep the plan's values; the rest
    # are pulled toward their day-weighted share of the (feasible) budget.
    share_groups = [
        (names, shares) for names, shares in (share_groups or [])
        if not any(n in targets for n in names)
    ]

    objective = instance.MasterObjective
    saved_expr = objective.expr
    saved_sense = objective.sense

    names = list(targets)
    block = pyo.Block(concrete=True)
    instance._benders_seed_projection = block
    try:
        block.idx = pyo.RangeSet(0, len(names) - 1)
        block.pos = pyo.Var(block.idx, domain=pyo.NonNegativeReals)
        block.neg = pyo.Var(block.idx, domain=pyo.NonNegativeReals)
        block.link = pyo.ConstraintList()
        terms = []
        for i, name in enumerate(names):
            var = instance.find_component(name)
            target = float(targets[name])
            block.link.add(var - target == block.pos[i] - block.neg[i])
            weight = 1.0 / max(1.0, abs(target))
            terms.append(weight * (block.pos[i] + block.neg[i]))
        block.share_link = pyo.ConstraintList()
        block.share_pos = pyo.VarList(domain=pyo.NonNegativeReals)
        block.share_neg = pyo.VarList(domain=pyo.NonNegativeReals)
        for group_names, shares in share_groups:
            variables = [instance.find_component(n) for n in group_names]
            total = sum(variables)
            for var, share in zip(variables, shares):
                pos, neg = block.share_pos.add(), block.share_neg.add()
                block.share_link.add(var - share * total == pos - neg)
                terms.append(share_weight * (pos + neg))
        objective.set_value(sum(terms))
        objective.sense = pyo.minimize

        results = optimiser.solve(instance, tee=False)
        term = results.solver.termination_condition
        if term != TerminationCondition.optimal:
            return {}, "projection LP ended with %s" % term
        projected = {}
        moved = 0.0
        for name in names:
            val = pyo.value(instance.find_component(name), exception=False)
            if val is not None:
                projected[name] = float(val)
                moved += abs(val - targets[name]) / max(1.0, abs(targets[name]))
        return projected, "moved %.6g (weighted L1)" % moved
    finally:
        objective.set_value(saved_expr)
        objective.sense = saved_sense
        instance.del_component("_benders_seed_projection")
