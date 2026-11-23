"""
Cheap, reusable bounds for the continuous linking variables of the Benders
master.

Why this module exists
----------------------
The recourse (operational) subproblems need a finite upper bound on every
continuous linking variable, and a master without cuts is only bounded if
every such variable has one.  Bounds are obtained, cheapest first, from

1. the variable's explicit Pyomo bound,
2. GridPath's capacity / potential constraints,
3. feasibility-based bound tightening (FBBT, this module): interval
   arithmetic through the master constraints, one pass, no solver calls,
4. optimization-based bound tightening (OBBT): one master LP per variable,
5. an assumed cap (``BENDERS_MASTER_VARIABLE_CAP``) enforced in the master.

Bounds depend only on the master's variables and constraints (not on its
objective and not on the iteration), so they are stored on disk under a
fingerprint of exactly those and reused by later runs and restarts.
"""

import hashlib
import json
import logging
import math
import os
import tempfile
import time

import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables
from pyomo.repn.standard_repn import generate_standard_repn

logger = logging.getLogger(__name__)

# Bump when the meaning of a stored bound changes; old files are then ignored.
BOUNDS_CACHE_VERSION = 2

# An FBBT upper bound at or above this is treated as "no useful bound": it is
# an artefact of big-M style chains and would only weaken the recourse LP.
FBBT_MAX_USEFUL_BOUND = 1e10


# ---------------------------------------------------------------------------
# FBBT
# ---------------------------------------------------------------------------


# FBBT compares propagated bounds of numeric constants with the constants
# themselves using an ABSOLUTE tolerance. With coefficients near 1e9 the
# floating-point rounding alone exceeds 1e-8, and FBBT then raises "Detected
# an infeasible constraint" for a perfectly feasible master. A larger
# tolerance is more conservative (it never removes feasible points), so the
# attempts below go from strict to loose.
FBBT_TOLERANCE_LADDER = (1e-8, 1e-6, 1e-4, 1e-2)
# Constraint-by-constraint sweeps (see below) use this tolerance.
FBBT_SWEEP_TOLERANCE = 1e-2
FBBT_SWEEPS = 3


def _useful_upper_bound(lb, ub):
    if ub is None or not math.isfinite(ub):
        return None
    if abs(ub) >= FBBT_MAX_USEFUL_BOUND:
        return None
    if lb is not None and math.isfinite(lb) and ub < lb - 1e-9:
        return None
    return float(ub)


def propagate_upper_bounds(
    model, target_vars, max_iter=10, max_seconds=None
):
    """
    Return ``{var.name: upper_bound}`` for the variables in ``target_vars``
    that FBBT can bound by a finite, useful value.

    ``pyomo.contrib.fbbt.fbbt.fbbt`` tightens variable bounds in place on the
    whole model.  Every variable's bounds are therefore recorded first and
    restored after each attempt, so the master is left exactly as it was;
    only the bounds for ``target_vars`` are returned.  An upper bound derived
    this way is implied by the master's constraints, so it is valid at every
    feasible master point (cuts only shrink the feasible set).

    Strategy:
    1. One whole-model pass at each tolerance in ``FBBT_TOLERANCE_LADDER``
       until one succeeds (see the comment on the ladder).
    2. If every whole-model pass raises, propagate constraint by constraint
       (``FBBT_SWEEPS`` sweeps, tolerance ``FBBT_SWEEP_TOLERANCE``), skipping
       and naming each constraint FBBT calls infeasible. One numerically
       awkward constraint then costs only its own propagation.

    Any other failure returns an empty dict; the caller then falls back to
    OBBT.
    """
    target_vars = list(target_vars)
    if not target_vars:
        return {}

    try:
        from pyomo.contrib.fbbt.fbbt import fbbt
        from pyomo.common.errors import InfeasibleConstraintException
    except Exception as exc:  # pragma: no cover - depends on the install
        logger.warning("FBBT is not available (%r); skipping.", exc)
        return {}

    if max_seconds is None:
        max_seconds = float(os.getenv("GRIDPATH_BENDERS_FBBT_MAX_SECONDS", "900"))

    all_vars = list(model.component_data_objects(pyo.Var, descend_into=True))
    if all_vars and not hasattr(all_vars[0], "_lb"):
        logger.warning(
            "This Pyomo version does not expose Var._lb/_ub, so the master's "
            "bounds cannot be restored after FBBT; skipping FBBT."
        )
        return {}

    snapshot = [(v, v._lb, v._ub) for v in all_vars]

    def restore_bounds():
        for var, lb, ub in snapshot:
            var.setlb(lb)
            var.setub(ub)

    started = time.time()
    found = {}
    mode = None

    # ---- 1. whole-model passes -------------------------------------------
    for tolerance in FBBT_TOLERANCE_LADDER:
        try:
            new_bounds = fbbt(model, max_iter=max_iter, feasibility_tol=tolerance)
            for var in target_vars:
                bounds = new_bounds.get(var)
                if bounds is None:
                    continue
                ub = _useful_upper_bound(*bounds)
                if ub is not None:
                    found[var.name] = ub
            mode = "whole-model pass, tolerance %g" % tolerance
            restore_bounds()
            break
        except InfeasibleConstraintException as exc:
            restore_bounds()
            logger.info(
                "FBBT whole-model pass at tolerance %g raised %r.", tolerance, exc
            )
        except Exception as exc:
            restore_bounds()
            logger.warning("FBBT failed (%r); using OBBT instead.", exc)
            return {}

    # ---- 2. constraint-by-constraint fallback ----------------------------
    if mode is None:
        skipped = []
        constraints = list(
            model.component_data_objects(
                pyo.Constraint, active=True, descend_into=True
            )
        )
        try:
            for sweep in range(FBBT_SWEEPS):
                for con in constraints:
                    if time.time() - started > max_seconds:
                        raise TimeoutError("FBBT time budget used up")
                    try:
                        fbbt(
                            con,
                            max_iter=max_iter,
                            feasibility_tol=FBBT_SWEEP_TOLERANCE,
                        )
                    except InfeasibleConstraintException:
                        if sweep == 0:
                            skipped.append(con.name)
            for var in target_vars:
                ub = _useful_upper_bound(var.lb, var.ub)
                if ub is not None:
                    found[var.name] = ub
            mode = "constraint-by-constraint, %d skipped" % len(skipped)
            if skipped:
                logger.warning(
                    "FBBT regards %d master constraint(s) as infeasible at "
                    "tolerance %g and skipped them; first ones: %s. This is "
                    "usually floating-point error with very large "
                    "coefficients, but check them if the master itself is "
                    "infeasible.",
                    len(skipped),
                    FBBT_SWEEP_TOLERANCE,
                    skipped[:5],
                )
        except Exception as exc:
            logger.warning("FBBT failed (%r); using OBBT instead.", exc)
            found = {}
        finally:
            restore_bounds()

    logger.info(
        "FBBT (%s) bounded %d of %d variable(s) in %.1f s.",
        mode,
        len(found),
        len(target_vars),
        time.time() - started,
    )
    return found


def find_free_recession_variables(model, candidate_vars):
    """
    Names of candidate variables whose value can grow without limit at any
    feasible master point, found without a solver call.

    If every constraint that contains ``x`` stays satisfied when only ``x``
    increases (a row with a finite upper side needs coefficient <= 0, a row
    with a finite lower side needs coefficient >= 0, an equality row must not
    contain ``x``), then ``x + t`` is feasible for every t >= 0 whenever ``x``
    is. Its upper bound is infinite, which is exactly what OBBT would find
    after one full master solve per variable ("Problem is dual infeasible").
    Variables in a non-linear or unparseable constraint are never reported.
    Only variables without an explicit upper bound should be passed in.
    """
    started = time.time()
    candidates = {id(v): v for v in candidate_vars if not v.fixed}
    blocked = set()

    for c in model.component_data_objects(
        pyo.Constraint, active=True, descend_into=True
    ):
        has_lower = c.has_lb()
        has_upper = c.has_ub()
        try:
            repn = generate_standard_repn(
                c.body, compute_values=True, quadratic=False
            )
            if not repn.is_linear():
                raise ValueError("non-linear")
            for v, coef in zip(repn.linear_vars, repn.linear_coefs):
                if id(v) not in candidates:
                    continue
                coef = float(coef)
                if coef == 0.0:
                    continue
                if (has_upper and coef > 0) or (has_lower and coef < 0):
                    blocked.add(id(v))
        except Exception:
            for v in identify_variables(c.body, include_fixed=False):
                blocked.add(id(v))

    free = {v.name for key, v in candidates.items() if key not in blocked}
    logger.info(
        "%d of %d unbounded variable(s) can grow without limit (no OBBT "
        "needed); %.1f s.",
        len(free),
        len(candidates),
        time.time() - started,
    )
    return free


# ---------------------------------------------------------------------------
# Fingerprint and on-disk cache
# ---------------------------------------------------------------------------


def _num(value):
    try:
        return repr(float(value))
    except (TypeError, ValueError):
        return repr(value)


def bounds_fingerprint(model, extras):
    """
    SHA-256 hex digest of everything the bounds depend on: every variable
    (name, domain, bounds, fixed value) and every active constraint (name,
    sides, linear coefficients), plus the settings in ``extras``.  The
    objective is excluded on purpose (OBBT replaces it), so changing costs
    does not invalidate the bounds.  Order of components does not matter.
    """
    started = time.time()
    outer = hashlib.sha256()
    outer.update(
        json.dumps(
            {"version": BOUNDS_CACHE_VERSION, "extras": extras},
            sort_keys=True,
            default=str,
        ).encode()
    )

    var_lines = []
    for v in model.component_data_objects(pyo.Var, descend_into=True):
        try:
            lb, ub = v.lb, v.ub
        except Exception:
            lb, ub = None, None
        var_lines.append(
            "|".join(
                (
                    v.name,
                    str(getattr(v.domain, "name", v.domain)),
                    _num(lb) if lb is not None else "-",
                    _num(ub) if ub is not None else "-",
                    _num(v.value) if v.fixed else "-",
                )
            )
        )
    var_lines.sort()
    for line in var_lines:
        outer.update(line.encode())
        outer.update(b"\n")

    con_digests = []
    n_cons = 0
    for c in model.component_data_objects(
        pyo.Constraint, active=True, descend_into=True
    ):
        n_cons += 1
        try:
            repn = generate_standard_repn(
                c.body, compute_values=True, quadratic=False
            )
            if repn.is_linear():
                terms = sorted(
                    (v.name, _num(coef))
                    for v, coef in zip(repn.linear_vars, repn.linear_coefs)
                )
                body = repr(terms) + _num(repn.constant)
            else:
                body = str(c.body)
        except Exception:
            body = str(c.body)
        lower = _num(pyo.value(c.lower)) if c.has_lb() else "-"
        upper = _num(pyo.value(c.upper)) if c.has_ub() else "-"
        con_digests.append(
            hashlib.sha1(
                "|".join((c.name, lower, upper, body)).encode()
            ).digest()
        )
    con_digests.sort()
    for digest in con_digests:
        outer.update(digest)

    logger.info(
        "Master fingerprint for the bounds cache computed over %d variables "
        "and %d constraints in %.1f s.",
        len(var_lines),
        n_cons,
        time.time() - started,
    )
    return outer.hexdigest()


def bounds_cache_path(cache_dir, fingerprint):
    return os.path.join(cache_dir, "linking_bounds_%s.json" % fingerprint[:32])


def load_bounds_cache(path, fingerprint):
    """Return the stored payload, or None if absent, unreadable or stale."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable bounds cache %s (%r).", path, exc)
        return None
    if (
        payload.get("version") != BOUNDS_CACHE_VERSION
        or payload.get("fingerprint") != fingerprint
    ):
        return None
    return payload


def save_bounds_cache(path, fingerprint, bounds, assumed, skipped, extras):
    """Write the cache atomically (a crash never leaves a half-written file)."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    payload = {
        "version": BOUNDS_CACHE_VERSION,
        "fingerprint": fingerprint,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "extras": extras,
        "bounds": {name: [lb, ub] for name, (lb, ub) in bounds.items()},
        "assumed": sorted(assumed),
        "skipped": sorted(skipped),
    }
    fd, tmp_path = tempfile.mkstemp(
        prefix="linking_bounds_", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
