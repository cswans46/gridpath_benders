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
from pyomo.repn.standard_repn import generate_standard_repn

logger = logging.getLogger(__name__)

# Bump when the meaning of a stored bound changes; old files are then ignored.
BOUNDS_CACHE_VERSION = 1

# An FBBT upper bound at or above this is treated as "no useful bound": it is
# an artefact of big-M style chains and would only weaken the recourse LP.
FBBT_MAX_USEFUL_BOUND = 1e10


# ---------------------------------------------------------------------------
# FBBT
# ---------------------------------------------------------------------------


def propagate_upper_bounds(
    model, target_vars, max_iter=10, feasibility_tol=1e-8
):
    """
    Return ``{var.name: upper_bound}`` for the variables in ``target_vars``
    that FBBT can bound by a finite, useful value.

    ``pyomo.contrib.fbbt.fbbt.fbbt`` tightens variable bounds in place on the
    whole model.  Every variable's bounds are therefore recorded first and
    restored afterwards, so the master is left exactly as it was; only the
    bounds for ``target_vars`` are returned.  An upper bound derived this way
    is implied by the master's constraints, so it is valid at every feasible
    master point (cuts only shrink the feasible set).

    Any failure (an exception raised by FBBT, an unsupported Pyomo version)
    returns an empty dict; the caller then falls back to OBBT.
    """
    target_vars = list(target_vars)
    if not target_vars:
        return {}

    try:
        from pyomo.contrib.fbbt.fbbt import fbbt
    except Exception as exc:  # pragma: no cover - depends on the install
        logger.warning("FBBT is not available (%r); skipping.", exc)
        return {}

    all_vars = list(model.component_data_objects(pyo.Var, descend_into=True))
    if all_vars and not hasattr(all_vars[0], "_lb"):
        logger.warning(
            "This Pyomo version does not expose Var._lb/_ub, so the master's "
            "bounds cannot be restored after FBBT; skipping FBBT."
        )
        return {}

    snapshot = [(v, v._lb, v._ub) for v in all_vars]
    started = time.time()
    found = {}

    try:
        new_bounds = fbbt(
            model,
            max_iter=max_iter,
            feasibility_tol=feasibility_tol,
        )
        for var in target_vars:
            bounds = new_bounds.get(var)
            if bounds is None:
                continue
            lb, ub = bounds
            if ub is None or not math.isfinite(ub):
                continue
            if abs(ub) >= FBBT_MAX_USEFUL_BOUND:
                continue
            if lb is not None and math.isfinite(lb) and ub < lb - 1e-9:
                continue
            found[var.name] = float(ub)
    except Exception as exc:
        logger.warning(
            "FBBT failed (%r); the master is unchanged and OBBT will be used "
            "for the remaining variables.",
            exc,
        )
        found = {}
    finally:
        for var, lb, ub in snapshot:
            var.setlb(lb)
            var.setub(ub)

    logger.info(
        "FBBT bounded %d of %d variable(s) in %.1f s.",
        len(found),
        len(target_vars),
        time.time() - started,
    )
    return found


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
