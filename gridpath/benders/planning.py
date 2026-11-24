import logging
import os
from dataclasses import dataclass
# from pyomo.environ import Expression
# instance.ePlanningCost = Expression(expr=expr)

from pyomo.environ import (
    Var,
    ConstraintList,
    Objective,
    value,
    maximize,
    Reals,
    SolverFactory,
)
from pyomo.opt import SolverStatus, TerminationCondition

from gridpath.benders.master_start import (
    MAX_CAP_ENLARGEMENTS,
    apply_scaled_caps,
    enlarge_artificial_caps,
    register_artificial_cap,
)

logger = logging.getLogger(__name__)
BENDERS_THETA_SCALE = 1e9


# -------------------------------------------------------------------------
# Dataclass for the planning solution
# -------------------------------------------------------------------------

@dataclass
class PlanningSolution:
    planning_cost: float
    values: dict[str, float]


# -------------------------------------------------------------------------
# Solve the master/planning problem
# -------------------------------------------------------------------------

BENDERS_MASTER_VARIABLE_CAP = 1e8


def solve_planning_problem(
    instance,
    planning_variables,
    optimiser,
    required_variables=None,
):
    """
    Solve the planning/master problem.

    Parameters
    ----------
    instance : ConcreteModel
        Instantiated Pyomo master model.
    planning_variables : list[str]
        Names of linking/planning variables.
    required_variables : list[str], optional
        Names of planning variables that must be included in the solution.
    optimiser : SolverFactory
        Solver selected by the Benders algorithm.

    Returns
    -------
    planning_sol : PlanningSolution
    UB : float
        Current master objective value.
    results : SolverResults
    """

    results = optimiser.solve(instance, tee=False)

    status = results.solver.status
    termination = results.solver.termination_condition

    # A fresh stock solver cannot read the interior solution that the
    # interior-point master relies on (no basic solution without
    # crossover), and re-solving an infeasible model cannot change the
    # answer, so only retry other failures with the configured solver.
    _retry_with_fresh_solver = not (
        getattr(optimiser, "_benders_interior_point", False)
        or termination == TerminationCondition.infeasible
    )
    if _retry_with_fresh_solver and (
        status != SolverStatus.ok or termination != TerminationCondition.optimal
    ):
        solver_name = getattr(optimiser, "name", None)

        if solver_name and solver_name != "gams":
            fresh_optimiser = SolverFactory(solver_name)

            if fresh_optimiser.available(exception_flag=False):
                fresh_optimiser.options.update(optimiser.options)
                results = fresh_optimiser.solve(instance, tee=False)
                status = results.solver.status
                termination = results.solver.termination_condition

    if termination in (
        TerminationCondition.unbounded,
        TerminationCondition.infeasibleOrUnbounded,
    ):
        # Cuts may not yet bound every planning variable (e.g. builds that
        # only appear in cuts). Cap unbounded continuous variables at a
        # value far above any physical one and re-solve.
        capped = 0
        names = set(planning_variables)
        strategy = getattr(instance, "_benders_cap_strategy", "legacy")
        factor = getattr(instance, "_benders_cap_factor", 10.0)
        to_cap = [
            var
            for var in instance.component_data_objects(Var, active=True)
            if (
                var.name in names
                and not var.is_integer()
                and not var.is_binary()
                and not var.has_ub()
            )
        ]
        if strategy == "scaled_caps":
            capped = apply_scaled_caps(
                instance,
                to_cap,
                factor,
                reference_variables=list(
                    instance.component_data_objects(Var, active=True)
                ),
            )
        else:
            for var in to_cap:
                var.setub(BENDERS_MASTER_VARIABLE_CAP)
                register_artificial_cap(
                    instance, var.name, "ub", BENDERS_MASTER_VARIABLE_CAP
                )
                capped += 1
        if capped:
            logger.warning(
                "Planning problem unbounded; capped %d variable bound(s) "
                "(strategy %s; legacy cap %g).",
                capped,
                strategy,
                BENDERS_MASTER_VARIABLE_CAP,
            )
            results = optimiser.solve(instance, tee=False)
            status = results.solver.status
            termination = results.solver.termination_condition
    if (
        termination == TerminationCondition.infeasible
        and getattr(instance, "_benders_cap_strategy", "legacy")
        == "scaled_caps"
    ):
        # A scaled cap may be too small for a requirement of the master
        # (for example a reserve margin). Enlarge every artificial bound and
        # retry before declaring the master infeasible.
        factor = getattr(instance, "_benders_cap_factor", 10.0)
        for attempt in range(MAX_CAP_ENLARGEMENTS):
            reg = getattr(instance, "_benders_artificial_caps", None)
            if not reg or not (reg["ub"] or reg["lb"]):
                break
            everything = {
                **{n: ("ub", c) for n, c in reg["ub"].items()},
                **{n: ("lb", c) for n, c in reg["lb"].items()},
            }
            enlarge_artificial_caps(instance, everything, factor)
            logger.warning(
                "Planning problem infeasible inside the artificial caps; "
                "enlarged %d bound(s) x%g (attempt %d).",
                len(everything), factor, attempt + 1,
            )
            results = optimiser.solve(instance, tee=False)
            status = results.solver.status
            termination = results.solver.termination_condition
            if termination == TerminationCondition.optimal:
                break
    if status != SolverStatus.ok or termination != TerminationCondition.optimal:
        failed_master_path = os.environ.get(
            "GRIDPATH_BENDERS_FAILURE_LP"
        )
        if failed_master_path:
            instance.write(
                failed_master_path,
                io_options={"symbolic_solver_labels": True},
            )
        raise RuntimeError(
            f"Planning problem failed.\n"
            f"Status: {status}\n"
            f"Termination: {termination}"
        )

    planning_sol = process_planning_sol(
        instance,
        planning_variables,
        required_variables=required_variables,
    )

    # Evaluate the objective directly from the solved model.
    UB = value(instance.MasterObjective)

    return planning_sol, UB, results


# -------------------------------------------------------------------------
# Add theta variables and modify the objective
# -------------------------------------------------------------------------

def add_approximate_variable_cost(instance, benders_subproblems):
    """
    Add theta variables and modify the objective to

        ePlanningNPV - BENDERS_THETA_SCALE * sum(theta_w)

    where theta_w represents net operational cost in billions of dollars.
    """

    if hasattr(instance, "vTHETA"):
        logger.warning("vTHETA already exists. Skipping.")
        return

    instance.vTHETA = Var(
        benders_subproblems,
        domain=Reals,
    )

    # -------------------------------------------------------------
    # Benders cut containers
    # -------------------------------------------------------------

    # Standard affine Benders optimality cuts.
    instance.BendersOptimalityCuts = ConstraintList()

    # Benders feasibility cuts.
    instance.BendersFeasibilityCuts = ConstraintList()

    # Discrete/combinatorial Benders optimality cuts.
    instance.BendersDiscreteOptimalityCuts = ConstraintList()

    # -------------------------------------------------------------
    # Theta lower bounds
    # -------------------------------------------------------------

    instance.ThetaLowerBound = ConstraintList()

    if hasattr(instance, "eLowerBoundOperatingCost"):

        for key in benders_subproblems:
            instance.ThetaLowerBound.add(
                instance.vTHETA[key]
                >= instance.eLowerBoundOperatingCost[key]
                / BENDERS_THETA_SCALE
            )

    else:

        for key in benders_subproblems:
            instance.ThetaLowerBound.add(
                instance.vTHETA[key] >= 0.0
            )

    # -------------------------------------------------------------
    # Original objective
    # -------------------------------------------------------------

    if not hasattr(instance, "NPV"):
        raise RuntimeError(
            "Expected the model to contain an NPV objective."
        )

    original_obj = instance.NPV

    instance.ePlanningNPV = original_obj.expr

    original_obj.deactivate()

    # -------------------------------------------------------------
    # Benders master objective
    # -------------------------------------------------------------

    instance.MasterObjective = Objective(
        expr=(
            instance.ePlanningNPV
            - BENDERS_THETA_SCALE * sum(
                instance.vTHETA[key]
                for key in benders_subproblems
            )
        ),
        sense=maximize,
    )


def set_recourse_cost_lower_bounds(instance, recourse_cost_lower_bounds):
    """Set valid lower bounds on net operational cost theta variables."""
    instance.ThetaLowerBound.clear()
    for subproblem_key, lower_bound in recourse_cost_lower_bounds.items():
        instance.ThetaLowerBound.add(
            instance.vTHETA[subproblem_key] >= float(lower_bound)
            / BENDERS_THETA_SCALE
        )


def process_planning_sol(
    instance,
    planning_variables,
    required_variables=None,
):
    """
    Create a PlanningSolution from the current master solution.

    All initialized planning-variable values are retained. Variables
    without values are tolerated only when they are not required by
    any operational subproblem.

    Values are returned without integer/binary rounding.
    """

    if required_variables is None:
        required_variables = set(planning_variables)
    else:
        required_variables = set(required_variables)

    var_lookup = {
        var.name: var
        for var in instance.component_data_objects(
            Var,
            active=True,
        )
    }

    planning_values = {}

    for name in planning_variables:

        if name not in var_lookup:
            if name in required_variables:
                raise KeyError(
                    f"Required linking variable '{name}' was not found "
                    f"in the active master variables."
                )
            continue

        val = value(var_lookup[name], exception=False)

        if val is None:
            if name in required_variables:
                raise ValueError(
                    f"Required linking variable '{name}' has no value "
                    f"after the master solve."
                )

            logger.debug(
                "Master variable '%s' has no solution value and is not "
                "required by any operational subproblem; skipping it.",
                name,
            )
            continue

        planning_values[name] = float(val)

    planning_cost_value = value(
        instance.ePlanningNPV,
        exception=False,
    )

    if planning_cost_value is None:
        raise ValueError(
            "The planning objective ePlanningNPV has no solution value "
            "after the master solve."
        )

    planning_cost = float(planning_cost_value)

    return PlanningSolution(
        planning_cost=planning_cost,
        values=planning_values,
    )