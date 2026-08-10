import logging
from dataclasses import dataclass
# from pyomo.environ import Expression
# instance.ePlanningCost = Expression(expr=expr)

from pyomo.environ import (
    Var,
    ConstraintList,
    Objective,
    ObjectiveList,
    value,
    maximize,
    SolverFactory,
)
from pyomo.opt import SolverStatus, TerminationCondition

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Dataclass for the planning solution
# -------------------------------------------------------------------------

@dataclass
class PlanningSolution:
    planning_cost: float
    values: dict


# -------------------------------------------------------------------------
# Solve the master/planning problem
# -------------------------------------------------------------------------

def solve_planning_problem(instance, planning_variables, optimiser):
    """
    Solve the planning (master) problem.

    Parameters
    ----------
    instance : ConcreteModel
        Instantiated Pyomo model.
    planning_variables : list[str]
        Names of the linking variables.
    optimiser : SolverFactory solver predetermined in Benders

    Returns
    -------
    PlanningSolution
    float
        (planning solution, lower bound)
    """

    results = optimiser.solve(instance, tee=False)

    status = results.solver.status
    termination = results.solver.termination_condition

    if (
        status == SolverStatus.ok
        and termination
        in (
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
        )
    ):

        planning_sol = process_planning_sol(
            instance,
            planning_variables,
        )

        LB = value(instance.ePlanningNPV)

        return planning_sol, LB

    raise RuntimeError(
        f"Planning problem failed.\n"
        f"Status: {status}\n"
        f"Termination: {termination}"
    )


# -------------------------------------------------------------------------
# Add theta variables and modify the objective
# -------------------------------------------------------------------------

def add_approximate_variable_cost(instance, n_subproblems):
    """
    Adds theta variables and replaces the objective by

        planning_cost + sum(theta)

    exactly as done in the Julia implementation.
    """

    if hasattr(instance, "vTHETA"):
        logger.warning(
            "vTHETA already exists. Skipping."
        )
        return

    #
    # Theta variables
    #

    instance.vTHETA = Var(
        range(1, n_subproblems + 1)
    )

    #
    # Lower bounds
    #

    instance.BendersCuts = ConstraintList()

    if hasattr(instance, "eLowerBoundOperatingCost"):

        instance.ThetaLowerBound = ConstraintList()

        for w in range(1, n_subproblems + 1):

            instance.ThetaLowerBound.add(
                instance.vTHETA[w]
                >= instance.eLowerBoundOperatingCost[w]
            )

    else:

        instance.ThetaLowerBound = ConstraintList()

        for w in range(1, n_subproblems + 1):

            instance.ThetaLowerBound.add(
                instance.vTHETA[w] >= 0.0
            )

    #
    # Save original objective
    #

    if not hasattr(instance, "NPV"):
        raise RuntimeError(
            "Expected the model to contain an NPV objective."
        )

    original_obj = instance.NPV
    #
    # Store planning cost expression
    #

    instance.ePlanningNPV = original_obj.expr

    #
    # Replace objective
    #

    original_obj.deactivate()

    instance.MasterObjective = Objective(
        expr=instance.ePlanningNPV
        # Because this is a NPV maximising problem, not a cost minimisation problem
        # Need to subtract approximate variable cost from subproblems.
        - sum(
            instance.vTHETA[w]
            for w in range(1, n_subproblems + 1)
        ),
        sense=maximize
    )


# -------------------------------------------------------------------------
# Extract solution
# -------------------------------------------------------------------------

def process_planning_sol(instance, planning_variables):
    """
    Create the PlanningSolution object.
    """

    planning_values = {}

    #
    # Capacity variables (if available)
    #

    negative_capacity = False

    if hasattr(instance, "eAvailableCapacity"):

        capacity_values = [
            value(v)
            for v in instance.eAvailableCapacity.values()
        ]

        negative_capacity = any(
            x < -1e-8
            for x in capacity_values
        )

    #
    # Build lookup dictionary
    #

    var_lookup = {
        var.name: var
        for var in instance.component_data_objects(
            Var,
            active=True,
        )
    }

    #
    # Extract values
    #

    for name in planning_variables:

        val = value(var_lookup[name])

        if negative_capacity:
            val = round_small_values(val)

        planning_values[name] = val

    #
    # Planning cost
    #

    planning_cost = value(instance.ePlanningNPV)

    return PlanningSolution(
        planning_cost=planning_cost,
        values=planning_values,
    )


# -------------------------------------------------------------------------
# Round numerical noise
# -------------------------------------------------------------------------

def round_small_values(z):
    """
    Replace tiny negative values by zero.
    """

    if z < -1e-8:
        return 0.0

    return z