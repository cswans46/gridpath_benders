from pyomo.environ import (
    Constraint,
    Objective,
    minimize,
    value,
)
from pyomo.opt import SolverStatus, TerminationCondition


def solve_int_level_set_problem(
    model,
    planning_variables,
    planning_sol,
    LB,
    UB,
    gamma,
    solver,
):
    """
    Interior level-set stabilization for Benders decomposition.
    """

    # ------------------------------------------------------------
    # Compute level-set bound
    # ------------------------------------------------------------
    level_bound = LB + gamma * (UB - LB)

    # ------------------------------------------------------------
    # Add / update level-set constraint (DO NOT delete/recreate)
    # ------------------------------------------------------------
    if not hasattr(model, "cLevelSet"):
        model.cLevelSet = Constraint(
            expr=Constraint.Skip
        )

    model.cLevelSet.set_value(
        model.ePlanningCost
        + sum(model.vTHETA[w] for w in model.vTHETA)
        <= level_bound
    )

    # ------------------------------------------------------------
    # Ensure master objective exists
    # (should already exist from model setup stage)
    # ------------------------------------------------------------
    if not hasattr(model, "MasterObjective"):
        model.MasterObjective = Objective(
            expr=model.ePlanningCost
            + sum(model.vTHETA[w] for w in model.vTHETA),
            sense=minimize,
        )

    # ------------------------------------------------------------
    # Create dummy level objective once (if not exists)
    # ------------------------------------------------------------
    if not hasattr(model, "LevelObjective"):
        model.LevelObjective = Objective(
            expr=0.0,
            sense=minimize,
        )
        model.LevelObjective.deactivate()

    # ------------------------------------------------------------
    # Switch objectives (SAFE for MOSEK persistent)
    # ------------------------------------------------------------
    model.MasterObjective.deactivate()
    model.LevelObjective.activate()

    # ------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------
    results = solver.solve(model, tee=False)

    status = results.solver.status
    term = results.solver.termination_condition

    # ------------------------------------------------------------
    # Extract solution
    # ------------------------------------------------------------
    if (
        status == SolverStatus.ok
        and term
        in (
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
        )
    ):
        planning_sol = process_planning_sol(
            model,
            planning_variables,
        )
    else:
        print("WARNING: interior level-set solve failed")

    # ------------------------------------------------------------
    # Restore original objective state (NO deletion)
    # ------------------------------------------------------------
    model.LevelObjective.deactivate()
    model.MasterObjective.activate()

    return planning_sol