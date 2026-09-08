from pyomo.environ import (
    Constraint,
    Objective,
    minimize,
    maximize,
    value,
)
from pyomo.opt import SolverStatus, TerminationCondition
from gridpath.benders.planning import process_planning_sol


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
    level_bound = UB - gamma * (UB - LB)

    # ------------------------------------------------------------
    # Add / update level-set constraint (DO NOT delete/recreate)
    # ------------------------------------------------------------
    if not hasattr(model, "cLevelSet"):
        model.cLevelSet = Constraint(
            expr=Constraint.Skip
        )

    model.cLevelSet.set_value(
        model.ePlanningNPV
        - sum(model.vTHETA[w] for w in model.vTHETA)
        >= level_bound
    )

    # ------------------------------------------------------------
    # Ensure master objective exists
    # (should already exist from model setup stage)
    # ------------------------------------------------------------
    if not hasattr(model, "MasterObjective"):
        model.MasterObjective = Objective(
            expr=(
                    model.ePlanningNPV
                    - sum(model.vTHETA[w] for w in model.vTHETA)
            ),
            sense=maximize,
        )

    # ------------------------------------------------------------
    # Create dummy level objective once (if not exists)
    # ------------------------------------------------------------
    if not hasattr(model, "LevelObjective"):
        first_w = next(iter(model.vTHETA))

        model.LevelObjective = Objective(
            expr=0 * model.vTHETA[first_w],
            sense=minimize,
        )
        model.LevelObjective.deactivate()

    # ------------------------------------------------------------
    # Switch objectives (SAFE for MOSEK persistent)
    # ------------------------------------------------------------
    model.MasterObjective.deactivate()
    model.LevelObjective.activate()
    model.cLevelSet.activate()

    # import pyomo.environ as pyo

    # print("\n========== LEVEL SET BEFORE SOLVE ==========")
    #
    # print("level_bound =", level_bound)
    #
    # print("cLevelSet.active =", model.cLevelSet.active)
    #
    # print("cLevelSet.expr =", model.cLevelSet.expr)
    #
    # print("cLevelSet.lower =", pyo.value(model.cLevelSet.lower))
    #
    # print("cLevelSet.upper =", model.cLevelSet.upper)
    #
    # print(
    #     "current master value =",
    #     pyo.value(
    #         model.ePlanningNPV
    #         - sum(model.vTHETA[w] for w in model.vTHETA)
    #     )
    # )
    #
    # print("MasterObjective.active =", model.MasterObjective.active)
    # print("LevelObjective.active =", model.LevelObjective.active)
    #
    # print("============================================")

    # ------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------
    results = solver.solve(model, tee=False)

    # level_value = pyo.value(
    #     model.ePlanningNPV
    #     - sum(model.vTHETA[w] for w in model.vTHETA)
    # )

    # print("\n========== LEVEL SET AFTER SOLVE ==========")
    #
    # print("termination =", results.solver.termination_condition)
    #
    # print("level_bound =", level_bound)
    #
    # print("level_value =", level_value)
    #
    # print("violation   =", level_bound - level_value)
    #
    # print("cLevelSet.active =", model.cLevelSet.active)
    #
    # print("===========================================")

    tol = 1e-6 * max(1.0, abs(level_bound))

    # if level_value < level_bound - tol:
    #     raise RuntimeError(
    #         "\nLEVEL-SET SOLVE RETURNED AN INFEASIBLE SOLUTION\n"
    #         f"level value = {level_value:.12e}\n"
    #         f"level bound = {level_bound:.12e}\n"
    #         f"violation   = {level_bound - level_value:.12e}\n"
    #     )

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
    model.cLevelSet.deactivate()

    return planning_sol