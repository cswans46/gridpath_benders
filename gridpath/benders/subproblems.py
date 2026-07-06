from pyomo.environ import value
from pyomo.environ import (
    Var,
    ConstraintList,
    Objective,
    ObjectiveList,
    value,
    minimize,
    SolverFactory,
)

def add_slacks_to_subproblem(model):

    model.EQ_CONSTRAINTS = ConstraintList()
    model.LE_CONSTRAINTS = ConstraintList()
    model.GE_CONSTRAINTS = ConstraintList()

    model.slack_max = Var(domain=NonNegativeReals)
    model.slack_eq = Var(model.EQ_CONSTRAINTS, domain=Reals)

    # <= constraints
    for c in model.LE_CONSTRAINTS.values():
        c.set_value(c.body + model.slack_max <= c.upper)

    # >= constraints
    for c in model.GE_CONSTRAINTS.values():
        c.set_value(c.body - model.slack_max >= c.lower)

    # = constraints
    for i, c in model.EQ_CONSTRAINTS.items():
        c.set_value(c.body - model.slack_eq[i] == 0)

        model.add_component(
            f"slack_eq_bound_{i}",
            Constraint(expr=model.slack_eq[i] <= model.slack_max),
        )
        model.add_component(
            f"slack_eq_bound_neg_{i}",
            Constraint(expr=-model.slack_eq[i] <= model.slack_max),
        )

    model.slack_fix = Constraint(expr=model.slack_max == 0)

def fix_linking_variables(model, planning_sol, linking_vars):

    for name in linking_vars:
        var = model.find_component(name)

        var.fix(planning_sol.values[name])

        # mimic JuMP behavior
        var.domain = Reals

def solve_subproblem(model, planning_sol, linking_vars, expect_feasible, solver):

    fix_linking_variables(model, planning_sol, linking_vars)

    results = solver.solve(model, tee=False)

    status = results.solver.status
    term = results.solver.termination_condition

    # ------------------------------------------------------------
    # feasible case
    # ------------------------------------------------------------
    if term == "optimal":

        op_cost = value(model.obj)

        duals = [
            model.dual[model.find_component(name)]
            for name in linking_vars
        ]

        return {
            "op_cost": op_cost,
            "lambda": duals,
            "theta_coeff": 1,
        }

    # ------------------------------------------------------------
    # infeasible case
    # ------------------------------------------------------------
    if expect_feasible:

        raise RuntimeError("Subproblem infeasible (expected feasible).")

    # ------------------------------------------------------------
    # feasibility cut
    # ------------------------------------------------------------
    model.slack_fix.deactivate()

    model.obj.deactivate()
    model.feasibility_obj = Objective(expr=model.slack_max)

    solver.solve(model)

    op_cost = value(model.feasibility_obj)

    duals = [
        model.dual[model.find_component(name)]
        for name in linking_vars
    ]

    model.feasibility_obj.deactivate()
    model.obj.activate()
    model.slack_fix.activate()

    return {
        "op_cost": op_cost,
        "lambda": duals,
        "theta_coeff": 0,
    }

def solve_local_subproblems(subproblems, planning_sol, expect_feasible, solver):

    results = {}

    for sp in subproblems:

        m = sp["model"]
        w = sp["subproblem_index"]
        linking = sp["linking_variables_sub"]

        results[w] = solve_subproblem(
            m,
            planning_sol,
            linking,
            expect_feasible,
            solver,
        )

    return results

def solve_subproblems_parallel(subproblems, planning_sol, expect_feasible, solver):

    results = Parallel(n_jobs=-1)(
        delayed(solve_local_subproblems)(
            subproblems,
            planning_sol,
            expect_feasible,
            solver,
        )
    )

    merged = {}
    for r in results:
        merged.update(r)

    return merged