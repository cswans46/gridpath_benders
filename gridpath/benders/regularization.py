import pyomo.environ as pyo

from pyomo.environ import (
    Constraint,
    Objective,
    minimize,
    maximize,
    value,
)

from pyomo.opt import (
    SolverStatus,
    TerminationCondition,
)

from gridpath.benders.planning import process_planning_sol

import pyomo.environ as pyo


def build_l1_stabilization(
    model,
    stabilization_variables,
    planning_sol,
    weights=None,
):
    if not stabilization_variables:
        return 0.0

    # Normalize VarData -> full Pyomo names
    stabilization_variables = [
        var.name if hasattr(var, "name") else var
        for var in stabilization_variables
    ]

    if weights is None:
        weights = {
            name: 1.0
            for name in stabilization_variables
        }

    # ---------------------------------------------------------
    # Create auxiliary components once
    # ---------------------------------------------------------

    if not hasattr(model, "BendersL1Positive"):

        model.BendersL1Positive = pyo.VarList(
            domain=pyo.NonNegativeReals
        )

        model.BendersL1Negative = pyo.VarList(
            domain=pyo.NonNegativeReals
        )

        model.BendersL1Constraint = pyo.ConstraintList()

        model.BendersL1VariableMap = {}

        for name in stabilization_variables:

            pos = model.BendersL1Positive.add()
            neg = model.BendersL1Negative.add()

            idx = len(model.BendersL1VariableMap)

            model.BendersL1VariableMap[name] = (
                idx,
                pos,
                neg,
            )

            var = model.find_component(name)

            if var is None:
                raise KeyError(
                    f"Could not find stabilization variable '{name}'."
                )

            if name not in planning_sol.values:
                raise KeyError(
                    f"Stabilization variable '{name}' is not "
                    "present in the planning solution."
                )

            xbar = planning_sol.values[name]

            model.BendersL1Constraint.add(
                var - xbar == pos - neg
            )

    else:

        existing_names = set(
            model.BendersL1VariableMap.keys()
        )

        if existing_names != set(stabilization_variables):
            raise ValueError(
                "The set of stabilization variables changed "
                "between Benders iterations."
            )

        # Update the existing constraints
        for name in stabilization_variables:

            _, pos, neg = (
                model.BendersL1VariableMap[name]
            )

            var = model.find_component(name)

            if var is None:
                raise KeyError(
                    f"Could not find stabilization variable '{name}'."
                )

            if name not in planning_sol.values:
                raise KeyError(
                    f"Stabilization variable '{name}' is not "
                    "present in the planning solution."
                )

            xbar = planning_sol.values[name]

            # ConstraintList entries are 1-based
            idx = (
                list(
                    model.BendersL1VariableMap.keys()
                ).index(name)
                + 1
            )

            model.BendersL1Constraint[idx].set_value(
                var - xbar == pos - neg
            )

    # ---------------------------------------------------------
    # Objective
    # ---------------------------------------------------------

    return sum(
        weights[name]
        * (
            pos + neg
        )
        for name, (_, pos, neg)
        in model.BendersL1VariableMap.items()
    )

def _normalize_stabilization_variables(stabilization_variables):
    """Return stabilization variables as Pyomo component-data names."""
    return [
        var.name if hasattr(var, "name") else var
        for var in stabilization_variables
    ]

def build_l2_stabilization(
    model,
    stabilization_variables,
    planning_sol,
    weights=None,
    use_conic=False,
):
    """
    Build the L2 stabilization term.

    Standard formulation:
        minimize    sum_i w_i (x_i - xbar_i)^2

    This is represented directly as a quadratic expression unless
    use_conic=True.

    Conic formulation:
        minimize    t

        subject to
            sum_i w_i (x_i - xbar_i)^2 / objective_scale^2 <= t

    represented exactly using a rotated quadratic cone in the
    custom MOSEK solver. The common positive scale leaves the minimizer
    unchanged and keeps cone affine data near unit scale.

    The conic formulation is used for MOSEK L2 level-set solves to avoid
    numerically huge linear coefficients from expanding large reference
    points into a quadratic objective. The same epigraph transformation is
    also used by non-MOSEK native-DJC configurations where supported.
    """

    if not stabilization_variables:
        return 0.0

    # Normalize to Pyomo component names.
    stabilization_variables = [
        var.name if hasattr(var, "name") else var
        for var in stabilization_variables
    ]

    if weights is None:
        weights = {
            name: 1.0
            for name in stabilization_variables
        }

    # ------------------------------------------------------------
    # Standard L2 formulation
    # ------------------------------------------------------------
    if not use_conic:
        expr = 0.0
        for name in stabilization_variables:
            var = model.find_component(name)
            if var is None:
                raise KeyError(
                    f"Could not find stabilization variable {name}"
                )
            if name not in planning_sol.values:
                raise KeyError(
                    f"{name} not found in planning solution"
                )
            xbar = float(planning_sol.values[name])
            weight = float(weights[name])
            if weight < 0.0:
                raise ValueError(
                    f"L2 stabilization weight for {name} is negative: "
                    f"{weight}"
                )
            expr += weight * (var - xbar) ** 2

        return expr

    # ------------------------------------------------------------
    # Exact conic epigraph formulation
    # ------------------------------------------------------------

    # Remove any previous L2 conic metadata/components.
    if hasattr(model, "BendersL2ConicData"):
        delattr(model, "BendersL2ConicData")

    if hasattr(model, "BendersL2Epigraph"):
        model.del_component(model.BendersL2Epigraph)

    model.BendersL2Epigraph = pyo.Var(
        domain=pyo.NonNegativeReals
    )

    # Store everything the native MOSEK layer needs.
    model.BendersL2ConicData = {
        "variables": list(stabilization_variables),
        "weights": {
            name: float(weights[name])
            for name in stabilization_variables
        },
        "points": {
            name: float(planning_sol.values[name])
            for name in stabilization_variables
        },
    }

    # The level-set subproblem minimizes t.
    #
    # The custom MOSEK layer will add the exact constraint
    #
    #       sum_i w_i (x_i-xbar_i)^2 <= t
    #
    # as a rotated quadratic cone.
    return model.BendersL2Epigraph

def solve_int_level_set_problem(
    model,
    planning_variables,
    planning_sol,
    LB,
    UB,
    gamma,
    solver,
    stabilization_method="none",
    stabilization_variables=None,
    stabilization_weights=None,
    master_discrete_method="pj_relax",
    fallback_planning_sol=None,
    required_variables=None,
):
    """
    Solve the Benders master over the interior level set.

    ``required_variables`` (optional) names the planning variables that must
    have a value after the solve; it is passed to ``process_planning_sol`` so
    planning variables not used by any subproblem may be left unset, exactly
    as in ``solve_planning_problem``.

    The level-set constraint is

        MasterObjective >= UB - gamma * (UB - LB)

    The active objective selects the next trial point:

        none:
            minimize MasterObjective (Pecci--Jenkins levelization)

        l1:
            sum_i w_i |x_i - xbar_i|

        l2:
            sum_i w_i (x_i - xbar_i)^2

    L1 and L2 act only on the supplied continuous linking variables.
    ``none`` retains the original P&J level-set objective. If that linear
    MOSEK solve is non-optimal, it is retried with dual simplex; the model
    objective and level-set constraint are unchanged by that retry.
    """

    # ---------------------------------------------------------
    # Validate arguments
    # ---------------------------------------------------------

    if stabilization_method not in {
        "none",
        "l1",
        "l2",
    }:
        raise ValueError(
            f"Unknown stabilization method: "
            f"{stabilization_method!r}. "
            "Expected 'none', 'l1', or 'l2'."
        )

    if stabilization_variables is None:
        stabilization_variables = []

    # ---------------------------------------------------------
    # Validate bounds
    # ---------------------------------------------------------

    if UB < LB:
        raise ValueError(
            "Cannot construct level-set constraint because "
            f"UB < LB: UB={UB}, LB={LB}"
        )

    # ---------------------------------------------------------
    # Compute level-set bound
    # ---------------------------------------------------------

    level_bound = UB - gamma * (UB - LB)

    # ---------------------------------------------------------
    # Ensure master objective exists
    # ---------------------------------------------------------

    if not hasattr(model, "MasterObjective"):
        raise AttributeError(
            "Planning model does not contain MasterObjective."
        )

    # ---------------------------------------------------------
    # Add / update level-set constraint
    # ---------------------------------------------------------

    if not hasattr(model, "cLevelSet"):

        model.cLevelSet = Constraint(
            expr=Constraint.Skip
        )

    model.cLevelSet.set_value(
        model.MasterObjective.expr >= level_bound
    )

    # ---------------------------------------------------------
    # Create level objective once
    # ---------------------------------------------------------

    if not hasattr(model, "LevelObjective"):

        model.LevelObjective = Objective(
            expr=0.0,
            sense=minimize,
        )

        model.LevelObjective.deactivate()

    # ---------------------------------------------------------
    # Construct stabilization objective
    # ---------------------------------------------------------

    if stabilization_method == "none":

        # -----------------------------------------------------
        # Original Pecci & Jenkins regularization
        #
        # Minimize the ordinary master objective while forcing
        # the solution into the interior level set.
        # -----------------------------------------------------

        stabilization_expr = model.MasterObjective.expr

    elif stabilization_method == "l1":

        stabilization_expr = build_l1_stabilization(
            model,
            stabilization_variables,
            planning_sol,
            stabilization_weights,
        )


    elif stabilization_method == "l2":

        use_conic_l2 = (
            master_discrete_method == "djc"
            or str(getattr(solver, "name", "")).lower().startswith(
                "mosek"
            )
        )

        stabilization_expr = build_l2_stabilization(
            model,
            stabilization_variables,
            planning_sol,
            weights=stabilization_weights,
            use_conic=use_conic_l2,
        )

    else:

        raise ValueError(
            f"Unknown stabilization method: "
            f"{stabilization_method!r}"
        )

    # ---------------------------------------------------------
    # Update objective expression
    # ---------------------------------------------------------

    model.LevelObjective.set_value(
        stabilization_expr
    )

    # ---------------------------------------------------------
    # Switch objectives
    # ---------------------------------------------------------

    model.MasterObjective.deactivate()
    model.LevelObjective.activate()
    model.cLevelSet.activate()

    # ---------------------------------------------------------
    # Solve
    # ---------------------------------------------------------

    results = solver.solve(
        model,
        tee=False,
    )

    status = results.solver.status
    term = results.solver.termination_condition

    if (
        stabilization_method == "none"
        and term != TerminationCondition.optimal
        and str(getattr(solver, "name", "")).lower().startswith(
            "mosek"
        )
    ):
        optimizer_key = "MSK_IPAR_OPTIMIZER"
        had_optimizer_option = optimizer_key in solver.options
        previous_optimizer = solver.options.get(optimizer_key)
        try:
            solver.options[optimizer_key] = (
                "MSK_OPTIMIZER_DUAL_SIMPLEX"
            )
            results = solver.solve(
                model,
                tee=False,
            )
        finally:
            if had_optimizer_option:
                solver.options[optimizer_key] = previous_optimizer
            else:
                solver.options.pop(optimizer_key, None)

        status = results.solver.status
        term = results.solver.termination_condition

    # ---------------------------------------------------------
    # Extract solution
    # ---------------------------------------------------------

    if (
        status == SolverStatus.ok
        and term == TerminationCondition.optimal
    ):

        planning_sol = process_planning_sol(
            model,
            planning_variables,
            required_variables=required_variables,
        )

    else:

        print(
            "WARNING: interior level-set solve failed"
        )
        if fallback_planning_sol is not None:
            planning_sol = fallback_planning_sol

    # ---------------------------------------------------------
    # Restore ordinary master state
    # ---------------------------------------------------------

    model.LevelObjective.deactivate()
    model.MasterObjective.activate()
    model.cLevelSet.deactivate()

    return planning_sol