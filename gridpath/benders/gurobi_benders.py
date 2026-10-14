import gurobipy as gp
from pyomo.solvers.plugins.solvers.gurobi_direct import (
    GurobiDirect,
)


class GurobiBendersDirect(GurobiDirect):

    def _set_instance(self, model, kwds):

        # Let Pyomo's GurobiDirect plugin first create the
        # standard Gurobi model and variable mapping.
        super()._set_instance(model, kwds)

        # Make sure all standard Pyomo variables/constraints
        # have been registered with the Gurobi model before
        # adding native Gurobi constraints.
        self._solver_model.update()

        # Add the native Gurobi Benders indicator constraints.
        self._add_benders_indicators(model)

    def _add_benders_indicators(self, model):

        # --------------------------------------------------------
        # Retrieve Benders data stored by algorithms.py
        # --------------------------------------------------------

        data = getattr(
            model,
            "_benders_native_indicator_data",
            None,
        )

        if data is None:
            return

        benders_states = data["benders_states"]

        linking_variables_sub_by_type = (
            data["linking_variables_sub_by_type"]
        )

        # --------------------------------------------------------
        # Add native indicator formulation for every known
        # feasible discrete state.
        # --------------------------------------------------------

        for (
            subproblem_key,
            state,
        ), state_data in benders_states.items():

            # ----------------------------------------------------
            # Only feasible operational states generate
            # optimality cuts.
            #
            # Infeasible states are handled separately by the
            # combinatorial feasibility-cut machinery.
            # ----------------------------------------------------

            if not state_data["feasible"]:
                continue

            variable_types = (
                linking_variables_sub_by_type[
                    subproblem_key
                ]
            )

            binary_names = variable_types["binary"]
            integer_names = variable_types["integer"]
            continuous_names = variable_types["continuous"]

            # ----------------------------------------------------
            # Get Gurobi variables corresponding to Pyomo binary
            # linking variables.
            # ----------------------------------------------------

            binary_vars = []

            for name in binary_names:

                var = model.find_component(name)

                if var is None:
                    raise KeyError(
                        f"Could not find binary linking "
                        f"variable '{name}'."
                    )

                if var not in self._pyomo_var_to_solver_var_map:
                    raise KeyError(
                        f"Pyomo binary linking variable "
                        f"'{name}' is not present in the "
                        f"Pyomo-to-Gurobi variable mapping."
                    )

                gurobi_var = (
                    self._pyomo_var_to_solver_var_map[var]
                )

                binary_vars.append(gurobi_var)

            # ----------------------------------------------------
            # Get Gurobi variables corresponding to Pyomo integer
            # linking variables.
            # ----------------------------------------------------

            integer_vars = []

            for name in integer_names:

                var = model.find_component(name)

                if var is None:
                    raise KeyError(
                        f"Could not find integer linking "
                        f"variable '{name}'."
                    )

                if var not in self._pyomo_var_to_solver_var_map:
                    raise KeyError(
                        f"Pyomo integer linking variable "
                        f"'{name}' is not present in the "
                        f"Pyomo-to-Gurobi variable mapping."
                    )

                gurobi_var = (
                    self._pyomo_var_to_solver_var_map[var]
                )

                integer_vars.append(gurobi_var)

            # ----------------------------------------------------
            # Get Gurobi variables corresponding to Pyomo
            # continuous linking variables.
            # ----------------------------------------------------

            continuous_vars = {}

            for name in continuous_names:

                var = model.find_component(name)

                if var is None:
                    raise KeyError(
                        f"Could not find continuous linking "
                        f"variable '{name}'."
                    )

                if var not in self._pyomo_var_to_solver_var_map:
                    raise KeyError(
                        f"Pyomo continuous linking variable "
                        f"'{name}' is not present in the "
                        f"Pyomo-to-Gurobi variable mapping."
                    )

                gurobi_var = (
                    self._pyomo_var_to_solver_var_map[var]
                )

                continuous_vars[name] = gurobi_var

            # ----------------------------------------------------
            # Get the Gurobi variable corresponding to theta.
            # ----------------------------------------------------

            theta_var = model.vTHETA[subproblem_key]

            if theta_var not in self._pyomo_var_to_solver_var_map:
                raise KeyError(
                    f"vTHETA[{subproblem_key}] is not present "
                    f"in the Pyomo-to-Gurobi variable mapping."
                )

            theta_gurobi = (
                self._pyomo_var_to_solver_var_map[
                    theta_var
                ]
            )

            # ----------------------------------------------------
            # Add one native indicator formulation for every
            # Benders cut associated with this discrete state.
            #
            # Multiple cuts can exist for the same state because
            # they may correspond to different continuous
            # incumbent points.
            # ----------------------------------------------------

            for cut_number, cut in enumerate(
                state_data["cuts"]
            ):

                add_gurobi_benders_indicator(
                    model=self._solver_model,
                    theta_var=theta_gurobi,
                    binary_vars=binary_vars,
                    integer_vars=integer_vars,
                    discrete_state=state,
                    continuous_vars=continuous_vars,
                    continuous_lambda=cut[
                        "continuous_lambda"
                    ],
                    continuous_point=cut[
                        "continuous_point"
                    ],
                    op_cost=cut["op_cost"],
                    theta_coeff=cut.get(
                        "theta_coeff",
                        1.0,
                    ),
                )

        # --------------------------------------------------------
        # Register all newly created Gurobi variables and
        # constraints.
        # --------------------------------------------------------

        self._solver_model.update()


def add_gurobi_benders_indicator(
    model,
    theta_var,
    binary_vars,
    integer_vars,
    discrete_state,
    continuous_vars,
    continuous_lambda,
    continuous_point,
    op_cost,
    theta_coeff=1.0,
):
    """
    Add a state-conditioned Benders optimality cut using
    native Gurobi indicator constraints.

    For a discrete state d^k, the resulting logic is:

        d == d^k
            =>
        theta_coeff * theta
            >=
        op_cost
        + sum(lambda_i * (x_i - xbar_i))

    Equivalently:

        d == d^k
            =>
        theta_coeff * theta
        - op_cost
        - sum(lambda_i * x_i)
        + sum(lambda_i * xbar_i)
        >= 0

    No user-supplied Big-M is used.

    Continuous linking variables therefore do not require
    finite upper or lower bounds for this formulation.
    """

    # ============================================================
    # 1. Construct mismatch variables
    # ============================================================

    mismatch_vars = []

    # ------------------------------------------------------------
    # Binary linking variables
    # ------------------------------------------------------------

    for i, var in enumerate(binary_vars):

        target = int(discrete_state[i])

        mismatch = model.addVar(
            vtype=gp.GRB.BINARY,
            name=(
                f"_benders_mismatch_bin_"
                f"{model.NumVars}_{i}"
            ),
        )

        mismatch_vars.append(mismatch)

        if target == 1:

            # If target = 1:
            #
            #     mismatch = 1 - var
            #
            # Therefore:
            #
            #     var = 1 -> mismatch = 0
            #     var = 0 -> mismatch = 1

            model.addConstr(
                mismatch + var == 1
            )

        elif target == 0:

            # If target = 0:
            #
            #     mismatch = var
            #
            # Therefore:
            #
            #     var = 0 -> mismatch = 0
            #     var = 1 -> mismatch = 1

            model.addConstr(
                mismatch == var
            )

        else:

            raise ValueError(
                "Binary Benders state must contain "
                "only 0/1 values."
            )

    # ------------------------------------------------------------
    # Integer linking variables
    #
    # An integer variable mismatches its incumbent state if it
    # is either strictly below or strictly above the incumbent
    # value.
    #
    # No Big-M is required.
    # ------------------------------------------------------------

    integer_offset = len(binary_vars)

    for i, var in enumerate(integer_vars):

        target = int(
            discrete_state[
                integer_offset + i
            ]
        )

        mismatch = model.addVar(
            vtype=gp.GRB.BINARY,
            name=(
                f"_benders_mismatch_int_"
                f"{model.NumVars}_{i}"
            ),
        )

        below = model.addVar(
            vtype=gp.GRB.BINARY,
            name=(
                f"_benders_below_"
                f"{model.NumVars}_{i}"
            ),
        )

        above = model.addVar(
            vtype=gp.GRB.BINARY,
            name=(
                f"_benders_above_"
                f"{model.NumVars}_{i}"
            ),
        )

        mismatch_vars.append(mismatch)

        # --------------------------------------------------------
        # If mismatch = 0:
        #
        #     below = 0
        #     above = 0
        #
        # If mismatch = 1:
        #
        #     exactly one of below/above is 1.
        # --------------------------------------------------------

        model.addConstr(
            below + above == mismatch
        )

        # --------------------------------------------------------
        # mismatch = 0 -> var == target
        # --------------------------------------------------------

        model.addGenConstrIndicator(
            mismatch,
            False,
            var,
            gp.GRB.EQUAL,
            target,
        )

        # --------------------------------------------------------
        # below = 1 -> var <= target - 1
        # --------------------------------------------------------

        model.addGenConstrIndicator(
            below,
            True,
            var,
            gp.GRB.LESS_EQUAL,
            target - 1,
        )

        # --------------------------------------------------------
        # above = 1 -> var >= target + 1
        # --------------------------------------------------------

        model.addGenConstrIndicator(
            above,
            True,
            var,
            gp.GRB.GREATER_EQUAL,
            target + 1,
        )

    # ============================================================
    # 2. Construct state-mismatch binary
    # ============================================================

    state_mismatch = model.addVar(
        vtype=gp.GRB.BINARY,
        name=(
            f"_benders_state_mismatch_"
            f"{model.NumVars}"
        ),
    )

    if mismatch_vars:

        # --------------------------------------------------------
        # state_mismatch >= every individual mismatch
        #
        # Therefore, if any variable mismatches:
        #
        #     state_mismatch = 1
        # --------------------------------------------------------

        for mismatch in mismatch_vars:

            model.addConstr(
                state_mismatch >= mismatch
            )

        # --------------------------------------------------------
        # state_mismatch <= sum(mismatches)
        #
        # Therefore, if all variables match:
        #
        #     state_mismatch = 0
        # --------------------------------------------------------

        model.addConstr(
            state_mismatch
            <= gp.quicksum(mismatch_vars)
        )

    else:

        # --------------------------------------------------------
        # No discrete linking variables exist.
        #
        # Therefore the state always matches.
        # --------------------------------------------------------

        model.addConstr(
            state_mismatch == 0
        )

    # ============================================================
    # 3. Construct continuous Benders cut
    # ============================================================

    # Starting from:
    #
    #   theta_coeff * theta
    #       >=
    #   op_cost + sum(lambda_i * (x_i - xbar_i))
    #
    # Rearranging:
    #
    #   theta_coeff * theta
    #   - op_cost
    #   - sum(lambda_i * x_i)
    #   + sum(lambda_i * xbar_i)
    #   >= 0

    cut_expr = (
        theta_coeff * theta_var
        - float(op_cost)
    )

    for name, var in continuous_vars.items():

        lam = float(
            continuous_lambda[name]
        )

        xbar = float(
            continuous_point[name]
        )

        cut_expr += (
            -lam * var
            + lam * xbar
        )

    # ============================================================
    # 4. Native Gurobi indicator
    #
    # state_mismatch = 0
    #
    #       =>
    #
    # cut_expr >= 0
    #
    # Thus the Benders cut is enforced ONLY when the master
    # discrete variables reproduce the discrete state for which
    # this operational subproblem was solved.
    # ============================================================

    model.addGenConstrIndicator(
        state_mismatch,
        False,
        cut_expr,
        gp.GRB.GREATER_EQUAL,
        0.0,
    )


def create_gurobi_benders_solver():

    return GurobiBendersDirect()