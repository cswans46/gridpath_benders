import pyomo.environ as pyo
import mosek

from pyomo.solvers.plugins.solvers.mosek_direct import (
    MOSEKDirect,
)


class MosekBendersDirect(MOSEKDirect):
    """
    MOSEK Direct solver with support for GridPath's native
    Benders disjunctive cuts.

    Ordinary Pyomo constraints are handled by MOSEKDirect.

    Benders optimality cuts are added directly to the underlying
    MOSEK Task as native disjunctive constraints (DJCs).

    Current implementation:
        - binary discrete linking variables are supported;
        - continuous linking variables are supported;
        - integer discrete linking variables are not yet supported
          by the native DJC implementation.

    No user-supplied Big-M is used for the Benders optimality cuts.
    """

    def _set_instance(self, model, kwds):

        # -----------------------------------------------------
        # First let the normal Pyomo -> MOSEK conversion happen.
        # -----------------------------------------------------

        super()._set_instance(model, kwds)

        # # -----------------------------------------------------
        # # Diagnostic: Pyomo -> MOSEK variable mapping.
        # # -----------------------------------------------------
        #
        # print("\nMOSEK variable mapping:")
        #
        # for pyomo_var, mosek_idx in (
        #     self._pyomo_var_to_solver_var_map.items()
        # ):
        #     print(
        #         "   ",
        #         pyomo_var.name,
        #         "->",
        #         mosek_idx,
        #     )

        # -----------------------------------------------------
        # Then add native Benders DJCs.
        # -----------------------------------------------------

        self._add_benders_djcs(model)

    def _add_benders_djcs(self, model):

        data = getattr(
            model,
            "_benders_native_djc_data",
            None,
        )

        if data is None:
            return

        benders_states = data["benders_states"]

        linking_variables_sub_by_type = (
            data["linking_variables_sub_by_type"]
        )

        task = self._solver_model

        # -----------------------------------------------------
        # Add every accumulated feasible Benders cut.
        # -----------------------------------------------------

        for (
            subproblem_key,
            state,
        ), state_data in benders_states.items():

            # -------------------------------------------------
            # Infeasible states are handled separately by the
            # combinatorial feasibility-cut machinery.
            # -------------------------------------------------

            if not state_data["feasible"]:
                continue

            variable_types = (
                linking_variables_sub_by_type[
                    subproblem_key
                ]
            )

            binary_names = (
                variable_types["binary"]
            )

            integer_names = (
                variable_types["integer"]
            )

            continuous_names = (
                variable_types["continuous"]
            )

            # -------------------------------------------------
            # Current native MOSEK implementation only handles
            # binary discrete states.
            #
            # Do not silently ignore integer linking variables.
            # -------------------------------------------------

            if integer_names:

                raise NotImplementedError(
                    "Native MOSEK Benders DJCs currently "
                    "support binary discrete linking variables "
                    "only. Integer linking variables were found "
                    f"for subproblem '{subproblem_key}': "
                    f"{integer_names}"
                )

            # -------------------------------------------------
            # Validate binary state dimension.
            # -------------------------------------------------

            if len(state) != len(binary_names):

                raise ValueError(
                    f"Benders discrete state for subproblem "
                    f"'{subproblem_key}' contains {len(state)} "
                    f"values, but {len(binary_names)} binary "
                    "linking variables were found."
                )

            # -------------------------------------------------
            # Map binary linking variables.
            # -------------------------------------------------

            binary_indices = []

            for name in binary_names:

                var = model.find_component(name)

                if var is None:
                    raise KeyError(
                        f"Binary linking variable '{name}' "
                        f"not found in planning model."
                    )

                try:
                    solver_idx = (
                        self._pyomo_var_to_solver_var_map[var]
                    )
                except KeyError:
                    raise KeyError(
                        f"Binary linking variable '{name}' "
                        "does not have a MOSEK variable index."
                    )

                binary_indices.append(
                    solver_idx
                )

            # -------------------------------------------------
            # Map theta.
            # -------------------------------------------------

            theta_var = model.vTHETA[
                subproblem_key
            ]

            try:
                theta_idx = (
                    self._pyomo_var_to_solver_var_map[
                        theta_var
                    ]
                )

            except KeyError:
                raise KeyError(
                    f"Theta variable for subproblem "
                    f"'{subproblem_key}' does not have "
                    "a MOSEK variable index."
                )

            # -------------------------------------------------
            # Add every cut stored for this discrete state.
            # -------------------------------------------------

            for cut_number, cut in enumerate(
                state_data["cuts"]
            ):

                continuous_indices = {}

                for name in continuous_names:

                    var = model.find_component(name)

                    if var is None:
                        raise KeyError(
                            f"Continuous linking variable "
                            f"'{name}' not found in planning model."
                        )

                    try:
                        solver_idx = (
                            self._pyomo_var_to_solver_var_map[
                                var
                            ]
                        )
                    except KeyError:
                        raise KeyError(
                            f"Continuous linking variable "
                            f"'{name}' does not have a "
                            "MOSEK variable index."
                        )

                    continuous_indices[name] = (
                        solver_idx
                    )

                # print(
                #     f"    Adding cut {cut_number}: "
                #     f"op_cost={cut['op_cost']:.6e}, "
                #     f"theta_coeff="
                #     f"{cut.get('theta_coeff', 1.0)}"
                # )

                add_benders_djc(
                    task=task,
                    theta_idx=theta_idx,
                    binary_indices=binary_indices,
                    binary_state=state,
                    continuous_indices=continuous_indices,
                    continuous_lambda=cut[
                        "continuous_lambda"
                    ],
                    continuous_point=cut[
                        "continuous_point"
                    ],
                    op_cost=cut[
                        "op_cost"
                    ],
                    theta_coeff=cut.get(
                        "theta_coeff",
                        1.0,
                    ),
                )

        # -----------------------------------------------------
        # Ensure all native DJCs are registered before solve.
        # -----------------------------------------------------

        task.putintparam(
            mosek.iparam.intpnt_solve_form,
            mosek.solveform.primal,
        )


def add_benders_djc(
    task,
    theta_idx,
    binary_indices,
    binary_state,
    continuous_indices,
    continuous_lambda,
    continuous_point,
    op_cost,
    theta_coeff=1.0,
):
    """
    Add one state-specific Benders optimality cut as a
    native MOSEK disjunctive constraint.

    Mathematically:

        [d = d^k,
         theta_coeff * theta >=
             op_cost + lambda' (x - xbar)]
        OR
        [d != d^k]

    Equivalently:

        d = d^k
            =>
        theta_coeff * theta
            >=
        op_cost + lambda' (x - xbar)

    The formulation contains no Big-M involving continuous
    linking variables.

    Binary discrete variables are used to construct the
    mismatch expression:

        m(d) =
            sum(1-d_i)   if d_i^k = 1
            sum(d_i)     if d_i^k = 0

    Thus:

        m(d) = 0
            <=> d = d^k

        m(d) >= 1
            <=> d != d^k
    """

    # =========================================================
    # 1. Validate the binary state.
    # =========================================================

    if len(binary_indices) != len(binary_state):

        raise ValueError(
            "Number of binary MOSEK indices does not match "
            "the length of the Benders binary state."
        )

    # =========================================================
    # 2. Mismatch affine expression
    #
    # m(d) =
    #
    #   sum(1-d_i)   if d_i^k = 1
    #   sum(d_i)     if d_i^k = 0
    #
    # =========================================================

    mismatch_afe = task.getnumafe()

    task.appendafes(1)

    mismatch_constant = 0.0

    for idx, target in zip(
        binary_indices,
        binary_state,
    ):

        target = float(target)

        if abs(target - 1.0) < 1e-8:

            # -------------------------------------------------
            # Target is 1:
            #
            #     mismatch contribution = 1 - d
            # -------------------------------------------------

            task.putafefentry(
                mismatch_afe,
                idx,
                -1.0,
            )

            mismatch_constant += 1.0

        elif abs(target) < 1e-8:

            # -------------------------------------------------
            # Target is 0:
            #
            #     mismatch contribution = d
            # -------------------------------------------------

            task.putafefentry(
                mismatch_afe,
                idx,
                1.0,
            )

        else:

            raise ValueError(
                "Native MOSEK Benders DJCs expect binary "
                "discrete states containing only 0/1 values. "
                f"Received state value {target}."
            )

    task.putafeg(
        mismatch_afe,
        mismatch_constant,
    )

    # =========================================================
    # 3. Benders affine expression
    #
    # theta_coeff * theta
    # - lambda' x
    # + lambda' xbar
    # - op_cost >= 0
    #
    # =========================================================

    cut_afe = task.getnumafe()

    task.appendafes(1)

    task.putafefentry(
        cut_afe,
        theta_idx,
        float(theta_coeff),
    )

    cut_constant = -float(
        op_cost
    )

    for name, solver_idx in (
        continuous_indices.items()
    ):

        lam = float(
            continuous_lambda[name]
        )

        xbar = float(
            continuous_point[name]
        )

        # -----------------------------------------------------
        # -lambda * x
        # -----------------------------------------------------

        if abs(lam) > 1e-15:

            task.putafefentry(
                cut_afe,
                solver_idx,
                -lam,
            )

        # -----------------------------------------------------
        # +lambda * xbar
        # -----------------------------------------------------

        cut_constant += (
            lam * xbar
        )

    task.putafeg(
        cut_afe,
        cut_constant,
    )

    # =========================================================
    # 4. Inactive branch
    #
    # m(d) - 1 >= 0
    #
    # For binary variables this is equivalent to:
    #
    #     d != d^k
    #
    # =========================================================

    inactive_afe = task.getnumafe()

    task.appendafes(1)

    for idx, target in zip(
        binary_indices,
        binary_state,
    ):

        target = float(target)

        if abs(target - 1.0) < 1e-8:

            # Contribution from 1-d is -d.
            task.putafefentry(
                inactive_afe,
                idx,
                -1.0,
            )

        elif abs(target) < 1e-8:

            # Contribution from d is +d.
            task.putafefentry(
                inactive_afe,
                idx,
                1.0,
            )

        else:

            raise ValueError(
                "Native MOSEK Benders DJCs expect binary "
                "discrete states containing only 0/1 values."
            )

    task.putafeg(
        inactive_afe,
        mismatch_constant - 1.0,
    )

    # print(
    #     f"      inactive constant = "
    #     f"{mismatch_constant - 1.0}"
    # )

    # =========================================================
    # 5. MOSEK domains
    # =========================================================

    # m(d) = 0
    zero_domain = (
        task.appendrzerodomain(1)
    )

    # Benders cut >= 0
    positive_domain = (
        task.appendrplusdomain(1)
    )

    # m(d) - 1 >= 0
    inactive_domain = (
        task.appendrplusdomain(1)
    )

    # =========================================================
    # 6. Native DJC
    #
    # Term 1:
    #
    #     m(d) = 0
    #     Benders cut >= 0
    #
    # Term 2:
    #
    #     m(d) - 1 >= 0
    #
    # Therefore:
    #
    #     [state matches AND cut holds]
    #         OR
    #     [state does not match]
    #
    # =========================================================

    djc_idx = task.getnumdjc()

    task.appenddjcs(1)

    task.putdjc(
        djc_idx,
        [
            zero_domain,
            positive_domain,
            inactive_domain,
        ],
        [
            mismatch_afe,
            cut_afe,
            inactive_afe,
        ],
        [
            0.0,
            0.0,
            0.0,
        ],
        [
            2,
            1,
        ],
    )

    return djc_idx


def create_mosek_benders_solver():

    return MosekBendersDirect()