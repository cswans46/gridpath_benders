import pyomo.environ as pyo
import mosek
from gridpath.benders.planning import BENDERS_THETA_SCALE

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

        # Pyomo creates the normal linear master problem.
        #
        # When L2 stabilization supplies conic metadata, its epigraph
        # is represented as a native MOSEK affine conic constraint.
        super()._set_instance(model, kwds)

        self._add_benders_l2_cone(model)

        self._add_benders_djcs(model)

    # Set by master_start.apply_master_interior_point. With the interior-point
    # optimiser and no crossover (MSK_IPAR_INTPNT_BASIS = never) MOSEK holds no
    # basic solution, so reading the "bas" solution fails with
    # "The solution with code 1 is not defined". Read the interior solution.
    _benders_interior_point = False

    def _set_whichsol(self):
        super()._set_whichsol()
        if (
            self._benders_interior_point
            and self._solver_model.getnumintvar() < 1
            and self._whichsol == mosek.soltype.bas
        ):
            self._whichsol = mosek.soltype.itr

    def _apply_solver(self):

        return super()._apply_solver()

    def _add_benders_l2_cone(self, model):
        """
        Add the exact L2 stabilization epigraph as a MOSEK
        rotated quadratic cone.

        Represents:

            t >= sum_i w_i (x_i - xbar_i)^2

        using:

            (t, 1/2, sqrt(w_i)*(x_i-xbar_i)) in Q_r

        because MOSEK's rotated quadratic cone satisfies

            2*y0*y1 >= sum_i y_i^2.
        """

        data = getattr(model, "BendersL2ConicData", None)

        if data is None:
            return

        variables = data["variables"]
        weights = data["weights"]
        points = data["points"]

        if not variables:
            return

        task = self._solver_model

        # ------------------------------------------------------------
        # Get solver indices of the Pyomo variables.
        # ------------------------------------------------------------

        def solver_index(pyomo_var):
            try:
                return self._pyomo_var_to_solver_var_map[pyomo_var]
            except Exception as exc:
                raise RuntimeError(
                    "Could not find MOSEK solver index for "
                    f"Pyomo variable {pyomo_var.name}"
                ) from exc

        theta_var = model.BendersL2Epigraph

        theta_idx = solver_index(theta_var)

        # ------------------------------------------------------------
        # Create n + 2 affine expressions:
        #
        #   AFE 0     = t
        #   AFE 1     = 1/2
        #   AFE 2+i   = sqrt(w_i) * (x_i - xbar_i)
        # ------------------------------------------------------------

        n = len(variables)

        afe_start = task.getnumafe()

        task.appendafes(n + 2)

        # AFE 0 = t
        task.putafefentry(
            afe_start,
            theta_idx,
            1.0,
        )

        task.putafeg(
            afe_start,
            0.0,
        )

        # AFE 1 = 1/2
        #
        # This is an affine expression with no variable terms.
        task.putafeg(
            afe_start + 1,
            0.5,
        )

        # AFE 2+i =
        #
        #     sqrt(w_i) * x_i
        #     - sqrt(w_i) * xbar_i
        #
        for i, name in enumerate(variables):

            var = model.find_component(name)

            if var is None:
                raise KeyError(
                    f"Could not find L2 stabilization variable {name}"
                )

            weight = float(weights[name])

            if weight < 0.0:
                raise ValueError(
                    f"L2 stabilization weight for {name} "
                    f"is negative: {weight}"
                )

            sqrt_weight = weight ** 0.5

            afe_idx = afe_start + 2 + i

            var_idx = solver_index(var)

            if abs(sqrt_weight) > 0.0:
                task.putafefentry(
                    afe_idx,
                    var_idx,
                    sqrt_weight,
                )

            task.putafeg(
                afe_idx,
                -sqrt_weight * float(points[name]),
            )

        # ------------------------------------------------------------
        # Rotated quadratic cone:
        #
        #   2 * y0 * y1 >= sum_i y_i^2
        #
        # with
        #
        #   y0 = t
        #   y1 = 1/2
        #   y_i = sqrt(w_i)*(x_i-xbar_i)
        #
        # gives
        #
        #   t >= sum_i w_i*(x_i-xbar_i)^2.
        # ------------------------------------------------------------

        cone_dim = n + 2

        rquad_domain = task.appendrquadraticconedomain(
            cone_dim
        )

        task.appendaccseq(
            rquad_domain,
            afe_start,
            None,
        )

    def _add_benders_djcs(self, model):

        data = getattr(
            model,
            "_benders_native_djc_data",
            None,
        )

        if data is None:
            return

        benders_states = data["benders_states"]
        benders_feasibility_states = data.get(
            "benders_feasibility_states",
            {},
        )

        linking_variables_sub_by_type = (
            data["linking_variables_sub_by_type"]
        )

        task = self._solver_model

        task.putintparam(
            mosek.iparam.mio_conic_outer_approximation,
            mosek.onoffkey.on,
        )

        task.putintparam(
            mosek.iparam.mio_qcqo_reformulation_method,
            mosek.miqcqoreformmethod.linearization
        )

        # ---------------------------------------------------------
        # Add every accumulated Benders state-specific cut.
        #
        # Current structure:
        #
        #   benders_states[
        #       ScenarioKey
        #   ] = [
        #       state_data,
        #       state_data,
        #       ...
        #   ]
        #
        # ---------------------------------------------------------

        subproblem_keys = list(
            dict.fromkeys(
                list(benders_states) + list(benders_feasibility_states)
            )
        )
        for subproblem_key in subproblem_keys:

            state_list = (
                benders_states.get(subproblem_key, [])
                + benders_feasibility_states.get(subproblem_key, [])
            )

            variable_types = (
                linking_variables_sub_by_type[
                    subproblem_key
                ]
            )

            binary_names = variable_types["binary"]
            integer_names = variable_types["integer"]
            continuous_names = variable_types["continuous"]

            # -----------------------------------------------------
            # Native DJCs currently support binary discrete linking
            # variables only.
            # -----------------------------------------------------

            if integer_names:
                raise NotImplementedError(
                    "Native MOSEK Benders DJCs currently "
                    "support binary discrete linking variables "
                    "only. Integer linking variables were found "
                    f"for subproblem '{subproblem_key}': "
                    f"{integer_names}"
                )

            # -----------------------------------------------------
            # Map binary linking variables once per subproblem.
            # -----------------------------------------------------

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

                binary_indices.append(solver_idx)

            # -----------------------------------------------------
            # Map theta once per subproblem.
            # -----------------------------------------------------

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

            # -----------------------------------------------------
            # Map continuous linking variables once per
            # subproblem.
            # -----------------------------------------------------

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

                continuous_indices[name] = solver_idx

            # -----------------------------------------------------
            # Add each accumulated state-specific Benders cut.
            # -----------------------------------------------------

            for cut_number, state_data in enumerate(
                    state_list
            ):

                discrete_state = state_data[
                    "master_discrete_state"
                ]

                binary_state_dict = discrete_state[
                    "binary"
                ]

                integer_state_dict = discrete_state[
                    "integer"
                ]

                # -------------------------------------------------
                # This implementation only supports binary states.
                # -------------------------------------------------

                if integer_state_dict:
                    raise NotImplementedError(
                        "Native MOSEK Benders DJCs currently "
                        "support binary discrete linking variables "
                        "only. An integer discrete state was "
                        f"found for subproblem "
                        f"'{subproblem_key}', cut "
                        f"{cut_number}: "
                        f"{integer_state_dict}"
                    )

                # -------------------------------------------------
                # Construct the binary state vector in exactly the
                # same order as binary_names.
                # -------------------------------------------------

                missing_binary_names = [
                    name
                    for name in binary_names
                    if name not in binary_state_dict
                ]

                if missing_binary_names:
                    raise KeyError(
                        f"Benders binary state for subproblem "
                        f"'{subproblem_key}', cut "
                        f"{cut_number} is missing binary "
                        f"linking variables: "
                        f"{missing_binary_names}"
                    )

                binary_state = [
                    binary_state_dict[name]
                    for name in binary_names
                ]

                # -------------------------------------------------
                # Validate state dimension.
                # -------------------------------------------------

                if len(binary_state) != len(binary_indices):
                    raise ValueError(
                        f"Benders discrete state for subproblem "
                        f"'{subproblem_key}', cut "
                        f"{cut_number} contains "
                        f"{len(binary_state)} values, but "
                        f"{len(binary_indices)} binary MOSEK "
                        "indices were found."
                    )

                # -------------------------------------------------
                # Add native DJC.
                # -------------------------------------------------

                theta_coeff = state_data.get(
                    "theta_coeff",
                    1.0,
                )
                cut_scale = (
                    BENDERS_THETA_SCALE
                    if abs(float(theta_coeff)) > 1e-12
                    else 1.0
                )

                add_benders_djc(
                    task=task,
                    theta_idx=theta_idx,
                    binary_indices=binary_indices,
                    binary_state=binary_state,
                    continuous_indices=continuous_indices,
                    continuous_point=state_data[
                        "continuous_point"
                    ],
                    op_cost=state_data[
                        "op_cost"
                    ] / cut_scale,
                    continuous_lambda={
                        name: float(value) / cut_scale
                        for name, value in state_data[
                            "continuous_lambda"
                        ].items()
                    },
                    theta_coeff=theta_coeff,
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
    Add one state-specific Benders optimality or feasibility cut as a
    native MOSEK disjunctive constraint. Set ``theta_coeff=0`` for a
    restoration-dual feasibility cut.

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