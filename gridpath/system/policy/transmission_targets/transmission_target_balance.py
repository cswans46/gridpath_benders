# Copyright 2022 (c) Crown Copyright, GC.
# Modifications Copyright Blue Marble Analytics LLC 2023.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Transmission targets by balancing type, horizon, and line direction.

Annual transmission targets are allocated across Benders subproblems. Daily
(and other non-annual) targets continue to use their original input values.
"""

from pyomo.environ import (
    Constraint,
    Expression,
    NonNegativeReals,
    Set,
    Var,
    value,
)

from gridpath.common_functions import create_results_df, update_results_df
from gridpath.system.policy.transmission_targets import TX_TARGETS_DF


# Each allocation-target set contains only annual horizons (hz in PERIODS)
# for which the corresponding target is active.
_ALLOCATION_TARGET_SET_DEFINITIONS = (
    (
        "POS_DIR_MIN",
        "TRANSMISSION_TARGET_POS_DIR_MIN_ALLOCATION_TARGETS",
        "transmission_target_pos_dir_min_mwh",
        "min",
    ),
    (
        "POS_DIR_MAX",
        "TRANSMISSION_TARGET_POS_DIR_MAX_ALLOCATION_TARGETS",
        "transmission_target_pos_dir_max_mwh",
        "max",
    ),
    (
        "NEG_DIR_MIN",
        "TRANSMISSION_TARGET_NEG_DIR_MIN_ALLOCATION_TARGETS",
        "transmission_target_neg_dir_min_mwh",
        "min",
    ),
    (
        "NEG_DIR_MAX",
        "TRANSMISSION_TARGET_NEG_DIR_MAX_ALLOCATION_TARGETS",
        "transmission_target_neg_dir_max_mwh",
        "max",
    ),
)

def _make_allocation_target_set_rule(
    parameter_name,
    bound_type,
):
    """Create an initializer for an annual transmission-target set."""

    def allocation_target_set_rule(mod):
        parameter = getattr(mod, parameter_name)

        annual_targets = []

        for z, bt, hz in (
            mod.TRANSMISSION_TARGET_ZONE_BLN_TYPE_HRZS_WITH_TRANSMISSION_TARGET
        ):
            # Only allocate annual targets indexed by model periods.
            if bt != "year" or hz not in mod.PERIODS:
                continue

            target_value = float(parameter[z, bt, hz])

            if bound_type == "min":
                is_active = target_value > 0
            else:
                is_active = target_value < float("inf")

            if is_active:
                annual_targets.append((z, bt, hz))

        return annual_targets

    return allocation_target_set_rule


def _add_annual_allocation_target_sets(m):
    """Declare annual target sets shared by the master and subproblems."""

    for (
        direction,
        target_set_name,
        parameter_name,
        bound_type,
    ) in _ALLOCATION_TARGET_SET_DEFINITIONS:

        if hasattr(m, target_set_name):
            continue

        setattr(
            m,
            target_set_name,
            Set(
                dimen=3,
                initialize=_make_allocation_target_set_rule(
                    parameter_name,
                    bound_type,
                ),
            ),
        )
        
def _make_allocation_index_rule(target_set_name):
    def allocation_index_rule(mod):
        return [
            (z, bt, hz, str(sp))
            for z, bt, hz in getattr(mod, target_set_name)
            for sp in mod.SUBPROBLEM
        ]

    return allocation_index_rule

def _add_subproblem_allocation_variables(m, subproblem):
    if not getattr(m, "_is_benders_subproblem", False):
        return

    if not hasattr(m, "SUBPROBLEM"):
        m.SUBPROBLEM = Set(initialize=[str(subproblem)])

    for direction, target_set_name, _, _ in _ALLOCATION_TARGET_SET_DEFINITIONS:
        index_name = f"TRANSMISSION_TARGET_{direction}_ALLOCATION_INDEX"
        variable_name = f"TRANSMISSION_TARGET_{direction}_ALLOCATION"

        if not hasattr(m, index_name):
            setattr(
                m,
                index_name,
                Set(
                    dimen=4,
                    initialize=_make_allocation_index_rule(target_set_name),
                ),
            )

        if not hasattr(m, variable_name):
            setattr(
                m,
                variable_name,
                Var(
                    getattr(m, index_name),
                    within=NonNegativeReals,
                ),
            )


def _target_allocation_or_input(mod, direction, z, bt, hz, input_parameter):
    """Return this subproblem's annual allocation, or the original target.

    On a Benders subproblem, annual active targets use the local allocation
    variable. All other targets (including daily targets) retain their input
    parameter. On the master, the operational balance constraints retain the
    original target; the master allocation-sum constraints are declared in
    policy_constraint.py.
    """
    if not getattr(mod, "_is_benders_subproblem", False):
        return input_parameter[z, bt, hz]

    target_set_name = (
        f"TRANSMISSION_TARGET_{direction}_ALLOCATION_TARGETS"
    )
    target_set = getattr(mod, target_set_name)

    if (z, bt, hz) not in target_set:
        return input_parameter[z, bt, hz]

    allocation_variable = getattr(
        mod, f"TRANSMISSION_TARGET_{direction}_ALLOCATION"
    )

    return sum(
        allocation_variable[z, bt, hz, str(sp)]
        for sp in mod.SUBPROBLEM
    )


def add_model_components(
    m,
    d,
    scenario_directory,
    weather_iteration,
    hydro_iteration,
    availability_iteration,
    subproblem,
    stage,
):
    """Add transmission-target balance variables and constraints."""

    # These sets must be declared before policy_constraint.py attempts to
    # retrieve them to create the master allocation variables.
    _add_annual_allocation_target_sets(m)

    # Violation variables.
    m.Transmission_Target_Shortage_Pos_Dir_Min_MWh = Var(
        m.TRANSMISSION_TARGET_ZONE_BLN_TYPE_HRZS_WITH_TRANSMISSION_TARGET,
        within=NonNegativeReals,
    )

    m.Transmission_Target_Overage_Pos_Dir_Max_MWh = Var(
        m.TRANSMISSION_TARGET_ZONE_BLN_TYPE_HRZS_WITH_TRANSMISSION_TARGET,
        within=NonNegativeReals,
    )

    m.Transmission_Target_Shortage_Neg_Dir_Min_MWh = Var(
        m.TRANSMISSION_TARGET_ZONE_BLN_TYPE_HRZS_WITH_TRANSMISSION_TARGET,
        within=NonNegativeReals,
    )

    m.Transmission_Target_Overage_Neg_Dir_Max_MWh = Var(
        m.TRANSMISSION_TARGET_ZONE_BLN_TYPE_HRZS_WITH_TRANSMISSION_TARGET,
        within=NonNegativeReals,
    )

    def violation_pos_dir_min_expression_rule(mod, z, bt, hz):
        if mod.transmission_target_allow_violation[z]:
            return mod.Transmission_Target_Shortage_Pos_Dir_Min_MWh[z, bt, hz]
        else:
            return 0

    m.Transmission_Target_Shortage_Pos_Dir_Min_MWh_Expression = Expression(
        m.TRANSMISSION_TARGET_ZONE_BLN_TYPE_HRZS_WITH_TRANSMISSION_TARGET,
        rule=violation_pos_dir_min_expression_rule,
    )

    def violation_pos_dir_max_expression_rule(mod, z, bt, hz):
        if mod.transmission_target_allow_violation[z]:
            return mod.Transmission_Target_Overage_Pos_Dir_Max_MWh[z, bt, hz]
        else:
            return 0

    m.Transmission_Target_Overage_Pos_Dir_Max_MWh_Expression = Expression(
        m.TRANSMISSION_TARGET_ZONE_BLN_TYPE_HRZS_WITH_TRANSMISSION_TARGET,
        rule=violation_pos_dir_max_expression_rule,
    )

    def violation_neg_dir_min_expression_rule(mod, z, bt, hz):
        if mod.transmission_target_allow_violation[z]:
            return mod.Transmission_Target_Shortage_Neg_Dir_Min_MWh[z, bt, hz]
        else:
            return 0

    m.Transmission_Target_Shortage_Neg_Dir_Min_MWh_Expression = Expression(
        m.TRANSMISSION_TARGET_ZONE_BLN_TYPE_HRZS_WITH_TRANSMISSION_TARGET,
        rule=violation_neg_dir_min_expression_rule,
    )

    def violation_neg_dir_max_expression_rule(mod, z, bt, hz):
        if mod.transmission_target_allow_violation[z]:
            return mod.Transmission_Target_Overage_Neg_Dir_Max_MWh[z, bt, hz]
        else:
            return 0

    m.Transmission_Target_Overage_Neg_Dir_Max_MWh_Expression = Expression(
        m.TRANSMISSION_TARGET_ZONE_BLN_TYPE_HRZS_WITH_TRANSMISSION_TARGET,
        rule=violation_neg_dir_max_expression_rule,
    )

    _add_subproblem_allocation_variables(m, subproblem)

    # Positive-direction minimum.
    def transmission_target_pos_dir_min_rule(mod, z, bt, hz):
        target = mod.transmission_target_pos_dir_min_mwh[z, bt, hz]
        if float(target) == 0:
            return Constraint.Skip

        allocation = _target_allocation_or_input(
            mod, "POS_DIR_MIN", z, bt, hz,
            mod.transmission_target_pos_dir_min_mwh,
        )

        return (
            mod.Total_Transmission_Target_Energy_Pos_Dir_MWh[z, bt, hz]
            + mod.Transmission_Target_Shortage_Pos_Dir_Min_MWh_Expression[
                z, bt, hz
            ]
            >= allocation
        )

    m.Transmission_Target_Pos_Dir_Min_Constraint = Constraint(
        m.TRANSMISSION_TARGET_ZONE_BLN_TYPE_HRZS_WITH_TRANSMISSION_TARGET,
        rule=transmission_target_pos_dir_min_rule,
    )

    # Positive-direction maximum.
    def transmission_target_pos_dir_max_rule(mod, z, bt, hz):
        target = mod.transmission_target_pos_dir_max_mwh[z, bt, hz]
        if float(target) == float("inf"):
            return Constraint.Skip

        allocation = _target_allocation_or_input(
            mod, "POS_DIR_MAX", z, bt, hz,
            mod.transmission_target_pos_dir_max_mwh,
        )

        return (
            mod.Total_Transmission_Target_Energy_Pos_Dir_MWh[z, bt, hz]
            - mod.Transmission_Target_Overage_Pos_Dir_Max_MWh_Expression[
                z, bt, hz
            ]
            <= allocation
        )

    m.Transmission_Target_Pos_Dir_Max_Constraint = Constraint(
        m.TRANSMISSION_TARGET_ZONE_BLN_TYPE_HRZS_WITH_TRANSMISSION_TARGET,
        rule=transmission_target_pos_dir_max_rule,
    )

    # Negative-direction minimum.
    def transmission_target_neg_dir_min_rule(mod, z, bt, hz):
        target = mod.transmission_target_neg_dir_min_mwh[z, bt, hz]
        if float(target) == 0:
            return Constraint.Skip

        allocation = _target_allocation_or_input(
            mod, "NEG_DIR_MIN", z, bt, hz,
            mod.transmission_target_neg_dir_min_mwh,
        )

        return (
            mod.Total_Transmission_Target_Energy_Neg_Dir_MWh[z, bt, hz]
            + mod.Transmission_Target_Shortage_Neg_Dir_Min_MWh_Expression[
                z, bt, hz
            ]
            >= allocation
        )

    m.Transmission_Target_Neg_Dir_Min_Constraint = Constraint(
        m.TRANSMISSION_TARGET_ZONE_BLN_TYPE_HRZS_WITH_TRANSMISSION_TARGET,
        rule=transmission_target_neg_dir_min_rule,
    )

    # Negative-direction maximum.
    def transmission_target_neg_dir_max_rule(mod, z, bt, hz):
        target = mod.transmission_target_neg_dir_max_mwh[z, bt, hz]
        if float(target) == float("inf"):
            return Constraint.Skip

        allocation = _target_allocation_or_input(
            mod, "NEG_DIR_MAX", z, bt, hz,
            mod.transmission_target_neg_dir_max_mwh,
        )

        return (
            mod.Total_Transmission_Target_Energy_Neg_Dir_MWh[z, bt, hz]
            - mod.Transmission_Target_Overage_Neg_Dir_Max_MWh_Expression[
                z, bt, hz
            ]
            <= allocation
        )

    m.Transmission_Target_Neg_Dir_Max_Constraint = Constraint(
        m.TRANSMISSION_TARGET_ZONE_BLN_TYPE_HRZS_WITH_TRANSMISSION_TARGET,
        rule=transmission_target_neg_dir_max_rule,
    )


def _get_reporting_target(m, direction, z, bt, hz, input_parameter):
    """Get the target denominator used for subproblem result reporting."""
    target = float(input_parameter[z, bt, hz])

    if not getattr(m, "_is_benders_subproblem", False):
        return target

    target_set_name = (
        f"TRANSMISSION_TARGET_{direction}_ALLOCATION_TARGETS"
    )
    if (z, bt, hz) not in getattr(m, target_set_name):
        return target

    allocation_variable = getattr(
        m, f"TRANSMISSION_TARGET_{direction}_ALLOCATION"
    )
    return sum(
        value(allocation_variable[z, bt, hz, str(sp)])
        for sp in m.SUBPROBLEM
    )


def export_results(
    scenario_directory,
    weather_iteration,
    hydro_iteration,
    availability_iteration,
    subproblem,
    stage,
    m,
    d,
):
    """Export transmission-target performance and violation results."""

    results_columns = [
        "fraction_of_transmission_target_pos_dir_min_met",
        "transmission_target_shortage_pos_dir_min_mwh",
        "fraction_of_transmission_target_pos_dir_max_met",
        "transmission_target_overage_pos_dir_max_mwh",
        "fraction_of_transmission_target_neg_dir_min_met",
        "transmission_target_shortage_neg_dir_min_mwh",
        "fraction_of_transmission_target_neg_dir_max_met",
        "transmission_target_overage_neg_dir_min_mwh",
    ]

    data = []

    for z, bt, hz in (
        m.TRANSMISSION_TARGET_ZONE_BLN_TYPE_HRZS_WITH_TRANSMISSION_TARGET
    ):
        pos_min_target = _get_reporting_target(
            m, "POS_DIR_MIN", z, bt, hz,
            m.transmission_target_pos_dir_min_mwh,
        )
        pos_max_target = _get_reporting_target(
            m, "POS_DIR_MAX", z, bt, hz,
            m.transmission_target_pos_dir_max_mwh,
        )
        neg_min_target = _get_reporting_target(
            m, "NEG_DIR_MIN", z, bt, hz,
            m.transmission_target_neg_dir_min_mwh,
        )
        neg_max_target = _get_reporting_target(
            m, "NEG_DIR_MAX", z, bt, hz,
            m.transmission_target_neg_dir_max_mwh,
        )

        pos_energy = value(
            m.Total_Transmission_Target_Energy_Pos_Dir_MWh[z, bt, hz]
        )
        neg_energy = value(
            m.Total_Transmission_Target_Energy_Neg_Dir_MWh[z, bt, hz]
        )

        data.append(
            [
                z,
                bt,
                hz,
                (
                    None
                    if pos_min_target == 0
                    else pos_energy / pos_min_target
                ),
                value(
                    m.Transmission_Target_Shortage_Pos_Dir_Min_MWh_Expression[
                        z, bt, hz
                    ]
                ),
                (
                    None
                    if pos_max_target == float("inf") or pos_max_target == 0
                    else pos_energy / pos_max_target
                ),
                value(
                    m.Transmission_Target_Overage_Pos_Dir_Max_MWh_Expression[
                        z, bt, hz
                    ]
                ),
                (
                    None
                    if neg_min_target == 0
                    else neg_energy / neg_min_target
                ),
                value(
                    m.Transmission_Target_Shortage_Neg_Dir_Min_MWh_Expression[
                        z, bt, hz
                    ]
                ),
                (
                    None
                    if neg_max_target == float("inf") or neg_max_target == 0
                    else neg_energy / neg_max_target
                ),
                value(
                    m.Transmission_Target_Overage_Neg_Dir_Max_MWh_Expression[
                        z, bt, hz
                    ]
                ),
            ]
        )

    results_df = create_results_df(
        _index_columns_=["transmission_target_zone", "balancing_type", "horizon"],
        _results_columns_=results_columns,
        _data_=data,
    )

    update_results_df(getattr(d, TX_TARGETS_DF), results_df)
