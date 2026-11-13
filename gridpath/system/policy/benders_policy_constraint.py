"""
This module implements variables and constraints involved in constructing
the policy constraint for Benders to make the allocation of the period
variables across subproblems.
"""

import os.path
import sys
from db.common_functions import connect_to_database
from gridpath.get_scenario_inputs import parse_arguments
from gridpath.auxiliary.scenario_chars import SubScenarios
from gridpath.auxiliary.db_interface import get_scenario_id_and_name

from pyomo.environ import (
    Any,
    Boolean,
    Constraint,
    Param,
    Var,
    NonNegativeReals,
    Set,
    Reals
)

Infinity = float("inf")


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

    # ============================================================
    # Subproblem index only for associated periods
    # ============================================================

    args = sys.argv[1:]

    parsed_arguments = parse_arguments(args=args)

    db_path = parsed_arguments.database

    # Stage is "" at this point for nesting function purposes.
    if stage == "":
        stage_temp = 1
    else:
        stage_temp = stage

    conn = connect_to_database(db_path=db_path)
    try:
        c = conn.cursor()
        scenario_id, _ = get_scenario_id_and_name(
            scenario_id_arg=parsed_arguments.scenario_id,
            scenario_name_arg=parsed_arguments.scenario,
            c=c,
            script="get_scenario_inputs",
        )

        subscenarios = SubScenarios(conn=conn, scenario_id=scenario_id)

        c.execute(
            """
            SELECT DISTINCT period, subproblem_id
            FROM inputs_temporal
            WHERE temporal_scenario_id = ?
              AND stage_id = ?
              AND spinup_or_lookahead = 0
            """,
            (subscenarios.TEMPORAL_SCENARIO_ID, int(stage_temp)),
        )
        rows = c.fetchall()
    finally:
        conn.close()

    subproblems_by_period = {}
    for period, subproblem_id in rows:
        subproblems_by_period.setdefault(int(period), []).append(str(subproblem_id))

    # ============================================================
    # Subproblem Sets
    # ============================================================

    if not hasattr(m, "SUBPROBLEM"):
        if isinstance(subproblem, (list, tuple, set)):
            m.SUBPROBLEM = Set(initialize=list(subproblem))
        else:
            m.SUBPROBLEM = Set(initialize=[str(subproblem)])

    m.SUBPROBLEMS_BY_PERIOD = Set(
        m.PERIODS,
        initialize=lambda mod, period: subproblems_by_period.get(period, []),
    )

    # ============================================================
    # Variables and conditional sets
    # ============================================================

    if hasattr(m, "ENERGY_TARGET_ZONE_PERIODS_WITH_ENERGY_TARGET"):
        m.ENERGY_TARGET_ZONE_SUBPROBLEMS_BY_PERIOD = Set(
            dimen=3,
            initialize=lambda mod: [
                (zone, period, subproblem)
                for zone, period in mod.ENERGY_TARGET_ZONE_PERIODS_WITH_ENERGY_TARGET
                for subproblem in mod.SUBPROBLEMS_BY_PERIOD[period]
            ],
        )
        m.PERIOD_ENERGY_TARGET_ALLOCATION = Var(
            m.ENERGY_TARGET_ZONE_SUBPROBLEMS_BY_PERIOD,
            within=NonNegativeReals,
        )

    if hasattr(m, "PERFORMANCE_STANDARD_ZONE_PERIODS_WITH_PERFORMANCE_STANDARD"):
        m.PERFORMANCE_STANDARD_MWH_ZONE_PERIODS = Set(
            dimen=2,
            initialize=lambda mod: [
                (zone, period)
                for zone, period in mod.PERFORMANCE_STANDARD_ZONE_PERIODS_WITH_PERFORMANCE_STANDARD
                if mod.performance_standard_tco2_per_mwh[zone, period] != float("inf")
            ],
        )
        m.PERFORMANCE_STANDARD_MW_ZONE_PERIODS = Set(
            dimen=2,
            initialize=lambda mod: [
                (zone, period)
                for zone, period in mod.PERFORMANCE_STANDARD_ZONE_PERIODS_WITH_PERFORMANCE_STANDARD
                if mod.performance_standard_tco2_per_mw[zone, period] != float("inf")
            ],
        )
        m.PERFORMANCE_STANDARD_MWH_SUBPROBLEMS_BY_PERIOD = Set(
            dimen=3,
            initialize=lambda mod: [
                (zone, period, subproblem)
                for zone, period in mod.PERFORMANCE_STANDARD_MWH_ZONE_PERIODS
                for subproblem in mod.SUBPROBLEMS_BY_PERIOD[period]
            ],
        )
        m.PERFORMANCE_STANDARD_MW_SUBPROBLEMS_BY_PERIOD = Set(
            dimen=3,
            initialize=lambda mod: [
                (zone, period, subproblem)
                for zone, period in mod.PERFORMANCE_STANDARD_MW_ZONE_PERIODS
                for subproblem in mod.SUBPROBLEMS_BY_PERIOD[period]
            ],
        )
        m.PERFORMANCE_STANDARD_MWH_ALLOCATION = Var(
            m.PERFORMANCE_STANDARD_MWH_SUBPROBLEMS_BY_PERIOD,
        )
        m.PERFORMANCE_STANDARD_MW_ALLOCATION = Var(
            m.PERFORMANCE_STANDARD_MW_SUBPROBLEMS_BY_PERIOD,
        )

    if hasattr(m, "CARBON_CAP_ZONE_PERIODS_WITH_CARBON_CAP"):
        m.CARBON_CAP_ZONE_SUBPROBLEMS_BY_PERIOD = Set(
            dimen=3,
            initialize=lambda mod: [
                (zone, period, subproblem)
                for zone, period in mod.CARBON_CAP_ZONE_PERIODS_WITH_CARBON_CAP
                for subproblem in mod.SUBPROBLEMS_BY_PERIOD[period]
            ],
        )
        m.CARBON_CAP_ALLOCATION = Var(
            m.CARBON_CAP_ZONE_SUBPROBLEMS_BY_PERIOD,
            within=NonNegativeReals,
        )

    if (
        hasattr(m, "CARBON_CREDITS_ZONES")
        and not getattr(m, "_is_benders_subproblem", False)
    ):
        m.benders_carbon_credits_generation_zone = Param(
            m.PROJECTS,
            within=Any,
            default=None,
        )
        m.benders_project_intensity_credit_threshold = Param(
            m.PROJECTS,
            m.PERIODS,
            within=NonNegativeReals,
            default=Infinity,
        )
        m.benders_project_absolute_credit_threshold = Param(
            m.PROJECTS,
            m.PERIODS,
            within=NonNegativeReals,
            default=Infinity,
        )
        m.BENDERS_PROJECT_CARBON_CREDIT_ABSOLUTE_THRESHOLD_PROJECT_PERIODS = Set(
            dimen=2,
            initialize=lambda mod: [
                (prj, period)
                for prj in mod.PROJECTS
                for period in mod.PERIODS
                if mod.benders_project_absolute_credit_threshold[prj, period]
                < Infinity
            ],
        )
        m.BENDERS_PROJECT_CARBON_CREDIT_ABSOLUTE_THRESHOLD_ALLOCATION_INDEX = Set(
            dimen=3,
            initialize=lambda mod: [
                (prj, period, worker)
                for prj, period in mod.BENDERS_PROJECT_CARBON_CREDIT_ABSOLUTE_THRESHOLD_PROJECT_PERIODS
                for worker in mod.SUBPROBLEMS_BY_PERIOD[period]
            ],
        )
        m.BENDERS_PROJECT_CARBON_CREDIT_ABSOLUTE_THRESHOLD_ALLOCATION = Var(
            m.BENDERS_PROJECT_CARBON_CREDIT_ABSOLUTE_THRESHOLD_ALLOCATION_INDEX,
            within=NonNegativeReals,
            bounds=lambda mod, prj, period, worker: (
                0,
                mod.benders_project_absolute_credit_threshold[prj, period],
            ),
        )
        m.benders_allow_carbon_credits_infinite_demand = Param(
            m.CARBON_CREDITS_ZONES, m.PERIODS, within=Boolean, default=0
        )
        m.BENDERS_CARBON_CREDIT_ABSOLUTE_THRESHOLD_COMPLETE_PERIODS = Set(
            initialize=lambda mod: [
                period
                for period in mod.PERIODS
                if all(
                    mod.benders_project_intensity_credit_threshold[prj, period]
                    == Infinity
                    and mod.benders_project_absolute_credit_threshold[prj, period]
                    < Infinity
                    for prj in mod.PROJECTS
                    if mod.benders_carbon_credits_generation_zone[prj]
                    in mod.CARBON_CREDITS_ZONES
                    and (
                        mod.benders_project_intensity_credit_threshold[prj, period]
                        < Infinity
                        or mod.benders_project_absolute_credit_threshold[prj, period]
                        < Infinity
                    )
                )
            ],
        )
        m.BENDERS_CARBON_CREDIT_GENERATED_ABSOLUTE_LIMIT_INDEX = Set(
            dimen=3,
            initialize=lambda mod: [
                (zone, period, worker)
                for zone in mod.CARBON_CREDITS_ZONES
                for period in mod.BENDERS_CARBON_CREDIT_ABSOLUTE_THRESHOLD_COMPLETE_PERIODS
                for worker in mod.SUBPROBLEMS_BY_PERIOD[period]
            ],
        )
        m.benders_carbon_credits_demand_tco2 = Param(
            m.CARBON_CREDITS_ZONES,
            m.PERIODS,
            within=NonNegativeReals,
            default=0,
        )
        m.benders_allow_carbon_credits_infinite_supply = Param(
            m.CARBON_CREDITS_ZONES, m.PERIODS, within=Boolean, default=0
        )
        m.benders_carbon_credits_supply_tco2 = Param(
            m.CARBON_CREDITS_ZONES,
            m.PERIODS,
            within=NonNegativeReals,
            default=0,
        )
        m.BENDERS_CARBON_CREDIT_DEMAND_ZONE_PERIODS = Set(
            dimen=2,
            initialize=lambda mod: [
                (zone, period)
                for zone in mod.CARBON_CREDITS_ZONES
                for period in mod.PERIODS
                if not mod.benders_allow_carbon_credits_infinite_demand[
                    zone, period
                ]
            ],
        )
        m.BENDERS_CARBON_CREDIT_SUPPLY_ZONE_PERIODS = Set(
            dimen=2,
            initialize=lambda mod: [
                (zone, period)
                for zone in mod.CARBON_CREDITS_ZONES
                for period in mod.PERIODS
                if not mod.benders_allow_carbon_credits_infinite_supply[
                    zone, period
                ]
            ],
        )
        m.BENDERS_CARBON_CREDIT_SUBPROBLEMS_BY_PERIOD = Set(
            dimen=3,
            initialize=lambda mod: [
                (zone, period, worker)
                for zone in mod.CARBON_CREDITS_ZONES
                for period in mod.PERIODS
                for worker in mod.SUBPROBLEMS_BY_PERIOD[period]
            ],
        )
        m.BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION_INDEX = Set(
            dimen=3,
            initialize=lambda mod: [
                (zone, period, worker)
                for zone in mod.CARBON_CREDITS_ZONES
                for period in mod.PERIODS
                for worker in mod.SUBPROBLEMS_BY_PERIOD[period]
            ],
        )
        m.BENDERS_CARBON_CREDIT_SUPPLY_ALLOCATION_INDEX = Set(
            dimen=3,
            initialize=lambda mod: [
                (zone, period, worker)
                for zone, period in mod.BENDERS_CARBON_CREDIT_SUPPLY_ZONE_PERIODS
                for worker in mod.SUBPROBLEMS_BY_PERIOD[period]
            ],
        )
        m.BENDERS_CARBON_CREDIT_GENERATED_TOTAL = Var(
            m.BENDERS_CARBON_CREDIT_SUBPROBLEMS_BY_PERIOD,
            within=NonNegativeReals,
        )
        m.BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION = Var(
            m.BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION_INDEX,
            within=NonNegativeReals,
            bounds=lambda mod, zone, period, worker: (0, None)
            if mod.benders_allow_carbon_credits_infinite_demand[zone, period]
            else (0, mod.benders_carbon_credits_demand_tco2[zone, period]),
        )
        m.BENDERS_CARBON_CREDIT_SUPPLY_ALLOCATION = Var(
            m.BENDERS_CARBON_CREDIT_SUPPLY_ALLOCATION_INDEX,
            within=NonNegativeReals,
            bounds=lambda mod, zone, period, worker: (0, None)
            if mod.benders_allow_carbon_credits_infinite_supply[zone, period]
            else (0, mod.benders_carbon_credits_supply_tco2[zone, period]),
        )
        
    CAP_FACTOR_ALLOCATION_DEFINITIONS = (
        (
            "MIN",
            "CAP_FACTOR_MIN_ALLOCATION_TARGETS",
            "min_cap_factor",
        ),
        (
            "MAX",
            "CAP_FACTOR_MAX_ALLOCATION_TARGETS",
            "max_cap_factor",
        ),
    )
    
    # ============================================================
    # Capacity-factor allocation variables (master only)
    # ============================================================

    if not getattr(m, "_is_benders_subproblem", False) and all(
        hasattr(m, target_set_name)
        for _, target_set_name, _ in CAP_FACTOR_ALLOCATION_DEFINITIONS
    ):

        for direction, target_set_name, _ in (
            CAP_FACTOR_ALLOCATION_DEFINITIONS
        ):
            index_name = f"CAP_FACTOR_{direction}_ALLOCATION_INDEX"
            variable_name = f"CAP_FACTOR_{direction}_ALLOCATION"

            setattr(
                m,
                index_name,
                Set(
                    dimen=4,
                    initialize=_make_period_allocation_index_rule(
                        target_set_name
                    ),
                ),
            )

            setattr(
                m,
                variable_name,
                Var(
                    getattr(m, index_name),
                    within=Reals,
                ),
            )
        
    # ============================================================
    # Fuel-burn allocation variables (master only)
    # ============================================================
        
    FUEL_BURN_ALLOCATION_DEFINITIONS = (
        (
            "MIN_ABS",
            "FUEL_BURN_MIN_ABS_ALLOCATION_TARGETS",
            "fuel_burn_min_unit",
        ),
        (
            "MAX_ABS",
            "FUEL_BURN_MAX_ABS_ALLOCATION_TARGETS",
            "fuel_burn_max_unit",
        ),
    )

    if not getattr(m, "_is_benders_subproblem", False) and all(
        hasattr(m, target_set_name)
        for _, target_set_name, _ in FUEL_BURN_ALLOCATION_DEFINITIONS
    ):

        for direction, target_set_name, _ in (
            FUEL_BURN_ALLOCATION_DEFINITIONS
        ):
            target_set = getattr(m, target_set_name)

            index_name = f"FUEL_BURN_{direction}_ALLOCATION_INDEX"
            variable_name = f"FUEL_BURN_{direction}_ALLOCATION"

            setattr(
                m,
                index_name,
                Set(
                    dimen=4,
                    initialize=_make_period_allocation_index_rule(
                        target_set_name
                    ),
                ),
            )

            setattr(
                m,
                variable_name,
                Var(
                    getattr(m, index_name),
                    within=NonNegativeReals,
                ),
            )

    # ============================================================
    # Transmission target allocation variables (master only)
    # ============================================================

    if not getattr(m, "_is_benders_subproblem", False) and all(
        hasattr(m, name)
        for name in [
            "TRANSMISSION_TARGET_POS_DIR_MIN_ALLOCATION_TARGETS",
            "TRANSMISSION_TARGET_POS_DIR_MAX_ALLOCATION_TARGETS",
            "TRANSMISSION_TARGET_NEG_DIR_MIN_ALLOCATION_TARGETS",
            "TRANSMISSION_TARGET_NEG_DIR_MAX_ALLOCATION_TARGETS",
        ]
    ):

        transmission_allocation_definitions = [
            (
                "POS_DIR_MIN",
                "TRANSMISSION_TARGET_POS_DIR_MIN_ALLOCATION_TARGETS",
            ),
            (
                "POS_DIR_MAX",
                "TRANSMISSION_TARGET_POS_DIR_MAX_ALLOCATION_TARGETS",
            ),
            (
                "NEG_DIR_MIN",
                "TRANSMISSION_TARGET_NEG_DIR_MIN_ALLOCATION_TARGETS",
            ),
            (
                "NEG_DIR_MAX",
                "TRANSMISSION_TARGET_NEG_DIR_MAX_ALLOCATION_TARGETS",
            ),
        ]

        for direction, target_set_name in transmission_allocation_definitions:

            target_set = getattr(m, target_set_name)

            index_name = (
                f"TRANSMISSION_TARGET_{direction}_ALLOCATION_INDEX"
            )

            variable_name = (
                f"TRANSMISSION_TARGET_{direction}_ALLOCATION"
            )

            setattr(
                m,
                index_name,
                Set(
                    dimen=4,
                    initialize=lambda mod, target_set=target_set: [
                        (z, bt, hz, str(sp))
                        for z, bt, hz in target_set
                        for sp in mod.SUBPROBLEMS_BY_PERIOD[hz]
                    ],
                ),
            )

            setattr(
                m,
                variable_name,
                Var(
                    getattr(m, index_name),
                    within=NonNegativeReals,
                ),
            )

    # ============================================================
    # Constraint rules
    # ============================================================
    
    def _make_period_allocation_index_rule(target_set_name):
        def allocation_index_rule(mod):
            return [
                (ba, bt, hz, str(sp))
                for ba, bt, hz in getattr(mod, target_set_name)
                for sp in mod.SUBPROBLEMS_BY_PERIOD[hz]
            ]

        return allocation_index_rule
    
    def cap_factor_min_allocation_rule(mod, prj, bt, h):
        return sum(
            mod.CAP_FACTOR_MIN_ALLOCATION[prj, bt, h, str(sp)]
            for sp in mod.SUBPROBLEMS_BY_PERIOD[h]
        ) == (
            mod.min_cap_factor[prj, bt, h]
            * mod.Possible_Power_Provision_in_Horizon_Expression[
                prj, bt, h
            ]
        )

    def cap_factor_max_allocation_rule(mod, prj, bt, h):
        return sum(
            mod.CAP_FACTOR_MAX_ALLOCATION[prj, bt, h, str(sp)]
            for sp in mod.SUBPROBLEMS_BY_PERIOD[h]
        ) == (
            mod.max_cap_factor[prj, bt, h]
            * mod.Possible_Power_Provision_in_Horizon_Expression[
                prj, bt, h
            ]
        )
    
    def fuel_burn_min_abs_allocation_rule(mod, ba, bt, hz):
        return sum(
            mod.FUEL_BURN_MIN_ABS_ALLOCATION[ba, bt, hz, sp]
            for sp in mod.SUBPROBLEMS_BY_PERIOD[hz]
        ) == mod.fuel_burn_min_unit[ba, bt, hz]

    def fuel_burn_max_abs_allocation_rule(mod, ba, bt, hz):
        return sum(
            mod.FUEL_BURN_MAX_ABS_ALLOCATION[ba, bt, hz, sp]
            for sp in mod.SUBPROBLEMS_BY_PERIOD[hz]
        ) == mod.fuel_burn_max_unit[ba, bt, hz]

    def transmission_target_pos_dir_min_allocation_rule(mod, z, bt, hz):
        return sum(
            mod.TRANSMISSION_TARGET_POS_DIR_MIN_ALLOCATION[z, bt, hz, sp]
            for sp in mod.SUBPROBLEMS_BY_PERIOD[hz]
        ) == mod.transmission_target_pos_dir_min_mwh[z, bt, hz]

    def transmission_target_pos_dir_max_allocation_rule(mod, z, bt, hz):
        return sum(
            mod.TRANSMISSION_TARGET_POS_DIR_MAX_ALLOCATION[z, bt, hz, sp]
            for sp in mod.SUBPROBLEMS_BY_PERIOD[hz]
        ) == mod.transmission_target_pos_dir_max_mwh[z, bt, hz]

    def transmission_target_neg_dir_min_allocation_rule(mod, z, bt, hz):
        return sum(
            mod.TRANSMISSION_TARGET_NEG_DIR_MIN_ALLOCATION[z, bt, hz, sp]
            for sp in mod.SUBPROBLEMS_BY_PERIOD[hz]
        ) == mod.transmission_target_neg_dir_min_mwh[z, bt, hz]

    def transmission_target_neg_dir_max_allocation_rule(mod, z, bt, hz):
        return sum(
            mod.TRANSMISSION_TARGET_NEG_DIR_MAX_ALLOCATION[z, bt, hz, sp]
            for sp in mod.SUBPROBLEMS_BY_PERIOD[hz]
        ) == mod.transmission_target_neg_dir_max_mwh[z, bt, hz]

    def carbon_cap_allocation_rule(mod, z, p):
        return (
            sum(
                mod.CARBON_CAP_ALLOCATION[z, p, w]
                for w in mod.SUBPROBLEMS_BY_PERIOD[p]
            )
            == mod.carbon_cap_target[z, p]
        )

    def period_energy_target_allocation_mwh_rule(mod, z, p):
        return (
            sum(
                mod.PERIOD_ENERGY_TARGET_ALLOCATION[z, p, w]
                for w in mod.SUBPROBLEMS_BY_PERIOD[p]
            )
            >= mod.Period_Energy_Target[z, p]
        )

    def period_performance_standard_allocation_mwh_rule(mod, z, p):
        return (
            sum(
                mod.PERFORMANCE_STANDARD_MWH_ALLOCATION[z, p, w]
                for w in mod.SUBPROBLEMS_BY_PERIOD[p]
            )
            == 0
        )

    def period_performance_standard_allocation_mw_rule(mod, z, p):
        return (
            sum(
                mod.PERFORMANCE_STANDARD_MW_ALLOCATION[z, p, w]
                for w in mod.SUBPROBLEMS_BY_PERIOD[p]
            )
            == 0
        )

    def carbon_credit_demand_allocation_rule(mod, z, p):
        return sum(
            mod.BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION[z, p, worker]
            for worker in mod.SUBPROBLEMS_BY_PERIOD[p]
        ) <= mod.benders_carbon_credits_demand_tco2[z, p]

    def carbon_credit_supply_allocation_rule(mod, z, p):
        return sum(
            mod.BENDERS_CARBON_CREDIT_SUPPLY_ALLOCATION[z, p, worker]
            for worker in mod.SUBPROBLEMS_BY_PERIOD[p]
        ) <= mod.benders_carbon_credits_supply_tco2[z, p]

    def carbon_credit_demand_physical_limit_rule(mod, z, p, worker):
        return mod.BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION[z, p, worker] <= (
            mod.BENDERS_CARBON_CREDIT_GENERATED_TOTAL[z, p, worker]
        )

    def carbon_credit_generated_absolute_limit_rule(mod, z, p, worker):
        return mod.BENDERS_CARBON_CREDIT_GENERATED_TOTAL[z, p, worker] <= sum(
            mod.BENDERS_PROJECT_CARBON_CREDIT_ABSOLUTE_THRESHOLD_ALLOCATION[
                prj, p, worker
            ]
            for prj in mod.PROJECTS
            if mod.benders_carbon_credits_generation_zone[prj] == z
            and (prj, p, worker)
            in mod.BENDERS_PROJECT_CARBON_CREDIT_ABSOLUTE_THRESHOLD_ALLOCATION_INDEX
        )

    def project_carbon_credit_absolute_threshold_allocation_rule(
        mod, prj, period
    ):
        return sum(
            mod.BENDERS_PROJECT_CARBON_CREDIT_ABSOLUTE_THRESHOLD_ALLOCATION[
                prj, period, worker
            ]
            for worker in mod.SUBPROBLEMS_BY_PERIOD[period]
        ) == mod.benders_project_absolute_credit_threshold[prj, period]

    # ============================================================
    # Constraints
    # ============================================================

    if not getattr(m, "_is_benders_subproblem", False):
        
        if hasattr(m, "CAP_FACTOR_MIN_ALLOCATION_TARGETS"):
            m.Cap_Factor_Min_Allocation_Constraint = Constraint(
                m.CAP_FACTOR_MIN_ALLOCATION_TARGETS,
                rule=cap_factor_min_allocation_rule,
            )

        if hasattr(m, "CAP_FACTOR_MAX_ALLOCATION_TARGETS"):
            m.Cap_Factor_Max_Allocation_Constraint = Constraint(
                m.CAP_FACTOR_MAX_ALLOCATION_TARGETS,
                rule=cap_factor_max_allocation_rule,
            )
        
        if hasattr(m, "FUEL_BURN_MIN_ABS_ALLOCATION_TARGETS"):
            m.Fuel_Burn_Min_Abs_Allocation_Constraint = Constraint(
                m.FUEL_BURN_MIN_ABS_ALLOCATION_TARGETS,
                rule=fuel_burn_min_abs_allocation_rule,
            )

        if hasattr(m, "FUEL_BURN_MAX_ABS_ALLOCATION_TARGETS"):
            m.Fuel_Burn_Max_Abs_Allocation_Constraint = Constraint(
                m.FUEL_BURN_MAX_ABS_ALLOCATION_TARGETS,
                rule=fuel_burn_max_abs_allocation_rule,
            )

        if hasattr(m, "TRANSMISSION_TARGET_POS_DIR_MIN_ALLOCATION_TARGETS"):
            m.Transmission_Target_Pos_Dir_Min_Allocation_Constraint = Constraint(
                m.TRANSMISSION_TARGET_POS_DIR_MIN_ALLOCATION_TARGETS,
                rule=transmission_target_pos_dir_min_allocation_rule,
            )

        if hasattr(m, "TRANSMISSION_TARGET_POS_DIR_MAX_ALLOCATION_TARGETS"):
            m.Transmission_Target_Pos_Dir_Max_Allocation_Constraint = Constraint(
                m.TRANSMISSION_TARGET_POS_DIR_MAX_ALLOCATION_TARGETS,
                rule=transmission_target_pos_dir_max_allocation_rule,
            )

        if hasattr(m, "TRANSMISSION_TARGET_NEG_DIR_MIN_ALLOCATION_TARGETS"):
            m.Transmission_Target_Neg_Dir_Min_Allocation_Constraint = Constraint(
                m.TRANSMISSION_TARGET_NEG_DIR_MIN_ALLOCATION_TARGETS,
                rule=transmission_target_neg_dir_min_allocation_rule,
            )

        if hasattr(m, "TRANSMISSION_TARGET_NEG_DIR_MAX_ALLOCATION_TARGETS"):
            m.Transmission_Target_Neg_Dir_Max_Allocation_Constraint = Constraint(
                m.TRANSMISSION_TARGET_NEG_DIR_MAX_ALLOCATION_TARGETS,
                rule=transmission_target_neg_dir_max_allocation_rule,
            )

    if hasattr(m, "CARBON_CAP_ZONE_PERIODS_WITH_CARBON_CAP"):
        m.Carbon_Cap_Allocation_Constraint = Constraint(
            m.CARBON_CAP_ZONE_PERIODS_WITH_CARBON_CAP,
            rule=carbon_cap_allocation_rule,
        )

    if hasattr(m, "ENERGY_TARGET_ZONE_PERIODS_WITH_ENERGY_TARGET"):
        m.Period_Energy_Target_MWh_Allocation_Constraint = Constraint(
            m.ENERGY_TARGET_ZONE_PERIODS_WITH_ENERGY_TARGET,
            rule=period_energy_target_allocation_mwh_rule,
        )

    if hasattr(m, "PERFORMANCE_STANDARD_MWH_ZONE_PERIODS"):
        m.Performance_Standard_MWh_Allocation_Constraint = Constraint(
            m.PERFORMANCE_STANDARD_MWH_ZONE_PERIODS,
            rule=period_performance_standard_allocation_mwh_rule,
        )

    if hasattr(m, "BENDERS_CARBON_CREDIT_DEMAND_ZONE_PERIODS"):
        m.Benders_Carbon_Credit_Demand_Allocation_Constraint = Constraint(
            m.BENDERS_CARBON_CREDIT_DEMAND_ZONE_PERIODS,
            rule=carbon_credit_demand_allocation_rule,
        )
    if hasattr(m, "BENDERS_CARBON_CREDIT_SUPPLY_ZONE_PERIODS"):
        m.Benders_Carbon_Credit_Supply_Allocation_Constraint = Constraint(
            m.BENDERS_CARBON_CREDIT_SUPPLY_ZONE_PERIODS,
            rule=carbon_credit_supply_allocation_rule,
        )
    if hasattr(m, "BENDERS_CARBON_CREDIT_SUBPROBLEMS_BY_PERIOD"):
        m.Benders_Carbon_Credit_Demand_Physical_Limit_Constraint = Constraint(
            m.BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION_INDEX,
            rule=carbon_credit_demand_physical_limit_rule,
        )
    if hasattr(m, "BENDERS_CARBON_CREDIT_GENERATED_ABSOLUTE_LIMIT_INDEX"):
        m.Benders_Carbon_Credit_Generated_Absolute_Limit_Constraint = Constraint(
            m.BENDERS_CARBON_CREDIT_GENERATED_ABSOLUTE_LIMIT_INDEX,
            rule=carbon_credit_generated_absolute_limit_rule,
        )
    if hasattr(
        m,
        "BENDERS_PROJECT_CARBON_CREDIT_ABSOLUTE_THRESHOLD_PROJECT_PERIODS",
    ):
        m.Benders_Project_Carbon_Credit_Absolute_Threshold_Allocation_Constraint = (
            Constraint(
                m.BENDERS_PROJECT_CARBON_CREDIT_ABSOLUTE_THRESHOLD_PROJECT_PERIODS,
                rule=project_carbon_credit_absolute_threshold_allocation_rule,
            )
        )
    if hasattr(m, "PERFORMANCE_STANDARD_MW_ZONE_PERIODS"):
        m.Performance_Standard_MW_Allocation_Constraint = Constraint(
            m.PERFORMANCE_STANDARD_MW_ZONE_PERIODS,
            rule=period_performance_standard_allocation_mw_rule,
        )


def load_model_data(
    m,
    d,
    data_portal,
    scenario_directory,
    weather_iteration,
    hydro_iteration,
    availability_iteration,
    subproblem,
    stage,
):
    if hasattr(m, "benders_carbon_credits_demand_tco2"):
        input_file = os.path.join(
            scenario_directory,
            "inputs",
            "carbon_credits_params.tab",
        )
        if os.path.exists(input_file):
            data_portal.load(
                filename=input_file,
                index=m.CARBON_CREDITS_ZONES * m.PERIODS,
                param=(
                    m.benders_allow_carbon_credits_infinite_demand,
                    m.benders_carbon_credits_demand_tco2,
                    m.benders_allow_carbon_credits_infinite_supply,
                    m.benders_carbon_credits_supply_tco2,
                ),
                select=(
                    "carbon_credits_zone",
                    "period",
                    "allow_carbon_credits_infinite_demand",
                    "carbon_credits_demand_tco2",
                    "allow_carbon_credits_infinite_supply",
                    "carbon_credits_supply_tco2",
                ),
            )

    if hasattr(m, "benders_project_absolute_credit_threshold"):
        projects_file = os.path.join(scenario_directory, "inputs", "projects.tab")
        if os.path.exists(projects_file):
            data_portal.load(
                filename=projects_file,
                param=m.benders_carbon_credits_generation_zone,
                select=("project", "carbon_credits_generation_zone"),
            )
        input_file = os.path.join(
            scenario_directory,
            "inputs",
            "project_carbon_credits.tab",
        )
        if os.path.exists(input_file):
            data_portal.load(
                filename=input_file,
                index=m.PROJECTS * m.PERIODS,
                param=m.benders_project_absolute_credit_threshold,
                select=(
                    "project",
                    "period",
                    "absolute_threshold_emissions_toCO2",
                ),
            )
            data_portal.load(
                filename=input_file,
                index=m.PROJECTS * m.PERIODS,
                param=m.benders_project_intensity_credit_threshold,
                select=(
                    "project",
                    "period",
                    "intensity_threshold_emissions_toCO2_per_MWh",
                ),
            )

