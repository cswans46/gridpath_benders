# Copyright 2016-2023 Blue Marble Analytics LLC.
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
Sell carbon credits to other sectors and buy credits from other sectors.
"""

import csv
import os.path
from pyomo.environ import (
    Boolean,
    Constraint,
    NonNegativeReals,
    Param,
    Set,
    Var,
    value,
)

from gridpath.auxiliary.auxiliary import cursor_to_df
from gridpath.auxiliary.validations import (
    write_validation_to_database,
    validate_row_monotonicity,
    validate_column_monotonicity,
)
from gridpath.common_functions import create_results_df
from gridpath.auxiliary.dynamic_components import (
    carbon_credits_balance_purchase_components,
    carbon_credits_balance_generation_components,
)
from gridpath.system.policy.carbon_credits import CARBON_CREDITS_ZONE_PRD_DF


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
    """

    :param m:
    :param d:
    :return:
    """

    m.allow_carbon_credits_infinite_demand = Param(
        m.CARBON_CREDITS_ZONES, m.PERIODS, within=Boolean, default=0
    )

    m.carbon_credits_demand_tco2 = Param(
        m.CARBON_CREDITS_ZONES, m.PERIODS, within=NonNegativeReals, default=0
    )

    m.carbon_credits_demand_price = Param(
        m.CARBON_CREDITS_ZONES, m.PERIODS, within=NonNegativeReals, default=0
    )

    m.allow_carbon_credits_infinite_supply = Param(
        m.CARBON_CREDITS_ZONES, m.PERIODS, within=Boolean, default=0
    )

    m.carbon_credits_supply_tco2 = Param(
        m.CARBON_CREDITS_ZONES, m.PERIODS, within=NonNegativeReals, default=0
    )

    m.carbon_credits_supply_price = Param(
        m.CARBON_CREDITS_ZONES, m.PERIODS, within=NonNegativeReals, default=0
    )

    m.Sell_Carbon_Credits = Var(
        m.CARBON_CREDITS_ZONES, m.PERIODS, within=NonNegativeReals, initialize=0
    )

    m.Buy_Carbon_Credits = Var(
        m.CARBON_CREDITS_ZONES, m.PERIODS, within=NonNegativeReals, initialize=0
    )

    if getattr(m, "_is_benders_subproblem", False):
        m.BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION_INDEX = Set(
            dimen=3,
            initialize=lambda mod: [
                (z, prd, str(subproblem))
                for z in mod.CARBON_CREDITS_ZONES
                for prd in mod.PERIODS
            ],
        )
        m.BENDERS_CARBON_CREDIT_SUPPLY_ALLOCATION_INDEX = Set(
            dimen=3,
            initialize=lambda mod: [
                (z, prd, str(subproblem))
                for z in mod.CARBON_CREDITS_ZONES
                for prd in mod.PERIODS
                if not mod.allow_carbon_credits_infinite_supply[z, prd]
            ],
        )
        m.BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION = Var(
            m.BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION_INDEX,
            within=NonNegativeReals,
            bounds=lambda mod, z, prd, worker: (0, None)
            if mod.allow_carbon_credits_infinite_demand[z, prd]
            else (0, mod.carbon_credits_demand_tco2[z, prd]),
        )
        m.BENDERS_CARBON_CREDIT_SUPPLY_ALLOCATION = Var(
            m.BENDERS_CARBON_CREDIT_SUPPLY_ALLOCATION_INDEX,
            within=NonNegativeReals,
            bounds=lambda mod, z, prd, worker: (0, None)
            if mod.allow_carbon_credits_infinite_supply[z, prd]
            else (0, mod.carbon_credits_supply_tco2[z, prd]),
        )
        m.BENDERS_CARBON_CREDIT_GENERATED_TOTAL = Var(
            m.BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION_INDEX,
            within=NonNegativeReals,
        )
        def generated_credit_total_rule(mod, z, prd):
            if not hasattr(mod, "Project_Carbon_Credits_Generated"):
                project_credits = 0
            else:
                project_credits = sum(
                    mod.Project_Carbon_Credits_Generated[prj, prd]
                    for prj in mod.CARBON_CREDITS_GENERATION_PRJS_BY_CARBON_CREDITS_ZONE[z]
                    if (prj, prd) in mod.CARBON_CREDITS_GENERATION_PRJ_OPR_PRDS
                )
            return mod.BENDERS_CARBON_CREDIT_GENERATED_TOTAL[
                z, prd, str(subproblem)
            ] == project_credits

        m.Benders_Carbon_Credit_Generated_Total_Constraint = Constraint(
            m.CARBON_CREDITS_ZONES,
            m.PERIODS,
            rule=generated_credit_total_rule,
        )

        m.Benders_Carbon_Credit_Demand_Physical_Limit_Constraint = Constraint(
            m.BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION_INDEX,
            rule=lambda mod, z, prd, worker: mod.BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION[
                z, prd, worker
            ]
            <= mod.BENDERS_CARBON_CREDIT_GENERATED_TOTAL[z, prd, worker],
        )

    def max_sell_carbon_credits_rule(mod, z, prd):
        if getattr(mod, "_is_benders_subproblem", False):
            return mod.Sell_Carbon_Credits[z, prd] <= (
                mod.BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION[z, prd, str(subproblem)]
            )
        elif mod.allow_carbon_credits_infinite_demand[z, prd]:
            return Constraint.Skip
        else:
            return (
                mod.Sell_Carbon_Credits[z, prd]
                <= mod.carbon_credits_demand_tco2[z, prd]
            )

    m.Max_Sell_Carbon_Credits_Constraint = Constraint(
        m.CARBON_CREDITS_ZONES, m.PERIODS, rule=max_sell_carbon_credits_rule
    )

    def max_buy_carbon_credits_rule(mod, z, prd):
        if getattr(mod, "_is_benders_subproblem", False):
            if mod.allow_carbon_credits_infinite_supply[z, prd]:
                if hasattr(mod, "Total_Project_Carbon_Credits_Purchased"):
                    return mod.Buy_Carbon_Credits[z, prd] <= (
                        mod.Total_Project_Carbon_Credits_Purchased[z, prd]
                    )
                return mod.Buy_Carbon_Credits[z, prd] <= 0
            return mod.Buy_Carbon_Credits[z, prd] <= (
                mod.BENDERS_CARBON_CREDIT_SUPPLY_ALLOCATION[z, prd, str(subproblem)]
            )
        elif mod.allow_carbon_credits_infinite_supply[z, prd]:
            return Constraint.Skip
        else:
            return (
                mod.Buy_Carbon_Credits[z, prd] <= mod.carbon_credits_supply_tco2[z, prd]
            )

    m.Max_Buy_Carbon_Credits_Constraint = Constraint(
        m.CARBON_CREDITS_ZONES, m.PERIODS, rule=max_buy_carbon_credits_rule
    )

    # Add to the carbon credits tracking balance
    record_dynamic_components(d)


def record_dynamic_components(dynamic_components):
    """
    :param dynamic_components:
    """
    getattr(dynamic_components, carbon_credits_balance_purchase_components).append(
        "Sell_Carbon_Credits"
    )

    getattr(dynamic_components, carbon_credits_balance_generation_components).append(
        "Buy_Carbon_Credits"
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
    """

    :param m:
    :param d:
    :param data_portal:
    :param scenario_directory:
    :param subproblem:
    :param stage:
    :return:
    """
    input_file = os.path.join(
        scenario_directory,
        weather_iteration,
        hydro_iteration,
        availability_iteration,
        str(subproblem),
        str(stage),
        "inputs",
        "carbon_credits_params.tab",
    )
    if os.path.exists(input_file):
        data_portal.load(
            filename=input_file,
            param=(
                m.allow_carbon_credits_infinite_demand,
                m.carbon_credits_demand_tco2,
                m.carbon_credits_demand_price,
                m.allow_carbon_credits_infinite_supply,
                m.carbon_credits_supply_tco2,
                m.carbon_credits_supply_price,
            ),
        )


def get_inputs_from_database(
    scenario_id,
    subscenarios,
    weather_iteration,
    hydro_iteration,
    availability_iteration,
    subproblem,
    stage,
    conn,
):
    """
    :param subscenarios: SubScenarios object with all subscenario info
    :param subproblem:
    :param stage:
    :param conn: database connection
    :return:
    """
    subproblem = 1 if subproblem == "" else subproblem
    stage = 1 if stage == "" else stage
    c = conn.cursor()
    zones = c.execute(
        f"""SELECT carbon_credits_zone, period, allow_carbon_credits_infinite_demand, carbon_credits_demand_tco2,
                carbon_credits_demand_price, allow_carbon_credits_infinite_supply, carbon_credits_supply_tco2, 
                carbon_credits_supply_price
            FROM
            (SELECT carbon_credits_zone
                FROM inputs_geography_carbon_credits_zones
                WHERE carbon_credits_zone_scenario_id = {subscenarios.CARBON_CREDITS_ZONE_SCENARIO_ID}
            ) as cc_zone_tbl
            CROSS JOIN
            (SELECT period
                FROM inputs_temporal_periods
                WHERE temporal_scenario_id = {subscenarios.TEMPORAL_SCENARIO_ID}
            ) as period_tbl
            LEFT OUTER JOIN
            inputs_system_carbon_credits_params
            USING (carbon_credits_zone, period)
            WHERE carbon_credits_params_scenario_id = {subscenarios.CARBON_CREDITS_PARAMS_SCENARIO_ID};
            """
    )

    return zones


def validate_inputs(
    scenario_id,
    subscenarios,
    subproblem,
    weather_iteration,
    hydro_iteration,
    availability_iteration,
    stage,
    conn,
):
    """
    Get inputs from database and validate the inputs
    :param subscenarios: SubScenarios object with all subscenario info
    :param subproblem:
    :param stage:
    :param conn: database connection
    :return:
    """
    carbon_credits_params = get_inputs_from_database(
        scenario_id,
        subscenarios,
        weather_iteration,
        hydro_iteration,
        availability_iteration,
        subproblem,
        stage,
        conn,
    )

    # Convert input data into pandas DataFrame
    carbon_credits_params_df = cursor_to_df(carbon_credits_params)
    df_cols = carbon_credits_params_df.columns

    # Filter only for carbon_credit_zone-period combinations that infinite supply and demand is allowed
    carbon_credits_params_df = carbon_credits_params_df[
        (carbon_credits_params_df["allow_carbon_credits_infinite_demand"] == 1)
        & (carbon_credits_params_df["allow_carbon_credits_infinite_supply"] == 1)
    ]

    cols = [
        "carbon_credits_demand_price",
        "carbon_credits_supply_price",
    ]

    # check that min build <= max build
    if set(cols).issubset(set(df_cols)):
        write_validation_to_database(
            conn=conn,
            scenario_id=scenario_id,
            weather_iteration=weather_iteration,
            hydro_iteration=hydro_iteration,
            availability_iteration=availability_iteration,
            subproblem_id=subproblem,
            stage_id=stage,
            gridpath_module=__name__,
            db_table="inputs_system_carbon_credits_params",
            severity="High",
            errors=validate_column_monotonicity(
                df=carbon_credits_params_df,
                cols=cols,
                idx_col=["carbon_credits_zone", "period"],
            ),
        )


def write_model_inputs(
    scenario_directory,
    scenario_id,
    subscenarios,
    weather_iteration,
    hydro_iteration,
    availability_iteration,
    subproblem,
    stage,
    conn,
):
    """
    Get inputs from database and write out the model input
    carbon_tax_zones.tab file.
    :param scenario_directory: string, the scenario directory
    :param subscenarios: SubScenarios object with all subscenario info
    :param subproblem:
    :param stage:
    :param conn: database connection
    :return:
    """

    query_results = get_inputs_from_database(
        scenario_id,
        subscenarios,
        weather_iteration,
        hydro_iteration,
        availability_iteration,
        subproblem,
        stage,
        conn,
    )

    if subproblem != "":
        # Gather vintage, project, lifetime map for filtering
        from gridpath.temporal.investment.periods import (
            get_inputs_from_database as get_periods_from_database
        )

        if stage == "":
            stage_temp = 1
        else:
            stage_temp = stage
        # Fetch this subproblem's associated period.
        periods_for_filtering = get_periods_from_database(
            scenario_id,
            subscenarios,
            weather_iteration,
            hydro_iteration,
            availability_iteration,
            subproblem,
            stage_temp,
            conn,
        ).fetchall()

        # In case there are several periods (this should never happen)
        periods = {int(row[0]) for row in periods_for_filtering}

        carbon_credit_param_rows = query_results.fetchall()

        included_rows = [
            row
            for row in carbon_credit_param_rows
            if int(row[1]) in periods
        ]

        carbon_credit_param_rows = included_rows
        # Create a separate filtered projects class to avoid consuming the projects component before later
        # project.tab is actually written.
        class FilteredCarbonCreditParam:
            def __init__(self, rows):
                self.description = query_results.description
                self.rows = rows

            def fetchall(self):
                return self.rows

            def __iter__(self):
                return iter(self.rows)

        query_results = FilteredCarbonCreditParam(carbon_credit_param_rows)

    # carbon_tax_zones_carbon_credits_zone_mapping.tab
    df = cursor_to_df(query_results)
    df = df.fillna(".")
    fpath = os.path.join(
        scenario_directory,
        str(subproblem),
        str(stage),
        "inputs",
        "carbon_credits_params.tab",
    )
    df.to_csv(fpath, index=False, sep="\t")


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
    """

    :param scenario_directory:
    :param subproblem:
    :param stage:
    :param m:
    :param d:
    :return:
    """
    results_columns = [
        "buy_credits",
        "sell_credits",
    ]
    data = [
        [z, p, value(m.Buy_Carbon_Credits[z, p]), value(m.Sell_Carbon_Credits[z, p])]
        for z in m.CARBON_CREDITS_ZONES
        for p in m.PERIODS
    ]
    results_df = create_results_df(
        index_columns=["carbon_credits_zone", "period"],
        results_columns=results_columns,
        data=data,
    )

    for c in results_columns:
        getattr(d, CARBON_CREDITS_ZONE_PRD_DF)[c] = None
    getattr(d, CARBON_CREDITS_ZONE_PRD_DF).update(results_df)
