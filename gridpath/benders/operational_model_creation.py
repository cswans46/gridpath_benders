from pyomo.environ import AbstractModel
from pyomo.core.expr.visitor import identify_variables,identify_components
import logging
from pyomo.environ import Expression, Param,Set,Var
from pyomo.core.expr.visitor import identify_variables, identify_components
import os
from pyomo.common.collections import ComponentSet
import csv

logger = logging.getLogger(__name__)

def define_linking_vars(m):

    investment_exprs = [
        "Capacity_MW",
        "Tx_Min_Capacity_MW",
        "Tx_Max_Capacity_MW",
        "TxNewLin_Capacity_MW",
        "Energy_Storage_Capacity_MWh",
    ]

    budgeted_policy_components = (
        "CARBON_CAP_ALLOCATION",
        "PERIOD_ENERGY_TARGET_ALLOCATION",
        "PERFORMANCE_STANDARD_MWH_ALLOCATION",
        "PERFORMANCE_STANDARD_MW_ALLOCATION",
        "BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION",
        "BENDERS_CARBON_CREDIT_SUPPLY_ALLOCATION",
        "BENDERS_CARBON_CREDIT_GENERATED_TOTAL",
        "BENDERS_PROJECT_CARBON_CREDIT_ABSOLUTE_THRESHOLD_ALLOCATION",
        "TRANSMISSION_TARGET_POS_DIR_MIN_ALLOCATION",
        "TRANSMISSION_TARGET_POS_DIR_MAX_ALLOCATION",
        "TRANSMISSION_TARGET_NEG_DIR_MIN_ALLOCATION",
        "TRANSMISSION_TARGET_NEG_DIR_MAX_ALLOCATION",
        "FUEL_BURN_MIN_ABS_ALLOCATION",
        "FUEL_BURN_MAX_ABS_ALLOCATION",
        "CAP_FACTOR_MIN_ALLOCATION",
        "CAP_FACTOR_MAX_ALLOCATION",
    )
    
    investment_vars = [
        component_name
        for component_name in budgeted_policy_components
        if hasattr(m, component_name)
    ]

    linking_vars = set()

    # Variables identified through investment expressions
    for expr_name in investment_exprs:

        expr_component = getattr(m, expr_name)

        if expr_component.is_indexed():
            expressions = (
                expr_component[i].expr
                for i in expr_component
            )
        else:
            expressions = [expr_component.expr]

        for expr in expressions:

            for var in identify_variables(expr):
                linking_vars.add(var.name)

    # Explicitly add all VarData objects belonging to
    # policy allocation variables.
    for var_name in investment_vars:

        var_component = getattr(m, var_name)

        if var_component.is_indexed():

            for var in var_component.values():
                linking_vars.add(var.name)

        else:
            linking_vars.add(var_component.name)

    return sorted(linking_vars)

def create_operational_problem_benders(
        scenario_directory,
        key,
        dynamic_components,
        multi_stage,
        parsed_arguments,
):
    from gridpath.run_scenario import (
        set_up_gridpath_modules,
        create_abstract_model,
        load_scenario_data,
        create_problem_instance,
        fix_variables
    )

    """
    :param scenario_directory: the main scenario directory
    :param subproblem: the horizon subproblem name
    :param stage: the stage subproblem name
    :param parsed_arguments: the user-defined script arguments
    :return: modules_to_use (list of module names used in scenario),
        loaded_modules (Python objects), dynamic_inputs (the populated
        dynamic components class), instance (the problem instance), results
        (the optimization results)

    This method creates the problem instance for each Benders subproblem through multi-
    processing.

    """

    # No chan   ge in operational model structure between subproblems.
    # Therefore, use wholistic design.
    model = AbstractModel()

    # Here, we need to identify the subproblems as being operational subproblems
    # and not the master problem since the master problem reuses infrastructure for
    # generating the abstractmodel.
    model._is_benders_subproblem = True

    modules_to_use, loaded_modules = set_up_gridpath_modules(
        scenario_directory=scenario_directory,
        multi_stage=multi_stage,
        parsed_arguments=parsed_arguments
    )

    # if not hasattr(model, "CARBON_CAP_ZONE_PERIODS_WITH_CARBON_CAP"):
    #     model.CARBON_CAP_ZONE_PERIODS_WITH_CARBON_CAP = Set(
    #         dimen=2,
    #         initialize=_get_carbon_cap_zone_periods(
    #             scenario_directory,
    #             key.weather,
    #             key.hydro,
    #             key.availability,
    #             key.subproblem,
    #             key.stage,
    #         ),
    #     )

    create_abstract_model(
        model,
        dynamic_components,
        loaded_modules["Operational"],
        scenario_directory,
        key.weather,
        key.hydro,
        key.availability,
        key.subproblem,
        key.stage,
        temporal_subproblems=None
    )

    scenario_data_benders = load_scenario_data(
        model,
        dynamic_components,
        loaded_modules["Operational"],
        scenario_directory,
        key.weather,
        key.hydro,
        key.availability,
        # Change this to be only one subproblem and loop over it to build a dictionary of scenario_data.
        key.subproblem,
        key.stage,
    )

    instance = create_problem_instance(model, scenario_data_benders)

    # Fix variables if modules request so
    instance = fix_variables(
        instance,
        dynamic_components,
        scenario_directory,
        key.weather,
        key.hydro,
        key.availability,
        key.subproblem,
        key.stage,
        loaded_modules["Operational"],
    )

    return instance

def _get_carbon_cap_zone_periods(
        scenario_directory,
        weather_iteration,
        hydro_iteration,
        availability_iteration,
        subproblem,
        stage,
):
    filename = os.path.join(
        scenario_directory,
        weather_iteration,
        hydro_iteration,
        availability_iteration,
        str(subproblem),
        stage,
        "inputs",
        "carbon_cap.tab",
    )

    def initialize(model):
        with open(filename, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                yield (
                    row["carbon_cap_zone"],
                    int(row["period"]),
                )

    return initialize