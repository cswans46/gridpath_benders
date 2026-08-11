from pyomo.environ import AbstractModel
from pyomo.core.expr.visitor import identify_variables
import logging

logger = logging.getLogger(__name__)

def define_linking_vars(m):
    investment_vars = [
        "Capacity_MW"
    ]

    linking_vars = set()

    for expr_name in investment_vars:
        expr_component = getattr(m, expr_name)

        # Handle scalar Expression
        if expr_component.is_indexed() is False:
            expressions = [expr_component.expr]

        # Handle IndexedExpression
        else:
            expressions = [
                expr_component[i].expr
                for i in expr_component
            ]

        for expr in expressions:
            for var in identify_variables(expr):
                linking_vars.add(var.name)

    return list(linking_vars)

def create_operational_problem_benders(
        scenario_directory,
        key,
        dynamic_components,
        expect_feasible,
        multi_stage,
        parsed_arguments,
):
    from gridpath.run_scenario import (
        set_up_gridpath_modules,
        create_abstract_model,
        add_slacks_to_subproblem,
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

    if not parsed_arguments.quiet:
        print("Building operational model...")

    modules_to_use, loaded_modules = set_up_gridpath_modules(
        scenario_directory=scenario_directory,
        multi_stage=multi_stage,
        parsed_arguments=parsed_arguments
    )

    create_abstract_model(
        # Also have to create two separate models (operational and investment) to facilitate benders
        # Because it implies a change to the optimisation structure.
        model,
        dynamic_components,
        # This is why need to make loaded_modules a dictionary.
        # This is to create the model from the loaded modules from module_list.py and model_organisation.py.
        loaded_modules["Operational"],
        scenario_directory,
        key.weather,
        key.hydro,
        key.availability,
        key.subproblem,
        key.stage
    )

    if expect_feasible:
        print("Skipping feasibility cuts.")
        logger.info("Skipping feasibility cuts.")
    else:
        # We can add slacks to subproblem here because we haven't instantiated the model yet.
        add_slacks_to_subproblem(model)

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

    # print("Scenario data: ",scenario_data.data())
    if not parsed_arguments.quiet:
        print("Creating problem instance for operational subproblems...")
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