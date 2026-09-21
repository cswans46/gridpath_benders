from pyomo.environ import AbstractModel
from pyomo.core.expr.visitor import identify_variables,identify_components
import logging
from pyomo.environ import Expression, Param,Set,Var
from pyomo.core.expr.visitor import identify_variables, identify_components
import os
from pyomo.common.collections import ComponentSet

logger = logging.getLogger(__name__)

def define_linking_vars(m):

    investment_exprs = [
        "Capacity_MW",
        "Tx_Min_Capacity_MW",
        "Tx_Max_Capacity_MW",
        "TxNewLin_Capacity_MW",
        "Energy_Storage_Capacity_MWh",
    ]

    investment_vars = [
        "CARBON_CAP_ALLOCATION",
    ]

    if hasattr(m, "PERIOD_ENERGY_TARGET_ALLOCATION"):
        investment_vars = investment_vars + ["PERIOD_ENERGY_TARGET_ALLOCATION"]

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
        load_scenario_data,
        create_problem_instance,
        fix_variables
    )

    from gridpath.benders.subproblems import (
        add_slacks_to_subproblem
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

    inputs_path = os.path.join(
        scenario_directory,
        str(key.weather),
        str(key.hydro),
        str(key.availability),
        str(key.subproblem),
        str(key.stage),
        "inputs",
    )

    projects_path = os.path.join(
        inputs_path,
        "projects.tab",
    )

    # print(
    #     "\n--- BENDERS WORKER ---"
    #     f"\nPID: {os.getpid()}"
    #     f"\nkey: {key}"
    #     f"\ninputs_path: {inputs_path}"
    #     f"\nprojects_path: {projects_path}"
    #     f"\ninputs exists: {os.path.isdir(inputs_path)}"
    #     f"\nprojects exists: {os.path.isfile(projects_path)}"
    # )

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
        key.stage,
        temporal_subproblems=None
    )

    if expect_feasible:
        # print("Skipping feasibility cuts.")
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
    # if not parsed_arguments.quiet:
        # print("Creating problem instance for operational subproblems...")
    instance = create_problem_instance(model, scenario_data_benders)

    # print("\n--- Sets ---")
    # for s in instance.component_objects(Set):
    #     print(
    #         "SET:",
    #         s.name,
    #         "constructed:",
    #         s.is_constructed(),
    #         "size:",
    #         len(s) if s.is_constructed() else "NOT CONSTRUCTED",
    #     )
    #
    # print("\n--- Variables ---")
    # for var in instance.component_objects(Var, active=True):
    #     print(
    #         var.name,
    #         "indexed =", var.is_indexed(),
    #         "entries =", len(var) if var.is_indexed() else 1,
    #     )
    #     if var.is_indexed():
    #         for i in var:
    #             print("   ", i, "=", value(var[i], exception=False))
    #     else:
    #         print("   =", value(var, exception=False))
    #
    # print("\n--- Constraints ---")
    # for con in instance.component_objects(Constraint, active=True):
    #     print(
    #         con.name,
    #         "indexed =", con.is_indexed(),
    #         "entries =", len(con) if con.is_indexed() else 1,
    #     )
    #
    # print("\n--- Objectives ---")
    # for obj in instance.component_objects(Objective, active=True):
    #     print(obj.name)
    #     print("   expression:", obj.expr)
    #     print("   sense:", obj.sense)
    #
    # print("\n================== CHECKING CAPACITY SETS ====================")
    # print("PRJ_OPR_PRDS:")
    # for i in instance.PRJ_OPR_PRDS:
    #     print("   ", i)

    # print("\n================== EXPRESSION VARIABLE DEPENDENCIES ==================")
    #
    # for expr_component in instance.component_objects(
    #         Expression, active=True
    # ):
    #
    #     print(f"\n--- {expr_component.name} ---")
    #     print("indexed:", expr_component.is_indexed())
    #     print("constructed:", expr_component.is_constructed())
    #
    #     if expr_component.is_indexed():
    #
    #         # Indexed Expression
    #         for index in expr_component:
    #
    #             expr_data = expr_component[index]
    #             expr = expr_data.expr
    #
    #             variables = list(identify_variables(expr))
    #             params = list(identify_components(expr, {Param}))
    #
    #             print(f"\n{expr_data.name}[{index}]")
    #             print(f"  EXPR: {expr}")
    #
    #             if variables:
    #                 print("  VARIABLES:")
    #                 for var in variables:
    #                     print(f"    {var.name}")
    #             else:
    #                 print("  VARIABLES: NONE")
    #
    #             if params:
    #                 print("  PARAMS:")
    #                 for param in params:
    #                     print(f"    {param.name}")
    #             else:
    #                 print("  PARAMS: NONE")

        # else:
        #
        #     # Scalar Expression
        #     expr_data = expr_component
        #     expr = expr_data.expr
        #
        #     variables = list(identify_variables(expr))
        #     params = list(identify_components(expr, {Param}))
        #
        #     print(f"\n{expr_data.name}")
        #     print(f"  EXPR: {expr}")
        #
        #     if variables:
        #         print("  VARIABLES:")
        #         for var in variables:
        #             print(f"    {var.name}")
        #     else:
        #         print("  VARIABLES: NONE")
        #
        #     if params:
        #         print("  PARAMS:")
        #         for param in params:
        #             print(f"    {param.name}")
        #     else:
        #         print("  PARAMS: NONE")

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