# Copyright 2016-2023 Blue Marble Analytics LLC. All rights reserved.

from gridpath.auxiliary.auxiliary import load_subtype_modules
import csv
import os.path


def load_operational_type_modules(required_operational_modules):
    """
    Load a specified set of operational type modules
    :param required_operational_modules:
    :return: dictionary with the imported subtype modules
        {name of subtype module: Python module object}
    """
    return load_subtype_modules(
        required_subtype_modules=required_operational_modules,
        package="gridpath.project.operations.operational_types",
        required_attributes=[],
    )
    
def get_subproblem_projects(
    scenario_directory,
    weather_iteration,
    hydro_iteration,
    availability_iteration,
    subproblem,
    stage,
):
    """
    Get the projects included in the current subproblem's projects.tab.
    """
    projects_file = os.path.join(
        scenario_directory,
        weather_iteration,
        hydro_iteration,
        availability_iteration,
        subproblem,
        stage,
        "inputs",
        "projects.tab",
    )

    with open(projects_file, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row["project"] for row in reader}


def filter_project_input_rows(
    cursor,
    active_projects,
    period_index=None,
    periods=None,
):
    """
    Filter project-level input rows to projects in the current subproblem.

    Optionally filter a period-indexed input to the periods in the
    current subproblem.
    """
    rows = cursor.fetchall()

    filtered_rows = [
        row
        for row in rows
        if row[0] in active_projects
        and (
            period_index is None
            or int(str(row[period_index])[:4]) in periods
        )
    ]

    class FilteredRows:
        def __init__(self, description, rows):
            self.description = description
            self.rows = rows

        def fetchall(self):
            return self.rows

        def __iter__(self):
            return iter(self.rows)

    return FilteredRows(cursor.description, filtered_rows)

def resolve_op_type_rules(imported_operational_modules, rule_name, default_module):
    """
    Map each imported operational type module to its :code:`rule_name`
    function, falling back to the default op-type implementation in
    :code:`default_module` for modules that don't define it.

    Resolving these once at component-declaration time avoids per-index
    hasattr/getattr dispatch in rules that are called once per
    project-timepoint.
    """
    return {
        op_m: getattr(module, rule_name, getattr(default_module, rule_name))
        for op_m, module in imported_operational_modules.items()
    }
