# Define sets for capacity-related modules.
from pyomo.environ import Set
from gridpath.auxiliary.auxiliary import (
    get_required_subtype_modules,
    join_sets
)
from gridpath.project.capacity.common_functions import (
    load_project_capacity_type_modules,
)
from gridpath.project.capacity.capacity_types.common_methods import (
    spec_determine_inputs
)
from gridpath.auxiliary.dynamic_components import (
    capacity_type_operational_period_sets)

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
    m.GEN_SPEC_OPR_PRDS = Set(dimen=2, within=m.PROJECTS * m.PERIODS)
    
    # m.PRJ_OPR_PRDS = Set(
    #     dimen=2,
    #     within=m.PROJECTS * m.PERIODS,
    #     initialize=lambda mod: join_sets(
    #         mod,
    #         getattr(d, capacity_type_operational_period_sets),
    #     ),
    # )
    #
    # m.OPR_PRDS_BY_PRJ = Set(
    #     m.PROJECTS,
    #     initialize=lambda mod, project: operational_periods_by_project(
    #         prj=project,
    #         project_operational_periods=mod.PRJ_OPR_PRDS,
    #     ),
    # )
    #
    # m.PRJ_OPR_TMPS = Set(
    #     dimen=2,
    #     initialize=lambda mod: [
    #         (g, tmp)
    #         for g in mod.PROJECTS
    #         for p in mod.OPR_PRDS_BY_PRJ[g]
    #         for tmp in mod.TMPS_IN_PRD[p]
    #     ],
    # )
    #
    # m.OPR_PRJS_IN_TMP = Set(
    #     m.TMPS,
    #     initialize=op_gens_by_tmp,
    # )


# Set Rules
###############################################################################

def op_gens_by_tmp(mod):
    """
    Figure out which generators are operational in each timepoint.
    """
    tmp_to_projects = {}
    for g, tmp in mod.PRJ_OPR_TMPS:
        tmp_to_projects.setdefault(tmp, set()).add(g)

    for tmp in tmp_to_projects:
        tmp_to_projects[tmp] = list(tmp_to_projects[tmp])

    return tmp_to_projects

def operational_periods_by_project(prj, project_operational_periods):
    """ """
    return sorted(
        list(
            set(
                period
                for (project, period) in project_operational_periods
                if project == prj
            )
        )
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
    project_period_list, spec_params_dict = spec_determine_inputs(
        scenario_directory=scenario_directory,
        weather_iteration=weather_iteration,
        hydro_iteration=hydro_iteration,
        availability_iteration=availability_iteration,
        subproblem=subproblem,
        stage=stage,
        capacity_type="gen_spec",
    )

    data_portal.data()["GEN_SPEC_OPR_PRDS"] = project_period_list

    """ """
    required_capacity_modules = get_required_subtype_modules(
        scenario_directory=scenario_directory,
        weather_iteration=weather_iteration,
        hydro_iteration=hydro_iteration,
        availability_iteration=availability_iteration,
        subproblem=subproblem,
        stage=stage,
        which_type="capacity_type",
    )

    # Import needed capacity type modules
    imported_capacity_modules = load_project_capacity_type_modules(
        required_capacity_modules
    )

    for op_m in required_capacity_modules:
        if hasattr(imported_capacity_modules[op_m], "load_model_data"):
            imported_capacity_modules[op_m].load_model_data(
                m,
                d,
                data_portal,
                scenario_directory,
                weather_iteration,
                hydro_iteration,
                availability_iteration,
                subproblem,
                stage,
            )
