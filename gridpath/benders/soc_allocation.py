
"""
Shared definitions for chaining storage state of charge (SOC) across Benders
operational subproblems, e.g. daily subproblems for a pumped hydro project that
must balance over the whole period.

Used by BOTH the master (system.policy.benders_policy_constraint) and the
operational subproblems (project.operations.operational_types.stor), so it may
only rely on components from modules that are loaded in both models (the
"Sets" list in module_organisation.define_benders_modules): PROJECTS,
operational_type, balancing_type_project, PERIODS and PRJ_OPR_PRDS.
"""

from pyomo.environ import NonNegativeReals, Set, Var


STOR_SOC_CHAIN_BT = "year"
STOR_SOC_ALLOCATION_UNIT_MWH = 1000.0

STOR_START_SOC_VAR = "STOR_START_SOC_ALLOCATION"
STOR_END_SOC_VAR = "STOR_END_SOC_ALLOCATION"


def stor_soc_allocation_targets(mod):
    """
    Return (project, balancing type, period) for each storage project
    that chains its SOC across subproblems and is operational in that period.
    """
    return [
        (prj, STOR_SOC_CHAIN_BT, period)
        for prj in mod.PROJECTS
        if mod.operational_type[prj] == "stor"
        and mod.balancing_type_project[prj] == STOR_SOC_CHAIN_BT
        for period in mod.PERIODS
        if (prj, period) in mod.PRJ_OPR_PRDS
    ]


def add_stor_soc_allocation_components(mod, subproblem):
    """
    Add generic SOC allocation components to a Benders operational model.

    Must be called after the core project and period components are
    declared, but before storage operational constraints are constructed.
    """
    subproblem = str(subproblem)

    mod.STOR_SOC_ALLOCATION_TARGETS = Set(
        dimen=3,
        initialize=stor_soc_allocation_targets,
    )

    def _index_init(m):
        for prj, bt, hz in m.STOR_SOC_ALLOCATION_TARGETS:
            yield prj, bt, hz, subproblem

    mod.STOR_SOC_ALLOCATION_INDEX = Set(
        dimen=4,
        initialize=_index_init,
    )

    mod.add_component(
        STOR_START_SOC_VAR,
        Var(mod.STOR_SOC_ALLOCATION_INDEX, within=NonNegativeReals),
    )

    mod.add_component(
        STOR_END_SOC_VAR,
        Var(mod.STOR_SOC_ALLOCATION_INDEX, within=NonNegativeReals),
    )
