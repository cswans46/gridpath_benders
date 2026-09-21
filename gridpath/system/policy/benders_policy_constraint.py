"""
This module implements variables and constraints involved in constructing the policy constraint for
Benders to make the allocation of the period variables across subproblems.
"""

from pyomo.environ import Constraint, Var, NonNegativeReals,Set

def add_model_components(m,
    d,
    scenario_directory,
    weather_iteration,
    hydro_iteration,
    availability_iteration,
    subproblem,
    stage,
    ):

    m.SUBPROBLEM = Set(initialize=subproblem)

    # ============== Variables =======================

    if hasattr(m, "ENERGY_TARGET_ZONE_PERIODS_WITH_ENERGY_TARGET"):
        m.PERIOD_ENERGY_TARGET_ALLOCATION = Var(
            m.ENERGY_TARGET_ZONE_PERIODS_WITH_ENERGY_TARGET,
            m.SUBPROBLEM,
            within=NonNegativeReals,
        )

    if hasattr(m, "CARBON_CAP_ZONE_PERIODS_WITH_CARBON_CAP"):
        m.CARBON_CAP_ALLOCATION = Var(
            m.CARBON_CAP_ZONE_PERIODS_WITH_CARBON_CAP,
            m.SUBPROBLEM,
            within=NonNegativeReals,
        )

    print("subproblem argument:", subproblem)
    print("subproblem type:", type(subproblem))

    # ============ Constraint Rules ==================

    def carbon_cap_allocation_rule(mod, z, p):
        return (
                sum(
                    mod.CARBON_CAP_ALLOCATION[z, p, w]
                    for w in mod.SUBPROBLEM
                )
                <= mod.carbon_cap_target[z, p]
        )

    def period_energy_target_allocation_mwh_rule(mod, z, p):
        return (
                sum(
                    mod.PERIOD_ENERGY_TARGET_ALLOCATION[z, p, w]
                    for w in mod.SUBPROBLEM
                )
                >= mod.Period_Energy_Target[z, p]
        )

    # ================ Constraints ==================

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