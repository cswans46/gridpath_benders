"""Names of the Benders policy-allocation variables (shared, pyomo-free).

A policy allocation variable carries one representative day's share of a
period budget (a carbon cap, an energy target, ...). They are linking
variables: the master chooses them, the subproblem uses them as right-hand
sides.
"""

ALLOCATION_COMPONENTS = (
    "CARBON_CAP_ALLOCATION",
    "PERIOD_ENERGY_TARGET_ALLOCATION",
    "PERFORMANCE_STANDARD_MWH_ALLOCATION",
    "PERFORMANCE_STANDARD_MW_ALLOCATION",
    "BENDERS_CARBON_CREDIT_DEMAND_ALLOCATION",
    "BENDERS_CARBON_CREDIT_SUPPLY_ALLOCATION",
    "BENDERS_PROJECT_CARBON_CREDIT_ABSOLUTE_THRESHOLD_ALLOCATION",
    "TRANSMISSION_TARGET_POS_DIR_MIN_ALLOCATION",
    "TRANSMISSION_TARGET_POS_DIR_MAX_ALLOCATION",
    "TRANSMISSION_TARGET_NEG_DIR_MIN_ALLOCATION",
    "TRANSMISSION_TARGET_NEG_DIR_MAX_ALLOCATION",
    "FUEL_BURN_MIN_ABS_ALLOCATION",
    "FUEL_BURN_MAX_ABS_ALLOCATION",
    "CAP_FACTOR_MIN_ALLOCATION",
    "CAP_FACTOR_MAX_ALLOCATION",
    "GENERIC_POLICY_ALLOCATION",
)


def is_allocation_name(name):
    return any(
        name.startswith(prefix + "[") or name == prefix
        for prefix in ALLOCATION_COMPONENTS
    )
