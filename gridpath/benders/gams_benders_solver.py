"""GAMS shell adapter for Benders master and operational subproblem solves.

The adapter keeps the existing Pyomo solver-like interface used throughout
GridPath's Benders code while passing the selected optimizer and its options
to the GAMS shell interface on every solve.
"""

from pyomo.environ import SolverFactory


class GamsSolverAdapter:
    """Solver-like wrapper that runs a named optimizer through Pyomo/GAMS."""

    name = "gams"

    def __init__(self, solver_options, executable=None):
        solver_options = dict(solver_options or {})
        self.solver_name = str(solver_options.pop("solver", "")).strip()
        self.backend_name = self.solver_name.lower()
        solver_options.pop("solver_name", None)
        if not self.solver_name:
            raise ValueError(
                "GAMS Benders requires solver_options.csv to specify the "
                "underlying optimizer in the 'solver' option (for example, "
                "solver,mosek)."
            )

        self._solver = SolverFactory(
            "gams",
            solver_io="shell",
            **({"executable": executable} if executable is not None else {}),
        )
        if self._solver is None or not self._solver.available(exception_flag=False):
            raise RuntimeError(
                "The Pyomo GAMS shell solver is not available. Check the GAMS "
                "installation and solver executable configuration."
            )

        # Base options from solver_options.csv. This dictionary is deliberately
        # mutable: Benders temporarily changes LP algorithm/logging options.
        self.options = {str(k): str(v) for k, v in solver_options.items()}

    def solve(self, instance, **kwargs):
        """Solve through GAMS, regenerating the optimizer option file each time."""
        options = dict(self.options)
        add_options = [
            "GAMS_MODEL.optfile = 1;",
            f"$onecho > {self.solver_name}.opt",
        ]
        add_options.extend(
            f"{option_name} {option_value}"
            for option_name, option_value in options.items()
        )
        add_options.append("$offecho")

        # Preserve any caller-supplied GAMS statements after the option-file
        # setup. The optimizer itself is always controlled by solver_options.
        caller_add_options = kwargs.pop("add_options", None)
        if caller_add_options:
            add_options.extend(caller_add_options)

        kwargs["solver"] = self.solver_name
        kwargs["add_options"] = add_options
        return self._solver.solve(instance, **kwargs)


def create_gams_solver(solver_options, executable=None):
    """Construct the GAMS adapter used by Benders master/workers."""
    return GamsSolverAdapter(solver_options, executable=executable)
