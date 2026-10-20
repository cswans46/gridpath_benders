"""Regression tests for forwarding Benders solve context."""

import inspect
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from pyomo.environ import ConcreteModel, Objective, Var, maximize, value

from gridpath import run_scenario
from gridpath.auxiliary.dynamic_components import DynamicComponents
from gridpath.benders.planning import (
    BENDERS_THETA_SCALE,
    add_approximate_variable_cost,
)


class TestBendersSolveContext(unittest.TestCase):
    def test_stage_forwards_subproblems_and_context_to_solve(self):
        for multi_stage in (False, True):
            with self.subTest(multi_stage=multi_stage):
                with tempfile.TemporaryDirectory() as scenario_directory:
                    self.check_stage_context(scenario_directory, multi_stage)

    def check_stage_context(self, scenario_directory, multi_stage):
        parsed_arguments = run_scenario.parse_arguments(
            [
                "--scenario", "irrelevant",
                "--scenario_location", scenario_directory,
                "--benders", "1",
                "--quiet",
            ]
        )
        instance = ConcreteModel()
        instance.Build = Var(initialize=5)
        instance.NPV = Objective(expr=2 * instance.Build, sense=maximize)
        dynamic_components = DynamicComponents()
        results = SimpleNamespace(
            planning_results=SimpleNamespace(
                solver=SimpleNamespace(termination_condition="optimal"),
            ),
        )
        expected_keys = {
            run_scenario.ScenarioKey(
                weather=weather,
                hydro="1",
                availability="1",
                subproblem=subproblem,
                stage="1",
            )
            for weather in ("1", "2")
            for subproblem in ("1", "2")
        }

        def solve(
            instance,
            parsed_arguments,
            benders_subproblems=None,
            dynamic_components=None,
            multi_stage=None,
            prob_sol_files_directory=None,
        ):
            add_approximate_variable_cost(instance, benders_subproblems)
            return results

        with (
            mock.patch.object(
                run_scenario, "create_problem",
                return_value=(dynamic_components, instance),
            ),
            mock.patch.object(run_scenario, "solve", side_effect=solve) as solve_mock,
            mock.patch.object(run_scenario, "save_results") as save_results,
        ):
            run_scenario.run_optimization_for_subproblem_stage_benders(
                scenario_directory=scenario_directory,
                weather_iteration_directory=["1", "2"],
                hydro_iteration_directory=["1"],
                availability_iteration_directory=["1"],
                subproblem_directory=["1", "2"],
                stage_directory="1",
                multi_stage=multi_stage,
                parsed_arguments=parsed_arguments,
            )

        bound = inspect.signature(solve).bind(
            *solve_mock.call_args.args, **solve_mock.call_args.kwargs
        )
        self.assertIs(bound.arguments["instance"], instance)
        self.assertIs(bound.arguments["parsed_arguments"], parsed_arguments)
        self.assertEqual(
            bound.arguments["benders_subproblems"],
            {key: key for key in expected_keys},
        )
        self.assertIs(bound.arguments["dynamic_components"], dynamic_components)
        self.assertIs(bound.arguments["multi_stage"], multi_stage)
        self.assertEqual(
            bound.arguments["prob_sol_files_directory"],
            os.path.join(scenario_directory, "1", "prob_sol_files"),
        )
        self.assertEqual(set(instance.vTHETA), expected_keys)
        self.assertEqual(len(instance.ThetaLowerBound), len(expected_keys))
        self.assertFalse(instance.NPV.active)
        for key in expected_keys:
            instance.vTHETA[key].set_value(1)
        self.assertEqual(
            value(instance.MasterObjective),
            10 - BENDERS_THETA_SCALE * len(expected_keys),
        )
        self.assertIs(save_results.call_args.args[7], instance)
        self.assertIs(
            save_results.call_args.args[8], results.planning_results
        )


if __name__ == "__main__":
    unittest.main()
