# Copyright 2016-2023 Blue Marble Analytics LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script iterates over all modules required for a GridPath scenario and
calls their *write_model_inputs()* method, which queries the GridPath
database and writes the .tab input files to the scenario directory.

The main() function of this script can also be called with the
*gridpath_get_inputs* command when GridPath is installed.

Two mechanisms can restrict the run to part of the scenario's structure:

* ``--temporal_structure_csv_overwrite`` with
  ``--temporal_structure_csv_path`` REPLACES the database-derived
  scenario structure with the CSV's (columns: weather_iteration,
  hydro_iteration, availability_iteration, subproblem, stage; one row per
  subproblem/stage). Fully general -- any subset of cells, down to
  individual subproblems/stages. Every listed combination is validated to
  exist in the scenario's database-derived structure (combinations outside
  it have no input data and would otherwise fail late); the
  directory-layout flags are derived from the CSV's values (see
  get_scenario_structure_from_csv).
* The ``--single_draw WEATHER HYDRO AVAILABILITY`` option (0 for iteration
  levels the scenario doesn't use) SLICES the resolved structure to a
  single iteration draw. Use, for example, for re-materializing one draw of a
  scenario directory that was cleaned after import. The draw is checked to
  exist, and the scenario's directory-layout flags are preserved. The slice
  applies to whatever structure was resolved.

With either mechanism, the post-import cleanup marker (if present) is left
in place, since the rest of the tree is still in its cleaned state; solve
the regenerated part with run_scenario's ``--ignore_cleanup_marker``. Only
a full regeneration of the database-derived structure removes the marker.
"""

from argparse import ArgumentParser
import csv
import datetime
from multiprocessing import get_context
import os.path
import pandas as pd
import sys
import warnings
from functools import partial
import shutil
from pathlib import Path

from db.common_functions import connect_to_database
from gridpath.auxiliary.db_interface import get_scenario_id_and_name
from gridpath.common_functions import (
    determine_scenario_directory,
    create_directory_if_not_exists,
    get_db_parser,
    get_required_e2e_arguments_parser,
    get_temporal_structure_csv_overwrite_parser,
    get_get_inputs_parser,
    get_single_draw_parser,
    get_version_parser,
    ensure_empty_string,
    get_benders_flag_parser
)
from gridpath.auxiliary.module_list import determine_modules, load_modules
from gridpath.scenario_directory_cleanup import (
    clear_cleanup_marker,
    CLEANUP_MARKER_FILENAME,
)
from gridpath.auxiliary.scenario_chars import (
    build_single_draw_structure,
    get_diverging_layout_flags,
    OptionalFeatures,
    resolve_requested_draw,
    SubScenarios,
    get_scenario_structure_from_db,
    get_scenario_structure_from_csv,
    SolverOptions,
    ScenarioDirectoryStructure,
    validate_csv_structure_against_db,
)


def write_model_inputs(
    parsed_arguments,
    scenario_directory,
    scenario_structure,
    modules_to_use,
    scenario_id,
    subscenarios,
    db_path,
    n_parallel_subproblems,
    delete_prior_aux=True,
):
    """
    For each module, load the inputs from the database and write out the inputs
    into .tab files, which will be used to construct the optimization problem.

    :param parsed_arguments: needed for benders flag
    :param scenario_directory: local scenario directory
    :param scenario_structure: ScenarioStructure object with info on the
        weather/hydro iterations and subproblem/stage structure
    :param modules_to_use: list of imported modules (Python <class 'module'>
        objects)
    :param scenario_id: integer
    :param subscenarios: SubScenarios object with all subscenario info
    :param db_path: database connection
    :param n_parallel_subproblems: int; get inputs for subproblems in parallel
    :param delete_prior_aux: boolean; whether to delete the scenario-level
        auxiliary files first. Set to False when writing inputs for one
        iteration draw at a time (the per-draw E2E mode), where the
        scenario-level files are managed once for the whole run.

    :return:
    """
    # TODO: should probably delete the whole scenario directory first,
    #  as previous scenario with the same name could have had a different
    #  subproblem-stage structure
    # Delete auxiliary files that may have existed before to avoid phantom
    # files
    if delete_prior_aux:
        delete_prior_aux_files(scenario_directory=scenario_directory)

    # Get the directory structure from the scenario structure
    scenario_directory_structure = ScenarioDirectoryStructure(
        scenario_structure
    ).SCENARIO_DIRECTORY_STRUCTURE

    # Do a few checks on parallelization request
    if n_parallel_subproblems < 1:
        warnings.warn(
            "n_parallel_subproblem can't be 0. Solving without parallelization."
        )
        n_parallel_subproblems = 1

    # Check number of subproblems
    if scenario_structure.N_SUBPROBLEMS == 1:
        if n_parallel_subproblems > 1:
            warnings.warn(
                "GridPath WARNING: only a single subproblem in "
                "scenario. No parallelization possible."
            )
        n_parallel_subproblems = 1

    # If no parallelization requested, loop through the iterations
    # and subproblems
    if n_parallel_subproblems == 1:
        for weather_iteration_str in scenario_directory_structure.keys():
            for hydro_iteration_str in scenario_directory_structure[
                weather_iteration_str
            ].keys():
                for availability_iteration_str in scenario_directory_structure[
                    weather_iteration_str
                ][hydro_iteration_str].keys():
                    # We may have passed "empty_string" to avoid actual empty
                    # strings as dictionary keys; convert to actual empty
                    # strings here to pass to the directory creation methods
                    weather_iteration_str = ensure_empty_string(weather_iteration_str)
                    hydro_iteration_str = ensure_empty_string(hydro_iteration_str)
                    availability_iteration_str = ensure_empty_string(
                        availability_iteration_str
                    )

                    for subproblem_str in scenario_directory_structure[
                        weather_iteration_str
                    ][hydro_iteration_str][availability_iteration_str].keys():
                        for stage_str in scenario_directory_structure[
                            weather_iteration_str
                        ][hydro_iteration_str][availability_iteration_str][
                            subproblem_str
                        ]:

                            write_inputs(
                                parsed_arguments=parsed_arguments,
                                scenario_directory=scenario_directory,
                                weather_iteration_str=weather_iteration_str,
                                hydro_iteration_str=hydro_iteration_str,
                                availability_iteration_str=availability_iteration_str,
                                subproblem_str=subproblem_str,
                                stage_str=stage_str,
                                modules_to_use=modules_to_use,
                                scenario_id=scenario_id,
                                subscenarios=subscenarios,
                                db_path=db_path
                            )
    else:
        pool_data = []
        for weather_iteration_str in scenario_directory_structure.keys():
            for hydro_iteration_str in scenario_directory_structure[
                weather_iteration_str
            ].keys():
                for availability_iteration_str in scenario_directory_structure[
                    weather_iteration_str
                ][hydro_iteration_str].keys():
                    # We may have passed "empty_string" to avoid actual empty
                    # strings as dictionary keys; convert to actual empty
                    # strings here to pass to the directory creation methods
                    weather_iteration_str = ensure_empty_string(weather_iteration_str)
                    hydro_iteration_str = ensure_empty_string(hydro_iteration_str)
                    availability_iteration_str = ensure_empty_string(
                        availability_iteration_str
                    )

                    for subproblem_str in scenario_directory_structure[
                        weather_iteration_str
                    ][hydro_iteration_str][availability_iteration_str].keys():
                        for stage_str in scenario_directory_structure[
                            weather_iteration_str
                        ][hydro_iteration_str][availability_iteration_str][
                            subproblem_str
                        ]:
                            pool_data.append(
                                [
                                    scenario_directory,
                                    weather_iteration_str,
                                    hydro_iteration_str,
                                    availability_iteration_str,
                                    subproblem_str,
                                    stage_str,
                                    modules_to_use,
                                    scenario_id,
                                    subscenarios,
                                    db_path,
                                ]
                            )

        pool_data = tuple(pool_data)
        # Pool must use spawn to work properly on Linux
        pool = get_context("spawn").Pool(n_parallel_subproblems)
        worker = partial(
            get_inputs_for_subproblem_pool,
            parsed_arguments=parsed_arguments,
        )

        pool.map(worker, pool_data)
        pool.close()
        pool.join()


def write_inputs(
    parsed_arguments,
    scenario_directory,
    weather_iteration_str,
    hydro_iteration_str,
    availability_iteration_str,
    subproblem_str,
    stage_str,
    modules_to_use,
    scenario_id,
    subscenarios,
    db_path
):

    loaded_modules = load_modules(modules_to_use,parsed_arguments)

    inputs_directory = os.path.join(
        scenario_directory,
        weather_iteration_str,
        hydro_iteration_str,
        availability_iteration_str,
        subproblem_str,
        stage_str,
        "inputs",
    )

    if not os.path.exists(inputs_directory):
        os.makedirs(inputs_directory)

    # Delete input files that may have existed before to avoid
    # phantom inputs
    delete_prior_inputs(inputs_directory=inputs_directory)

    # Write model input .tab files for each of the loaded_modules if
    # appropriate. Note that all input files are saved in the
    # input_directory, even the non-temporal inputs that are not
    # dependent on the subproblem or stage. This simplifies the file
    # structure at the expense of unnecessarily duplicating
    # non-temporal input files such as projects.tab.

    conn = connect_to_database(db_path=db_path)

    # Only putting Operational here because this is the only one that matters
    # at this nesting level (i.e. the investment .tab files will be aggregated from the
    # individual scenarios below in main().
    if parsed_arguments.benders:
        loaded_modules = (
                loaded_modules["Investment"]
                + loaded_modules["Operational"]
        )
        # Drop duplicates
        loaded_modules = list(dict.fromkeys(loaded_modules))

    for m in loaded_modules:
        if hasattr(m, "write_model_inputs"):
            m.write_model_inputs(
                scenario_directory=scenario_directory,
                scenario_id=scenario_id,
                subscenarios=subscenarios,
                weather_iteration=weather_iteration_str,
                hydro_iteration=hydro_iteration_str,
                availability_iteration=availability_iteration_str,
                subproblem=subproblem_str,
                stage=stage_str,
                conn=conn,
            )

    conn.commit()
    conn.close()


def get_inputs_for_subproblem_pool(pool_datum,parsed_arguments):
    """
    Helper function to easily pass to pool.map if running subproblems in
    parallel
    """
    [
        scenario_directory,
        weather_iteration_str,
        hydro_iteration_str,
        availability_iteration_str,
        subproblem_str,
        stage_str,
        modules_to_use,
        scenario_id,
        subscenarios,
        db_path,
    ] = pool_datum

    write_inputs(
        parsed_arguments=parsed_arguments,
        scenario_directory=scenario_directory,
        weather_iteration_str=weather_iteration_str,
        hydro_iteration_str=hydro_iteration_str,
        availability_iteration_str=availability_iteration_str,
        subproblem_str=subproblem_str,
        stage_str=stage_str,
        modules_to_use=modules_to_use,
        scenario_id=scenario_id,
        subscenarios=subscenarios,
        db_path=db_path,
    )


def write_scenario_level_files(
    scenario_directory,
    conn,
    scenario_id,
    scenario_name,
    scenario_structure,
    optional_features,
    subscenarios,
    solver_options,
    feature_list,
):
    """
    Write the scenario-level (draw-independent) files: the multi-stage flag,
    the features list, the scenario description, the units file, the solver
    options file, and the linked-subproblems map.
    """
    write_multi_stage_flag(scenario_directory, scenario_structure.STAGE_FLAG)
    # Save the list of optional features to a file (will be used to
    # determine modules without database connection)
    write_features_csv(scenario_directory=scenario_directory, feature_list=feature_list)
    # Write full scenario description
    write_scenario_description(
        scenario_directory=scenario_directory,
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        optional_features=optional_features,
        subscenarios=subscenarios,
    )
    # Write the units used for all metrics
    write_units_csv(scenario_directory, conn)
    # Write the solver options file if needed
    write_solver_options(
        scenario_directory=scenario_directory, solver_options=solver_options
    )
    # Write the subproblem linked timepoints map file if needed
    write_linked_subproblems_map(scenario_directory, conn, subscenarios)


def delete_prior_aux_files(scenario_directory):
    """
    Delete all auxiliary files that may exist in the scenario directory
    :param scenario_directory: the scenario directory
    :return:
    """
    prior_aux_files = [
        "features.csv",
        "scenario_description.csv",
        "solver_options.csv",
        "linked_subproblems_map.csv",
    ]

    for f in prior_aux_files:
        if f in os.listdir(scenario_directory):
            os.remove(os.path.join(scenario_directory, f))


def delete_prior_inputs(inputs_directory):
    """
    Delete all .tab files that may exist in the specified directory
    :param inputs_directory: local directory where .tab files are saved
    :param scenario_directory: the scenario directory
    :return:
    """
    # First, in case a scenario changes, folders in the main scenario
    # directory must also be deleted

    prior_input_tab_files = [
        f for f in os.listdir(inputs_directory) if f.endswith(".tab")
    ]

    for f in prior_input_tab_files:
        os.remove(os.path.join(inputs_directory, f))


def parse_arguments(args):
    """
    :param args: the script arguments specified by the user
    :return: the parsed known argument values (<class 'argparse.Namespace'>
    Python object)

    Parse the known arguments.
    """
    parser = ArgumentParser(
        add_help=True,
        parents=[
            get_db_parser(),
            get_required_e2e_arguments_parser(),
            get_temporal_structure_csv_overwrite_parser(),
            get_get_inputs_parser(),
            get_single_draw_parser(
                context_help="Only this draw's inputs are written -- e.g. "
                "to re-materialize one draw of a scenario whose directory "
                "was cleaned after import, without regenerating the whole "
                "tree. The cleanup marker is left in place; solve the "
                "regenerated draw with run_scenario's "
                "--ignore_cleanup_marker."
            ),
            get_version_parser(),
            get_benders_flag_parser()
        ],
    )

    parsed_arguments = parser.parse_known_args(args=args)[0]

    return parsed_arguments


def write_features_csv(scenario_directory, feature_list):
    """
    Write the features.csv file that will be used to determine which
    GridPath modules to include
    :return:
    """
    with open(
        os.path.join(scenario_directory, "features.csv"), "w", newline=""
    ) as features_csv_file:
        writer = csv.writer(features_csv_file, delimiter=",", lineterminator="\n")

        # Write header
        writer.writerow(["features"])

        for feature in feature_list:
            writer.writerow([feature])


def write_scenario_description(
    scenario_directory, scenario_id, scenario_name, optional_features, subscenarios
):
    """

    :param scenario_directory:
    :param scenario_id:
    :param scenario_name:
    :param optional_features:
    :param subscenarios:
    :return:
    """
    feature_list = optional_features.get_all_available_features()
    subscenario_list = subscenarios.get_all_available_subscenarios()

    with open(
        os.path.join(scenario_directory, "scenario_description.csv"), "w", newline=""
    ) as scenario_description_file:
        writer = csv.writer(
            scenario_description_file, delimiter=",", lineterminator="\n"
        )

        # Scenario ID and scenario name
        writer.writerow(["scenario_id", scenario_id])
        writer.writerow(["scenario_name", scenario_name])

        # Optional features
        for feature in feature_list:
            writer.writerow(
                [
                    "of_{}".format(feature),
                    getattr(optional_features, "OF_{}".format(feature.upper())),
                ]
            )

        # Subscenarios
        for subscenario in subscenario_list:
            writer.writerow([subscenario, getattr(subscenarios, subscenario.upper())])


def write_units_csv(scenario_directory, conn):
    """

    :param scenario_directory:
    :param conn:
    :return:
    """
    sql = """
        SELECT metric, unit
        FROM mod_units;
        """
    df = pd.read_sql(sql, conn)
    df.to_csv(os.path.join(scenario_directory, "units.csv"), index=False)


def write_solver_options(scenario_directory, solver_options):
    """
    Write solver options to the scenario directory.

    GAMS uses a three-column format:
        solver_name, solver_option_name, solver_option_value

    Other solver interfaces retain the legacy two-column format:
        solver_name, solver_option
    """
    solver_name = solver_options.SOLVER_NAME
    options = solver_options.SOLVER_OPTIONS

    if solver_name is None or options is None:
        return

    solver_name = str(solver_name).strip()

    if not solver_name:
        raise ValueError("Solver name cannot be empty.")

    output_file = os.path.join(
        scenario_directory, "solver_options.csv"
    )

    with open(output_file, "w", newline="") as solver_options_file:
        writer = csv.writer(
            solver_options_file,
            delimiter=",",
            lineterminator="\n",
        )

        if solver_name.lower() == "gams":
            # GAMS: three-column format
            writer.writerow([
                "solver_name",
                "solver_option_name",
                "solver_option_value",
            ])

            for option_name, option_value in options.items():
                writer.writerow([
                    solver_name,
                    option_name,
                    option_value,
                ])

        else:
            # Other solvers: retain legacy two-column format
            writer.writerow(["solver_name", solver_name])

            for option_name, option_value in options.items():
                writer.writerow([
                    option_name,
                    option_value,
                ])


def write_multi_stage_flag(scenario_directory, stage_flag):
    """
    Write the multi-stage flag to the scenario directory.
    """
    if stage_flag:
        with open(
            os.path.join(scenario_directory, "multi_stage_flag.txt"),
            "w",
            newline="",
        ) as stage_flag_file:
            stage_flag_file.write(str(stage_flag))


def write_linked_subproblems_map(scenario_directory, conn, subscenarios):
    sql = """
        SELECT subproblem_id as subproblem, stage_id as stage, timepoint, 
        subproblem_id + 1 as subproblem_to_link, 
        linked_timepoint, number_of_hours_in_timepoint
        FROM inputs_temporal
        WHERE linked_timepoint IS NOT NULL
        AND temporal_scenario_id = ?;
        """

    df = pd.read_sql(sql=sql, con=conn, params=(subscenarios.TEMPORAL_SCENARIO_ID,))

    # Only write this file if there are any linked problems
    if not df.empty:
        df.to_csv(
            os.path.join(scenario_directory, "linked_subproblems_map.csv"), index=False
        )

def write_inputs_for_benders_planning_problem(
    scenario_directory,
    scenario_structure,
):
    """
    Collect all realization-specific .tab files into a single
    investment/planning inputs directory.

    Each .tab file is aggregated by filename across all
    weather/hydro/availability/subproblem/stage combinations.

    For example:

        scenario/
            1/2010/1/1/1/inputs/foo.tab
            1/2010/1/2/1/inputs/foo.tab
            2/2011/2/1/1/inputs/foo.tab

    becomes:

        scenario/
            inputs/foo.tab

    containing the rows from all realization-specific foo.tab files.
    """

    delete_prior_aux_files(
        scenario_directory=scenario_directory
    )

    # Get the directory structure from the scenario structure
    scenario_directory_structure = ScenarioDirectoryStructure(
        scenario_structure
    ).SCENARIO_DIRECTORY_STRUCTURE

    # Planning/investment input directory
    investment_inputs_directory = os.path.join(
        scenario_directory,
        "inputs",
    )
    # Write the planning problem directory.
    os.makedirs(
        investment_inputs_directory,
        exist_ok=True,
    )

    # Keep track of which output files have already had
    # their headers written.
    headers_written = set()
    seen_rows = {}

    for weather_iteration_str in scenario_directory_structure.keys():

        for hydro_iteration_str in scenario_directory_structure[
            weather_iteration_str
        ].keys():

            for availability_iteration_str in scenario_directory_structure[
                weather_iteration_str
            ][hydro_iteration_str].keys():

                weather_iteration_str = ensure_empty_string(
                    weather_iteration_str
                )

                hydro_iteration_str = ensure_empty_string(
                    hydro_iteration_str
                )

                availability_iteration_str = ensure_empty_string(
                    availability_iteration_str
                )

                for subproblem_str in scenario_directory_structure[
                    weather_iteration_str
                ][hydro_iteration_str][
                    availability_iteration_str
                ].keys():

                    for stage_str in scenario_directory_structure[
                        weather_iteration_str
                    ][hydro_iteration_str][
                        availability_iteration_str
                    ][subproblem_str]:

                        # Directory containing the realization-specific
                        # input files
                        source_inputs_directory = os.path.join(
                            scenario_directory,
                            weather_iteration_str,
                            hydro_iteration_str,
                            availability_iteration_str,
                            subproblem_str,
                            stage_str,
                            "inputs",
                        )

                        if not os.path.isdir(
                            source_inputs_directory
                        ):
                            continue

                        # Find all .tab files for this realization
                        for fname in os.listdir(
                            source_inputs_directory
                        ):

                            if not fname.endswith(".tab"):
                                continue

                            source_file = os.path.join(
                                source_inputs_directory,
                                fname,
                            )

                            destination_file = os.path.join(
                                investment_inputs_directory,
                                fname,
                            )

                            # Initialize set for this particular .tab file
                            if fname not in seen_rows:
                                seen_rows[fname] = set()

                            # Read source file
                            with open(
                                source_file,
                                "r",
                                newline="",
                            ) as source:

                                reader = csv.reader(
                                    source,
                                    delimiter="\t",
                                )

                                # Read header
                                header = next(reader, None)

                                if header is None:
                                    continue

                                # If this is the first time we've seen this file,
                                # create it and write the header.
                                if fname not in headers_written:

                                    with open(
                                            destination_file,
                                            "w",
                                            newline="",
                                    ) as destination:

                                        writer = csv.writer(
                                            destination,
                                            delimiter="\t",
                                            lineterminator="\n",
                                        )

                                        writer.writerow(header)

                                        for row in reader:

                                            row_tuple = tuple(row)

                                            if row_tuple not in seen_rows[fname]:
                                                writer.writerow(row)
                                                seen_rows[fname].add(row_tuple)

                                    headers_written.add(fname)

                                else:

                                    with open(
                                            destination_file,
                                            "a",
                                            newline="",
                                    ) as destination:

                                        writer = csv.writer(
                                            destination,
                                            delimiter="\t",
                                            lineterminator="\n",
                                        )

                                        for row in reader:

                                            row_tuple = tuple(row)

                                            if row_tuple not in seen_rows[fname]:
                                                writer.writerow(row)
                                                seen_rows[fname].add(row_tuple)

def main(args=None):
    """

    :return:
    """
    # Retrieve DB location and scenario_id and/or name from args
    if args is None:
        args = sys.argv[1:]

    parsed_arguments = parse_arguments(args=args)

    db_path = parsed_arguments.database
    scenario_id_arg = parsed_arguments.scenario_id
    scenario_name_arg = parsed_arguments.scenario
    scenario_location = parsed_arguments.scenario_location

    temporal_structure_csv_overwrite = parsed_arguments.temporal_structure_csv_overwrite
    temporal_structure_csv_path = parsed_arguments.temporal_structure_csv_path

    if (
        temporal_structure_csv_overwrite is False
        and temporal_structure_csv_path is not None
    ):
        raise ValueError(
            "You must turn on temporal_structure_csv_overwrite "
            "explicitly to overwrite the temporal structure from "
            f"{temporal_structure_csv_path}."
        )

    # Close the connection on any raise: a connection left open by an error
    # keeps the database file locked on Windows
    conn = connect_to_database(db_path=db_path)
    try:
        c = conn.cursor()

        if not parsed_arguments.quiet:
            print(
                "Getting inputs, started on {}... (connected to database {})".format(
                    datetime.datetime.now(), db_path
                )
            )

        scenario_id, scenario_name = get_scenario_id_and_name(
            scenario_id_arg=scenario_id_arg,
            scenario_name_arg=scenario_name_arg,
            c=c,
            script="get_scenario_inputs",
        )

        # Determine scenario directory and create it if needed
        scenario_directory = determine_scenario_directory(
            scenario_location=scenario_location, scenario_name=scenario_name
        )
        create_directory_if_not_exists(directory=scenario_directory)

        # Get scenario characteristics (features, scenario_id, subscenarios, subproblems)
        # TODO: it seems these fail silently if empty; we may want to implement
        #  some validation
        optional_features = OptionalFeatures(conn=conn, scenario_id=scenario_id)
        subscenarios = SubScenarios(conn=conn, scenario_id=scenario_id)
        if temporal_structure_csv_overwrite:
            scenario_structure = get_scenario_structure_from_csv(
                temporal_structure_csv_path
            )
            db_structure = get_scenario_structure_from_db(
                conn=conn, scenario_id=scenario_id
            )
            validate_csv_structure_against_db(
                csv_structure=scenario_structure, db_structure=db_structure
            )
            # If the CSV implies a different directory layout than the
            # scenario's own structure, the regenerated files won't line up
            # with a tree (or archive tarballs) generated from the full
            # structure: refuse on a cleaned/archived directory, where such
            # mixing is exactly what the user would be setting up, and warn
            # loudly otherwise (a stand-alone run in a fresh directory is
            # self-consistent at either layout)
            diverging_layout_flags = get_diverging_layout_flags(
                csv_structure=scenario_structure, db_structure=db_structure
            )
            if diverging_layout_flags:
                divergence_str = "; ".join(diverging_layout_flags)
                if os.path.exists(
                    os.path.join(scenario_directory, CLEANUP_MARKER_FILENAME)
                ):
                    raise ValueError(
                        f"The temporal structure CSV implies a different "
                        f"directory layout than the scenario's own "
                        f"structure: {divergence_str}. The regenerated "
                        f"files would not line up with this "
                        f"cleaned/archived scenario directory's original "
                        f"layout or its archive tarballs. Use whole "
                        f"subproblem/stage sets in the CSV, or "
                        f"--single_draw for one draw."
                    )
                else:
                    print(
                        f"WARNING: the temporal structure CSV implies a "
                        f"different directory layout than the scenario's "
                        f"own structure: {divergence_str}. This is "
                        f"self-consistent for a stand-alone run in this "
                        f"directory, but do not mix it with files "
                        f"generated from the scenario's full structure "
                        f"(including archive tarballs)."
                    )
        else:
            scenario_structure = get_scenario_structure_from_db(
                conn=conn, scenario_id=scenario_id
            )
        solver_options = SolverOptions(conn=conn, scenario_id=scenario_id)

        # If a single iteration draw was requested, narrow the structure to
        # that draw (e.g. to re-materialize one draw of a cleaned scenario
        # directory for debugging)
        single_draw_requested = parsed_arguments.single_draw is not None
        if single_draw_requested:
            draw = resolve_requested_draw(
                scenario_structure=scenario_structure,
                weather_iteration=parsed_arguments.single_draw[0],
                hydro_iteration=parsed_arguments.single_draw[1],
                availability_iteration=parsed_arguments.single_draw[2],
            )
            scenario_structure = build_single_draw_structure(scenario_structure, *draw)
            if not parsed_arguments.quiet:
                print(
                    f"Writing inputs for the single draw with weather "
                    f"iteration {draw[0]}, hydro iteration {draw[1]}, "
                    f"availability iteration {draw[2]} (0 = level not used)."
                )

        # Determine requested features and use this to determine what modules
        # to load for GridPath
        feature_list = optional_features.get_active_features()

        # Figure out which modules to use and load the modules
        modules_to_use = determine_modules(
            parsed_arguments=parsed_arguments,features=feature_list, multi_stage=scenario_structure.STAGE_FLAG
        )

        # Get appropriate inputs from database and write the .tab file model
        # inputs
        write_model_inputs(
            parsed_arguments=parsed_arguments,
            scenario_directory=scenario_directory,
            scenario_structure=scenario_structure,
            modules_to_use=modules_to_use,
            scenario_id=scenario_id,
            subscenarios=subscenarios,
            db_path=db_path,
            n_parallel_subproblems=int(parsed_arguments.n_parallel_get_inputs),
            delete_prior_aux=not single_draw_requested,
        )

        # Write the planning problem map
        if parsed_arguments.benders:
            write_inputs_for_benders_planning_problem(
                scenario_directory,
                scenario_structure,
            )

        write_scenario_level_files(
            scenario_directory=scenario_directory,
            conn=conn,
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            scenario_structure=scenario_structure,
            optional_features=optional_features,
            subscenarios=subscenarios,
            solver_options=solver_options,
            feature_list=feature_list,
        )

        # The scenario directory contents have been regenerated, so remove
        # the post-import cleanup marker if one was present (it blocks
        # run_scenario and import_scenario_results while the directory is in
        # a cleaned state). NOT done when only part of the structure was
        # regenerated -- a single draw or a temporal-structure-CSV subset --
        # since the rest of the tree is still in its cleaned state and the
        # marker must keep guarding it (run_scenario can solve the
        # regenerated subset with --ignore_cleanup_marker)
        full_structure_regenerated = not (
            single_draw_requested or temporal_structure_csv_overwrite
        )
        if full_structure_regenerated:
            clear_cleanup_marker(scenario_directory=scenario_directory)
        elif (
            os.path.exists(os.path.join(scenario_directory, CLEANUP_MARKER_FILENAME))
            and not parsed_arguments.quiet
        ):
            print(
                "The cleanup marker was left in place: only part of the "
                "scenario's structure was regenerated and the rest of the "
                "scenario directory is still cleaned. To solve the "
                "regenerated part, pass --ignore_cleanup_marker to "
                "run_scenario."
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
