@echo off

set scn_id=36
set stem=C:\Users\SwansonC\ECD\gridpath\db

cd %stem%
cd ./..

goto run_end_to_end

:create_database
python db/create_database.py --database %stem%\database.db

:port_csvs
python db/utilities/port_csvs_to_db.py --database %stem%/database.db --csv_location %stem%/csvs_test_examples

:import_scenarios
python db/utilities/scenario.py --database %stem%\database.db --csv_path %stem%\csvs_test_examples/scenarios.csv

:validate_scenarios
python gridpath/validate_inputs.py --database %stem%/database.db --scenario_id %scn_id%

:run_end_to_end
python gridpath/run_end_to_end.py --scenario_location C:/Users/SwansonC/ECD/gridpath/examples --database %stem%/database.db --scenario_id %scn_id% --solver gurobi
