@echo off

set scn_id=23
set stem=C:\Users\SwansonC\ECD\gridpath_ana\db
set benders=1
set database_it=1

cd %stem%
cd ./..

goto validate_scenarios

:create_database
python db/create_database.py --database %stem%\database%database_it%.db
:port_csvs
@echo on
echo importing csvs
@echo off
python db/utilities/port_csvs_to_db.py --database %stem%/database%database_it%.db --csv_location %stem%/csvs_test_examples
pause
:import_scenarios
@echo on
echo importing scenarios
@echo off
python db/utilities/scenario.py --database %stem%\database%database_it%.db --csv_path %stem%\csvs_test_examples/scenarios.csv
pause
:validate_scenarios
@echo on
echo validating scenarios
@REM @echo off
python gridpath/validate_inputs.py --database %stem%/database%database_it%.db --scenario_id %scn_id% --benders %benders%
pause
:run_end_to_end
@echo on
echo running end to end
@echo off
python gridpath/run_end_to_end.py --scenario_location C:\Users\SwansonC\ECD\gridpath_ana\examples --database %stem%/database%database_it%.db --scenario_id %scn_id% --solver gurobi --benders %benders%
pause
