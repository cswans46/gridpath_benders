@echo off

set scn_id=18
set stem=C:\Users\SwansonC\ECD\gridpath_ana\db
set benders=1
set database_it=2
set n_parallel_get_inputs=1
set solver=mosek
cd %stem%
cd ./..

goto run_end_to_end
:create_database
python db/create_database.py --database %stem%\database%database_it%.db --n_parallel_get_inputs=%n_parallel_get_inputs% --n_parallel_solve=%n_parallel_get_inputs%
:port_csvs
@echo on
echo importing csvs
@echo off
python db/utilities/port_csvs_to_db.py --database %stem%/database%database_it%.db --csv_location %stem%/csvs_test_examples --n_parallel_get_inputs=%n_parallel_get_inputs% --n_parallel_solve=%n_parallel_get_inputs%
pause
:import_scenarios
@echo on
echo importing scenarios
@echo off
python db/utilities/scenario.py --database %stem%\database%database_it%.db --csv_path %stem%\csvs_test_examples/scenarios.csv --n_parallel_get_inputs=%n_parallel_get_inputs% --n_parallel_solve=%n_parallel_get_inputs%
pause
:validate_scenarios
@echo on
echo validating scenarios
@REM @echo off
python gridpath/validate_inputs.py --database %stem%/database%database_it%.db --scenario_id %scn_id% --benders %benders% --n_parallel_get_inputs=%n_parallel_get_inputs% --n_parallel_solve=%n_parallel_get_inputs%
pause
:run_end_to_end
@echo on
echo running end to end
@echo off
python gridpath/run_end_to_end.py --scenario_location C:\Users\SwansonC\ECD\gridpath_ana\examples --database %stem%/database%database_it%.db --scenario_id %scn_id% --solver %solver% --benders %benders% --n_parallel_get_inputs=%n_parallel_get_inputs% --n_parallel_solve=%n_parallel_get_inputs%
@REM  > output.log 2>&1
@REM --load_gurobi_solution
goto:eof
:load_problem_file
@REM python gridpath\auxiliary\plugins\gurobi_solve.py --problem_file_directory C:\Users\SwansonC\ECD\gridpath_ana\examples\single_stage_prod_cost_w_startup_limit\3\prob_sol_files\ --problem_file_name problem_file.lp
pause
:eof