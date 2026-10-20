@echo off
setlocal

set PYTHON_VER=3.11
set ENV_NAME=gridpath_benders
@REM  Directory where the license file is
set MOSEK_DIR=C:\Users\SwansonC\mosek\
@REM  Name of the license file within the directory
set MOSEK_LIC=LIF-3263.lic

goto activate_n_instantiate

:delete_environment

echo Deletion of existing environment has been requested...
call conda env remove -n %ENV_NAME% --yes >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Environment '%ENV_NAME%' did not exist or was already removed. Continuing...
)

:create_environment

echo Creating environment '%ENV_NAME%' with Python %PYTHON_VER%...
call conda create -n %ENV_NAME% python=%PYTHON_VER% --yes
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create environment. Exiting.
    goto :end
)

:activate_n_instantiate

echo Activating environment and installing GridPath...
call conda.bat activate %ENV_NAME%
pip install -e ".[all]"
call conda.bat activate %ENV_NAME%
set MOSEKLM_LICENSE_FILE=%MOSEK_DIR%%MOSEK_LIC%

:end
echo Done!
pause
endlocal