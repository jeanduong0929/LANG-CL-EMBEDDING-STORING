@echo off

REM Check if pip3 is available
pip3 --version > nul 2>&1
if %errorlevel% == 0 (
    REM If pip3 exists, use python3
    python3 -m unittest src.test.test_lab
) else (
    REM If pip3 does not exist, use python
    python -m unittest src.test.test_lab
)
