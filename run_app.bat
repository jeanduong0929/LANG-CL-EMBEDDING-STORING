@echo off

REM Check if pip3 is available
pip3 --version > nul 2>&1
if %errorlevel% == 0 (
	REM If pip3 exists, use python3
	python3 src\main\app.py
) else (
	REM If pip3 does not exist, use python
	python src\main\app.py
)

