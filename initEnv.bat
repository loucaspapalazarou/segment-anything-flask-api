@echo off
setlocal enabledelayedexpansion

python -m venv base

call ""base/Scripts/activate""

python.exe -m pip install --upgrade pip
pip install --no-cache-dir -r extra-requirements.txt
pip install --no-cache-dir -r requirements.txt
pip install ultralytics

pause