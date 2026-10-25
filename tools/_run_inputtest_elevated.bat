@echo off
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
%~dp0..\.venv\Scripts\python.exe -u tools\input_method_test.py > data\logs\input_test.log 2>&1
