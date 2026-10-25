@echo off
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
%~dp0..\.venv\Scripts\python.exe -u tools\dbg_desktop2.py > data\logs\dbg7.log 2>&1
