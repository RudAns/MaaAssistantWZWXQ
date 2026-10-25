@echo off
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
%~dp0..\.venv\Scripts\python.exe -u tools\dbg_menu.py > data\logs\dbg_menu.log 2>&1
