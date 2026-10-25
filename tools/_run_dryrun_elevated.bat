@echo off
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
%~dp0..\.venv\Scripts\python.exe -u tools\update_card_db.py --dry-run > data\logs\dryrun.log 2>&1
