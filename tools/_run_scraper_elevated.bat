@echo off
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
%~dp0..\.venv\Scripts\python.exe -u tools\update_card_db.py > data\logs\scrape.log 2>&1
echo. >> data\logs\scrape.log
echo [exit code %errorlevel%] >> data\logs\scrape.log
pause
